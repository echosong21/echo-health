"""Retrieval evaluation runner.

Reads the frozen golden set, runs each question through the retrieval stage,
judges hit / partial / miss against the expected sources, and writes per-question
results plus an aggregate baseline to disk.

Frozen parameters (must stay identical across runs so numbers are comparable):
    top_k = 5, min_score = 0.0, mode = single

Usage:
    python evals/run_eval.py
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

from src import retrieval as rt     # 与 src/pipeline.py 共用同一套检索实现

load_dotenv(ROOT / ".env")

# ── frozen parameters ────────────────────────────────────────────────
# ⚠️ 集合名来自领域配置（domains/*.yaml），不写死 —— 换领域时本文件无需改动
from src import domain as domain_mod  # noqa: E402

try:
    _DOMAIN = domain_mod.load()
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(f"领域配置加载失败（检查 domains/*.yaml）：{exc}") from exc

COLLECTION = _DOMAIN["collection"]
TOP_K = 5
MIN_SCORE = 0.0
PER_FORM = 3

GOLDEN = Path(__file__).parent / "golden_questions.json"
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

emb = OpenAIEmbeddings(
    model=os.getenv("EMBED_MODEL"),
    base_url=os.getenv("EMBED_BASE_URL"),
    api_key=os.getenv("API_KEY"),
    check_embedding_ctx_length=False,
)
client = MilvusClient(uri=os.getenv("MILVUS_URL"))


def judge(expected_sources, hits):
    """Return (verdict, matched_sources, hit_count, expected_count).

    A source entry matches when it appears as a substring of a hit's source path
    (e.g. "faq/" matches any FAQ file; a full filename matches that file).
    """
    hit_sources = [h["source"] for h in hits]
    matched = [e for e in expected_sources if any(e in hs for hs in hit_sources)]
    n_exp = len(expected_sources)
    n_hit = len(matched)
    if n_hit == 0:
        verdict = "miss"
    elif n_hit == n_exp:
        verdict = "hit"
    else:
        verdict = "partial"
    return verdict, matched, n_hit, n_exp


# BM25 索引懒加载（只有 hybrid 模式需要，避免其他模式白等 2 秒）
BM25_CACHE_KEY = "935426c666faa59d"
_bm25_bundle = None


def _get_bm25_bundle():
    global _bm25_bundle
    if _bm25_bundle is None:
        import pickle

        from src import bm25 as bm
        cache = bm.CACHE_DIR / f"vectors_{BM25_CACHE_KEY}.pkl"
        items = pickle.loads(cache.read_bytes())["items"]
        bm25, docs, _stats, form_index = bm.build_or_load(items, BM25_CACHE_KEY)
        print(f"[hybrid] BM25 预热 {bm.warmup(bm25)} ms（含 jieba 词典加载）")
        _bm25_bundle = (bm25, docs, form_index)
    return _bm25_bundle


def run_one(item, mode, per_form, collection):
    t0 = time.perf_counter()
    qvec = emb.embed_query(item["question"])
    embed_ms = (time.perf_counter() - t0) * 1000

    if mode == "hybrid":
        raw, stage = rt.retrieve_hybrid(client, qvec, item["question"], collection,
                                        per_form, _get_bm25_bundle())
        search_ms = stage["dense_ms"] + stage["sparse_ms"] + stage["fuse_ms"]
        ordered = raw                       # 已按 RRF 排好，不能再按分数重排
    elif mode == "multi":
        raw, _per_form_ms, search_ms = rt.retrieve_dense_multi(client, qvec, collection, per_form)
        ordered = sorted(raw, key=lambda x: x["score"], reverse=True)
    else:  # single
        raw, search_ms = rt.retrieve_dense_single(client, qvec, collection, TOP_K)
        ordered = sorted(raw, key=lambda x: x["score"], reverse=True)

    # 相似度过滤：混合模式的 score 是 RRF 分数（尺度不同），故用 vec_score
    if mode == "hybrid":
        kept = [h for h in ordered if (h.get("vec_score") or 0.0) >= MIN_SCORE]
    else:
        kept = [h for h in ordered if h["score"] >= MIN_SCORE]

    verdict, matched, n_hit, n_exp = judge(item["expected_source"], kept)

    # ★ 名次类指标：命中率只看"在不在结果里"，看不到排序质量；
    #   而融合策略（如 RRF）恰恰只改排序不改集合 ⇒ 必须单独记录名次。
    srcs = [h["source"] for h in kept]
    top1_source = srcs[0] if srcs else None
    top1_expected = any(e in (top1_source or "") for e in item["expected_source"])
    best_rank = next(
        (i for i, s in enumerate(srcs, 1)
         if any(e in s for e in item["expected_source"])),
        None,
    )

    return {
        "id": item["id"],
        "form": item["form"],
        "question": item["question"],
        "difficulty": item.get("difficulty"),
        "mode": mode,
        "expected_source": item["expected_source"],
        "verdict": verdict,
        "matched_sources": matched,
        "n_hit": n_hit,
        "n_expected": n_exp,
        "top1": kept[0]["score"] if kept else None,
        "top1_source": top1_source,
        "top1_expected": top1_expected,
        "best_rank": best_rank,
        "hit_sources": srcs,
        "embed_ms": round(embed_ms, 1),
        "search_ms": round(search_ms, 1),
        "total_ms": round(embed_ms + search_ms, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "multi", "hybrid"], default="single")
    p.add_argument("--per-form", type=int, default=PER_FORM)
    p.add_argument("--collection", default=COLLECTION,
                   help="要评测的集合名（默认 law_docs）；换集合用于切分/索引对比实验")
    args = p.parse_args()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    mode = args.mode
    per_form = args.per_form
    collection = args.collection

    print(f"golden set: {len(golden)} questions | collection={collection} "
          f"| top_k={TOP_K} min_score={MIN_SCORE} mode={mode}")
    print("-" * 96)


    results = []
    for item in golden:
        r = run_one(item, mode, per_form, collection)
        results.append(r)
        mark = {"hit": "OK  ", "partial": "PART", "miss": "MISS"}[r["verdict"]]
        flag = "T1" if r["top1_expected"] else (f"r{r['best_rank']}" if r["best_rank"] else "--")
        print(
            f"{r['id']:>3} {mark} {r['form']:<9} {flag:>3} ({r['n_hit']}/{r['n_expected']}) "
            f"{r['total_ms']:>7.1f}ms  {r['question'][:26]}"
        )

    stamp = datetime.now().strftime("%Y-%m-%d")
    out = RESULT_DIR / f"results_{stamp}_{collection}_{mode}_pf{per_form}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    hit_like = sum(1 for r in results if r["verdict"] in ("hit", "partial"))
    by_form = defaultdict(lambda: [0, 0])
    for r in results:
        by_form[r["form"]][1] += 1
        if r["verdict"] in ("hit", "partial"):
            by_form[r["form"]][0] += 1

    lat = [r["total_ms"] for r in results]
    verdicts = Counter(r["verdict"] for r in results)

    # ★ 名次类指标：命中率对"排序变化"不敏感（只要还在结果集里就算命中），
    #   而融合策略恰恰只改排序 ⇒ 必须单独看名次，否则看不出融合的真实效果。
    top1_ok = sum(1 for r in results if r["top1_expected"])
    ranks = [r["best_rank"] for r in results if r["best_rank"]]
    mrr = round(sum(1.0 / k for k in ranks) / len(ranks), 4) if ranks else 0.0

    summary = {
        "date": stamp,
        "params": {
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "mode": mode,
            "per_form": per_form,
            "embed_model": os.getenv("EMBED_MODEL"),
            "collection": collection,
        },
        "n_questions": total,
        "verdicts": dict(verdicts),
        "overall_hit_rate": round(hit_like / total, 4),
        "by_form_hit_rate": {k: round(v[0] / v[1], 4) for k, v in sorted(by_form.items())},
        "latency_ms": {"mean": round(sum(lat) / len(lat), 1), "max": round(max(lat), 1)},
        "top1_expected_rate": round(top1_ok / total, 4),
        "mrr": mrr,
    }

    print("-" * 96)
    print(f"总体命中率 : {hit_like}/{total} = {summary['overall_hit_rate']:.1%}")
    print("分形态命中率:")
    for form, rate in summary["by_form_hit_rate"].items():
        n_ok, n_all = by_form[form]
        print(f"  {form:<10} {n_ok}/{n_all} = {rate:.1%}")
    print(f"延迟       : 均值 {summary['latency_ms']['mean']}ms / 最大 {summary['latency_ms']['max']}ms")
    print(f"★期待来源居首: {top1_ok}/{total} = {summary['top1_expected_rate']:.1%}    MRR = {mrr}")
    print(f"判定分布   : {dict(verdicts)}")
    print()
    print(f"结果落盘   : {out}")
    summary_path = RESULT_DIR / f"summary_{stamp}_{collection}_{mode}_pf{per_form}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总落盘   : {summary_path}")

    misses = [r for r in results if r["verdict"] == "miss"]
    if misses:
        print()
        print(f"未命中 {len(misses)} 题（需写失败归因）:")
        for r in misses:
            print(f"  {r['id']} [{r['form']}] {r['question'][:32]} | 期望 {r['expected_source']}")


if __name__ == "__main__":
    main()
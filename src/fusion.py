"""融合层：把多路召回的结果合成一个排序。

**为什么需要融合**
    混合检索有两条召回路径（向量 + 稀疏），各自会给出一个候选列表。直接拼接没有意义
    （两边的分数尺度完全不同），必须有一个把它们合成单一排序的策略。

**两种策略（本模块都实现，便于对照）**

    `rrf_fuse` —— Reciprocal Rank Fusion（默认）
        只用**排名**：score(d) = Σ_i 1 / (k + rank_i(d))
        · 不需要分数归一化 ⇒ 跨检索器的尺度差异被天然吸收
        · 几乎没有可调参数（只有 k，常取 60）

    `weighted_sum_fuse` —— 加权求和（作为对照实现）
        必须先**归一化**（余弦分 0–1 vs BM25 无界分，尺度差距可能有几十倍），
        然后按权重相加。
        · 归一化方法本身就是一个新的待调参数（min-max / z-score / 截断…）
        · 权重是第二个待调参数

**核心取舍（R7 必答）**：RRF 用"更少的自由度"换"同等甚至更好的效果"。
多一个归一化方法、多一组权重，就多一批需要标定的参数，而收益并不明确。

用法：
    python src/fusion.py                    # 自测：同一查询下对比 dense / sparse / 两种融合
    python src/fusion.py -q "你的问题"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from domain_rag_core import RRF_K, doc_key, rrf_fuse, route_weight  # noqa: E402,F401


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def weighted_sum_fuse(
    route_docs: dict[str, list[dict]],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """加权求和（**对照实现**，不是默认策略）。

    必须先归一化：余弦分在 0–1，而 BM25 分数是无界的（实测同一查询下可达 46），
    不归一化直接相加，结果完全由 BM25 主导。

    返回格式与 `rrf_fuse` 一致，便于并排比较。
    """
    weights = weights or {route: 1.0 for route in route_docs}
    scores: dict[tuple, float] = {}
    ranks: dict[tuple, dict[str, int]] = {}
    docs: dict[tuple, dict] = {}

    for route, items in route_docs.items():
        if not items:
            continue
        norm = _minmax([d["score"] for d in items])
        w = weights.get(route, 1.0)
        for rank, (doc, s) in enumerate(zip(items, norm), start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + w * s
            ranks.setdefault(key, {})[route] = rank
            docs.setdefault(key, doc)

    fused = [
        {"key": key, "doc": docs[key], "score": round(score, 6), "ranks": ranks[key]}
        for key, score in scores.items()
    ]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


# ── 自测：同一查询下，dense / sparse / 两种融合的对比 ──────────────────
def _self_test(question: str, top_k: int, cache_key: str) -> int:
    import pickle

    from langchain_openai import OpenAIEmbeddings
    from pymilvus import MilvusClient

    from config import settings
    from src import bm25 as bm25_mod
    from src import domain as domain_mod

    collection = domain_mod.load()["collection"]   # 自测也跟随领域配置

    print(f"查询：{question}\n")

    # 路 1：向量召回
    emb = OpenAIEmbeddings(
        model=settings.EMBED_MODEL, base_url=settings.EMBED_BASE_URL,
        api_key=settings.API_KEY, check_embedding_ctx_length=False,
    )
    client = MilvusClient(uri=settings.MILVUS_URL)
    hits = client.search(
        collection_name=collection,
        data=[emb.embed_query(question)],
        limit=top_k,
        output_fields=["text", "source", "chunk_index"],
    )[0]
    dense = [
        {
            "score": round(h["distance"], 4),
            "source": h["entity"]["source"],
            "chunk_index": h["entity"]["chunk_index"],
            "text": h["entity"]["text"],
        }
        for h in hits
    ]

    # 路 2：稀疏召回
    cache_path = bm25_mod.CACHE_DIR / f"vectors_{cache_key}.pkl"
    items = pickle.loads(cache_path.read_bytes())["items"]
    bm25, docs, _stats, _form_index = bm25_mod.build_or_load(items, cache_key)
    sparse, sparse_ms, _tokens = bm25_mod.search(question, bm25, docs, top_k=top_k)

    print(f"【向量召回 top-{len(dense)}】")
    for i, d in enumerate(dense, 1):
        print(f"  {i:2d}. {d['score']:>7.4f}  {d['source']}  #{d['chunk_index']}")
    print(f"\n【稀疏召回 top-{len(sparse)}】（{sparse_ms} ms）")
    for i, d in enumerate(sparse, 1):
        print(f"  {i:2d}. {d['score']:>7.3f}  {d['source']}  #{d['chunk_index']}")

    route_docs = {"dense": dense, "sparse": sparse}
    rrf = rrf_fuse(route_docs)
    ws = weighted_sum_fuse(route_docs)

    print(f"\n【RRF 融合 top-{top_k}】（k={RRF_K}）")
    for i, r in enumerate(rrf[:top_k], 1):
        tag = " ".join(f"{k}#{v}" for k, v in sorted(r["ranks"].items()))
        print(f"  {i:2d}. {r['score']:.6f}  {r['doc']['source']}  #{r['doc']['chunk_index']}"
              f"   [{tag}]")
    print(f"\n【加权求和融合 top-{top_k}】（归一化后等权）")
    for i, r in enumerate(ws[:top_k], 1):
        tag = " ".join(f"{k}#{v}" for k, v in sorted(r["ranks"].items()))
        print(f"  {i:2d}. {r['score']:.4f}  {r['doc']['source']}  #{r['doc']['chunk_index']}"
              f"   [{tag}]")

    print("\n两种融合的 top-1 是否一致：", doc_key(rrf[0]["doc"]) == doc_key(ws[0]["doc"]))
    print("说明：`[dense#n sparse#m]` 表示该块在两路里各自的名次 ⇒ 可看出是谁把它顶上来。")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="多路召回融合自测")
    ap.add_argument("-q", "--question", default="偷越国（边）境罪判几年？")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cache-key", default="935426c666faa59d")
    args = ap.parse_args()
    sys.exit(_self_test(args.question, args.top_k, args.cache_key))
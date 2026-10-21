"""稀疏检索层：BM25 索引与检索。

**为什么需要它**
    向量召回对**语义**敏感，但对**精确词**不敏感 —— 术语、法条编号、专有名词这类
    "关键词必须命中"的查询，稀疏检索（BM25）比向量更可靠。两者互补，
    组合起来才叫"混合检索"。

**⚠️ 中文分词（本模块最容易静默失败的地方）**
    `rank_bm25` 默认按空格切词。中文没有空格边界 ⇒ 整句话会变成**一个 token**
    ⇒ BM25 完全失效，**而且不报任何错**。所以必须显式接入 jieba。
    这一条已写进 `TROUBLESHOOTING.md`。

**索引来源**
    直接复用建库时的分块结果（向量缓存里的 `items`），保证 BM25 与向量库看到的是
    **同一批块、同一套切分** —— 两条召回路径的候选才能对齐。

**持久化（台账 D10）**
    BM25 是**纯内存索引**，2.3 万块每次进程启动都重建要花可观时间。
    因此落盘 pickle，并把「加载耗时」与「检索耗时」**分开记录** ——
    否则加载时间会被误算进检索延迟，把结论带偏。

用法：
    python src/bm25.py                       # 自测：构建/加载索引并跑一个查询
    python src/bm25.py -q "你的问题" --top-k 10
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402  （rank_bm25 已依赖 numpy）

try:
    import jieba  # noqa: F401
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # 依赖缺失时给出明确指引，而不是让调用方看到难懂的 ImportError
    print(f"[bm25] 缺少依赖（{exc.name}）⇒ 请先安装：pip install --no-cache-dir jieba rank_bm25",
          file=sys.stderr)
    raise

# 索引缓存目录（与实验工具一致，放在仓库外）
CACHE_DIR = Path(__file__).resolve().parents[2] / "_exp" / ".cache"


def tokenize(text: str) -> list[str]:
    """中文分词 —— BM25 的关键前提。

    两件事：
      ① **用 jieba 而不是按空格切**：中文没有空格边界，空格分词会把整句变成
         一个 token，BM25 随即失效（且不报错）。
      ② **丢掉纯标点/符号**：`（）`、`？` 这类片段不携带语义，进入索引只会
         增加噪声并稀释 IDF。

    ⚠️ 已知局限：jieba 默认词典不含法律专有名词，因此"偷越国（边）境罪"这类术语
    会被切碎（实测切成 `偷` / `越国` / `境罪判`）。当前仍能召回正确结果，
    但精度有损失。改进方向是加载自定义词典（见 `DESIGN.md` 的 BM25 一节）。
    """
    tokens: list[str] = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if not tok:
            continue
        # 纯标点/符号（不含任何字母或数字）⇒ 丢弃
        if not any(ch.isalnum() for ch in tok):
            continue
        tokens.append(tok)
    return tokens


def build_tokenized(items: list[dict]) -> list[list[str]]:
    t0 = time.perf_counter()
    tokenized: list[list[str]] = []
    for i, it in enumerate(items, 1):
        tokenized.append(tokenize(it["text"]))
        if i % 5000 == 0:
            print(f"[bm25] 分词 {i}/{len(items)}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    print(f"[bm25] 分词完成：{len(tokenized)} 块，耗时 {time.perf_counter() - t0:.0f}s")
    return tokenized


def _build_form_index(items: list[dict]) -> dict[str, list[int]]:
    """预计算「形态 → 该形态的块下标」。

    混合检索需要**按形态**取稀疏候选（与向量侧对齐）。如果每次查询都现场筛选
    2.3 万个块，光筛选就比打分还贵，因此在建索引时一次性算好。
    """
    index: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        form = it["source"].split("/")[0]
        index.setdefault(form, []).append(i)
    return index


def warmup(bm25: BM25Okapi) -> float:
    """预热：把**所有惰性初始化点**都先跑一遍。

    ⚠️ 实测教训（2026-10-20）：只预热 BM25 是**不够的**。里面有两处独立的惰性开销：

      ① **jieba 词典加载**（实测约 1 s）—— 发生在第一次 `tokenize()` 时；
      ② **BM25 打分侧的初始化** —— 发生在第一次 `get_scores()` 时。

    只做 ② 时，第一个真实查询仍要等 **1061 ms**（①在那一刻才被触发）。
    因此这里两步都做。**预热要覆盖全部初始化点，漏一个就等于没预热。**

    返回预热总耗时（毫秒），便于记录。
    """
    t0 = time.perf_counter()
    tokenize("预热")            # ① 触发 jieba 词典加载
    bm25.get_scores(["预热"])   # ② 触发打分侧初始化
    return round((time.perf_counter() - t0) * 1000, 1)


def build_or_load(
    items: list[dict],
    cache_key: str,
    cache_dir: Path = CACHE_DIR,
    rebuild: bool = False,
) -> tuple[BM25Okapi, list[dict], dict, dict[str, list[int]]]:
    """构建（或从缓存加载）BM25 索引。

    返回 `(bm25, items, stats, form_index)`：
      · `stats` —— 含 `load_s` / `fit_s` 等计时。**加载耗时必须单列**，
        否则会被误算进检索延迟；
      · `form_index` —— 「形态 → 块下标」的预计算表，供按形态取稀疏候选用。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"bm25_{cache_key}.pkl"

    if path.exists() and not rebuild:
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        load_s = time.perf_counter() - t0
        cached_items = payload["items"]
        if len(cached_items) != len(items):
            print(f"[bm25] 缓存块数不匹配（{len(cached_items)} vs {len(items)}）"
                  f"⇒ 重建索引")
        else:
            t0 = time.perf_counter()
            bm25 = BM25Okapi(payload["tokenized"])
            fit_s = time.perf_counter() - t0
            print(f"[bm25] 已从缓存加载：{path.name}"
                  f"（读取 {load_s:.1f}s｜建索引对象 {fit_s:.1f}s）")
            return bm25, cached_items, {
                "cache_hit": True, "load_s": round(load_s, 2),
                "fit_s": round(fit_s, 2), "n_docs": len(cached_items),
            }, _build_form_index(cached_items)

    tokenized = build_tokenized(items)
    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    fit_s = time.perf_counter() - t0
    with open(path, "wb") as f:
        pickle.dump({"items": items, "tokenized": tokenized}, f)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[bm25] 已构建并落盘：{path.name}（{size_mb:.1f} MB｜建索引 {fit_s:.1f}s）")
    return bm25, items, {
        "cache_hit": False, "fit_s": round(fit_s, 2),
        "n_docs": len(items), "cache_mb": round(size_mb, 1),
    }, _build_form_index(items)


def search(
    query: str,
    bm25: BM25Okapi,
    items: list[dict],
    top_k: int = 10,
    forms: list[str] | None = None,
    form_index: dict[str, list[int]] | None = None,
) -> tuple[list[dict], float, list[str]]:
    """稀疏召回。

    返回 `(hits, latency_ms, tokens)`：
      · hits —— `[{score, source, chunk_index, text}, ...]`，按分数降序
      · latency_ms —— **仅检索耗时**（不含索引加载，加载已单列）
      · tokens —— 查询的分词结果，便于排查"为什么没召回"

    `forms` 给定时**只在该形态内排序**（与向量侧"按形态取 per_form 条"对齐）——
    这样两条召回路径的候选构成才可比，融合才有意义。
    """
    t0 = time.perf_counter()
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    if forms:
        if form_index is None:
            form_index = _build_form_index(items)
        cand = np.concatenate([np.asarray(form_index[f], dtype=np.int64)
                               for f in forms if f in form_index])
    else:
        cand = np.arange(len(scores), dtype=np.int64)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    if len(cand) == 0:
        return [], latency_ms, tokens

    cand_scores = scores[cand]
    k = min(top_k, len(cand))
    order = (np.argpartition(cand_scores, -k)[-k:] if k < len(cand)
             else np.arange(len(cand)))
    idx = cand[order[np.argsort(cand_scores[order])[::-1]]]

    return [
        {
            "score": round(float(scores[i]), 4),
            "source": items[i]["source"],
            "chunk_index": items[i]["chunk_index"],
            "text": items[i]["text"],
        }
        for i in idx
    ], latency_ms, tokens


# ── 自测 ────────────────────────────────────────────────────────────
def _self_test(question: str, top_k: int, cache_key: str, rebuild: bool) -> int:
    cache_path = CACHE_DIR / f"vectors_{cache_key}.pkl"
    if not cache_path.exists():
        print(f"[自测] 找不到向量缓存：{cache_path}")
        print("      先用 _exp/exp_build.py 建一次库，或指定 --cache-key")
        return 2

    print(f"向量缓存：{cache_path.name}")
    items = pickle.loads(cache_path.read_bytes())["items"]
    print(f"候选块数：{len(items)}")

    # 演示"不分词会怎样"——这是本模块最值得记住的坑
    demo = items[0]["text"][:60]
    print(f"\n分词对比（同一段文本）：")
    print(f"  按空格切：{len(demo.split())} 个 token  ← 中文几乎全挤在一起")
    print(f"  jieba 切：{len(tokenize(demo))} 个 token  ← 正常")
    print(f"  两者示例：{tokenize(demo)[:8]}")

    bm25, docs, stats, _form_index = build_or_load(items, cache_key, rebuild=rebuild)
    print(f"\n索引统计：{stats}")

    print(f"\n查询：{question}")
    warm_ms = warmup(bm25)
    print(f"预热耗时：{warm_ms} ms（⚠️ 首次调用有惰性初始化开销，服务启动时应先预热）")
    hits, latency_ms, tokens = search(question, bm25, docs, top_k=top_k)
    print(f"查询分词：{tokens}")
    print(f"检索耗时：{latency_ms} ms（⚠️ 不含索引加载，加载须单列）")
    print(f"\n稀疏召回 top-{len(hits)}：")
    for i, h in enumerate(hits, 1):
        print(f"  {i:2d}. {h['score']:>8.3f}  {h['source']}  chunk#{h['chunk_index']}")
        print(f"      {h['text'][:56].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BM25 稀疏检索自测")
    ap.add_argument("-q", "--question", default="偷越国（边）境罪判几年？")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cache-key", default="935426c666faa59d",
                    help="向量缓存的 key（决定用哪批块建索引）")
    ap.add_argument("--rebuild", action="store_true", help="忽略 BM25 缓存，强制重建")
    args = ap.parse_args()
    sys.exit(_self_test(args.question, args.top_k, args.cache_key, args.rebuild))
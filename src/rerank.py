"""重排层：用专用 rerank 模型对召回结果做二次排序。

**为什么单独一层**
    向量召回保证的是"语义相近"，不保证"最相关排在最前"。重排模型专门做这件事，
    比让生成模型给候选打分更便宜、也更稳定（生成模型的判断还受提示词影响）。

**降级设计（关键）**
    重排是外部依赖，它的失败不该拖垮整条检索链路。因此本模块在**配置缺失**或
    **调用失败**时一律返回 `None`，由调用方回退到原始顺序 —— 保证"加了重排"
    不会让服务变得更脆弱。

**⚠️ 分数语义**
    接口返回的 `relevance_score` 是**本次请求内的相对分**（官方文档明确说明不可跨请求
    比较）。因此它只能用于同一批候选内部的排序，**不能当作绝对质量分**记录或横向比较。

配置（均在 `.env`，缺任一项即自动降级为不重排）：
    DASHSCOPE_API_KEY / WORKSPACE_ID / RERANK_MODEL / RERANK_REGION

用法：
    python src/rerank.py                     # 自测：单问跑通并打印重排前后差异
    python src/rerank.py -q "你的问题" --top-n 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import httpx  # noqa: E402

from config import settings  # noqa: E402

# 只把前 N 条候选送给重排模型 —— 重排按文档条数计费，截断是主要的成本控制手段。
RERANK_TOP_N = 10

_ENDPOINT_TMPL = (
    "https://{workspace}.{region}.maas.aliyuncs.com"
    "/api/v1/services/rerank/text-rerank/text-rerank"
)


def is_available(force: bool = False) -> bool:
    """重排是否可用。

    两道门：① **开关**（`RERANK_ENABLED`，默认关）② 配置齐全。
    开关默认关闭是因为实测该接口有约 3.6s 的固定延迟（见下），开启会让端到端翻倍。
    `force=True` 供自测/离线评测绕过开关。
    """
    if not force and not settings.RERANK_ENABLED:
        return False
    return bool(settings.DASHSCOPE_API_KEY and settings.WORKSPACE_ID)


def rerank(
    query: str,
    documents: list[str],
    top_n: int = RERANK_TOP_N,
    timeout: float = 30.0,
    force: bool = False,
) -> dict | None:
    """对候选文档做二次排序。

    返回 dict（含 results / latency_ms / total_tokens / model），
    或 **None** 表示"重排未启用 / 不可用 / 失败" —— 调用方须回退到原始顺序。

    ⚠️ 实测延迟特征（2026-10-20，见 design 文档）：延迟约 **3.6s 且与输入条数无关**
    （送 1 条 3582ms、送 10 条 3666ms），也与模型选择、冷启动无关。
    因此**不能靠减少 top_n 来控制延迟**。
    """
    if not documents:
        return None
    if not is_available(force=force):
        if not settings.RERANK_ENABLED:
            print("[rerank] 未启用（RERANK_ENABLED=false）⇒ 跳过重排", file=sys.stderr)
        else:
            print("[rerank] 未配置 WORKSPACE_ID / DASHSCOPE_API_KEY ⇒ 跳过重排",
                  file=sys.stderr)
        return None

    candidates = documents[:top_n]
    url = _ENDPOINT_TMPL.format(
        workspace=settings.WORKSPACE_ID, region=settings.RERANK_REGION
    )
    payload = {
        "model": settings.RERANK_MODEL,
        "input": {"query": query, "documents": candidates},
        # 不传 top_n ⇒ 返回全部输入的重排序结果（调用方需要完整顺序）
        "parameters": {"return_documents": False},
    }

    t0 = time.perf_counter()
    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 —— 任何失败都降级，不让检索链路挂掉
        print(f"[rerank] 调用失败，回退原顺序：{type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    raw_results = (data.get("output") or {}).get("results") or []
    if not raw_results:
        print(f"[rerank] 响应中没有 results，回退原顺序：{str(data)[:200]}",
              file=sys.stderr)
        return None

    return {
        "results": [
            {"index": r["index"], "score": round(r["relevance_score"], 6)}
            for r in raw_results
        ],
        "model": settings.RERANK_MODEL,
        "latency_ms": latency_ms,
        "total_tokens": (data.get("usage") or {}).get("total_tokens"),
        "n_candidates": len(candidates),
    }


# ── 自测：单问跑通，打印「重排前 vs 重排后」的顺序差异 ──────────────────
def _self_test(question: str, top_k: int) -> int:
    from langchain_openai import OpenAIEmbeddings
    from pymilvus import MilvusClient

    print(f"问题：{question}")
    print(f"重排模型：{settings.RERANK_MODEL}｜开关 RERANK_ENABLED="
          f"{settings.RERANK_ENABLED}｜配置可用：{is_available(force=True)}")
    print("-" * 78)

    emb = OpenAIEmbeddings(
        model=settings.EMBED_MODEL,
        base_url=settings.EMBED_BASE_URL,
        api_key=settings.API_KEY,
        check_embedding_ctx_length=False,
    )
    client = MilvusClient(uri=settings.MILVUS_URL)
    from src import domain as domain_mod
    collection = domain_mod.load()["collection"]   # 自测也跟随领域配置
    hits = client.search(
        collection_name=collection,
        data=[emb.embed_query(question)],
        limit=top_k,
        output_fields=["text", "source"],
    )[0]

    candidates = [h["entity"]["text"] for h in hits]
    before = [(h["entity"]["source"], round(h["distance"], 4)) for h in hits]

    print("【重排前】向量相似度顺序：")
    for i, (src, score) in enumerate(before, 1):
        print(f"  {i:2d}. {score}  {src}")

    out = rerank(question, candidates, force=True)
    if out is None:
        print("\n【重排后】不可用或失败 ⇒ 回退原顺序（这正是降级设计要保证的）")
        return 1

    print(f"\n【重排后】{out['model']} 顺序"
          f"（延迟 {out['latency_ms']}ms｜tokens {out['total_tokens']}）：")
    for i, r in enumerate(out["results"], 1):
        src = before[r["index"]][0]
        vec_score = before[r["index"]][1]
        moved = r["index"] + 1 - i
        arrow = "  " if moved == 0 else (f"↑{moved}" if moved > 0 else f"↓{-moved}")
        print(f"  {i:2d}. {r['score']}  {src}  (向量分 {vec_score}, 位移 {arrow})")

    print("\n顺序变化：")
    old = [f"{i+1}" for i in range(len(before))]
    new = [f"{r['index']+1}" for r in out["results"]]
    print(f"  前：{' '.join(old)}")
    print(f"  后：{' '.join(new)}")
    changed = sum(1 for i, r in enumerate(out["results"]) if r["index"] != i)
    print(f"  位置变化条数：{changed}/{len(out['results'])}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="重排层自测")
    ap.add_argument("-q", "--question", default="订婚后退婚，彩礼能不能要回来？")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    sys.exit(_self_test(args.question, args.top_k))
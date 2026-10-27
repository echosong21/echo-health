"""对比不同 `min_score` 下的 token 消耗与上下文条数。

**为什么需要这个脚本**

`run_eval.py` 是**检索评测**：它只调 `retrieve_dense_multi` / `retrieve_hybrid`，
**不调用 LLM**，所以天然没有 token 概念。这也意味着
「提高 min_score 能省多少 token」这个问题，用 eval 永远回答不了。

本脚本补的是**端到端**这一环：真实调用 `rag_query_multi`（含生成），
记录每题实际消耗的 `input_tokens` / `output_tokens` 与进入 prompt 的上下文条数。

**成本**：17 题 × 2 档 ≈ 34 次 LLM 调用，约 ¥0.08。
**这钱该花** —— 没有它，B2 的成本治理就只能写「无法验证」。

**口径**

- 与 `EVAL_REPORT` 的 5.2b（`traces.jsonl` 在线口径）不同：这里是**固定 17 题**，
  可比性强但题面单一。
- 与 5.2（evals 离线口径）也不同：那个不含生成。

**已知限制**：LLM 输出有随机性（`temperature` 未固定），
故 `output_tokens` 会波动；`input_tokens` 的下降才是 `min_score` 的直接效果。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "evals" / "golden_questions.json"
THRESHOLDS = (0.0, 0.5)


def run_one_round(pipeline, golden: list[dict], min_score: float) -> list[dict]:
    rows: list[dict] = []
    total = len(golden)
    for idx, item in enumerate(golden, 1):
        try:
            out = pipeline.rag_query_multi(
                item["question"], per_form=3, min_score=min_score, top_k=5,
            )
            usage = out.get("usage") or {}
            row = {
                "id": item["id"],
                "n_ctx": len(out.get("contexts") or []),
                "in_tok": int(usage.get("input_tokens") or 0),
                "out_tok": int(usage.get("output_tokens") or 0),
            }
            print(
                f"  [ms={min_score}] {idx:>2}/{total} {item['id']:<3} "
                f"ctx={row['n_ctx']} in={row['in_tok']:>4} out={row['out_tok']:>3}",
                flush=True,
            )
        except Exception as exc:                       # noqa: BLE001
            print(f"  [ms={min_score}] {idx:>2}/{total} {item['id']:<3} ERROR {type(exc).__name__}", flush=True)
            row = {"id": item["id"], "n_ctx": 0, "in_tok": 0, "out_tok": 0,
                   "error": f"{type(exc).__name__}: {exc}"[:80]}
        rows.append(row)
    return rows


def summarize(rows: list[dict], min_score: float) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"min_score": min_score, "n_ok": 0}
    n = len(ok)
    return {
        "min_score": min_score,
        "n_ok": n,
        "n_failed": len(rows) - n,
        "ctx_mean": sum(r["n_ctx"] for r in ok) / n,
        "in_mean": sum(r["in_tok"] for r in ok) / n,
        "out_mean": sum(r["out_tok"] for r in ok) / n,
        "in_total": sum(r["in_tok"] for r in ok),
        "out_total": sum(r["out_tok"] for r in ok),
    }


def preflight() -> bool:
    """跑之前先探活，避免「每题都先烧 embedding 才发现 Milvus 连不上」。

    ⚠️ 这条检查是踩出来的：第一次跑本脚本时隧道已断，
    34 题各调一次 embedding 后才在检索阶段失败 ——
    也就是说 34 次 API 调用全白花了。`build_index.py` 早就有同样的
    probe-before-embed 保护，新脚本必须照做。
    """
    try:
        from pymilvus import MilvusClient

        from config import settings          # 根目录的 config.py，不是 src.settings

        MilvusClient(uri=settings.MILVUS_URL, timeout=5)
        print(f"[OK] Milvus 可达：{settings.MILVUS_URL}")
        return True
    except Exception as exc:                            # noqa: BLE001
        print(f"[ERR] Milvus 不可达（{type(exc).__name__}），已中止 —— "
              f"未消耗任何 embedding 调用。\n"
              f"      请先建立隧道：ssh -L 19530:127.0.0.1:19530 root@<ECS地址>")
        return False


def main() -> int:
    if not preflight():
        return 2

    from src import pipeline

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"golden: {len(golden)} 题 | 阈值档位: {THRESHOLDS}\n")

    all_rows: dict[float, list[dict]] = {}
    for ms in THRESHOLDS:
        print(f"--- min_score = {ms} ---")
        all_rows[ms] = run_one_round(pipeline, golden, ms)
        print()

    stats = [summarize(all_rows[ms], ms) for ms in THRESHOLDS]

    print("## 汇总\n")
    print("| min_score | 有效题 | 上下文均条数 | input_tokens 均 | output_tokens 均 | input 总计 |")
    print("|---|---|--:|--:|--:|--:|")
    for s in stats:
        if not s["n_ok"]:
            print(f"| {s['min_score']} | 0 | - | - | - | - |")
            continue
        print(
            f"| {s['min_score']} | {s['n_ok']} | {s['ctx_mean']:.2f} | "
            f"{s['in_mean']:.0f} | {s['out_mean']:.0f} | {s['in_total']} |"
        )

    base, gov = stats[0], stats[1]
    print()
    if base["n_ok"] and gov["n_ok"]:
        d_in = gov["in_mean"] - base["in_mean"]
        pct = (d_in / base["in_mean"] * 100) if base["in_mean"] else 0.0
        print(f"**input_tokens 变化**：{base['in_mean']:.0f} → {gov['in_mean']:.0f} "
              f"（{d_in:+.0f}，{pct:+.1f}%）")
        print(f"**上下文条数变化**：{base['ctx_mean']:.2f} → {gov['ctx_mean']:.2f} "
              f"（{gov['ctx_mean'] - base['ctx_mean']:+.2f}）")
        if abs(d_in) < 5:
            print("\n> ⚠️ input_tokens 几乎没变 ⇒ `min_score=0.5` 在这批语料上"
                  "**没有实际过滤掉候选**，成本治理未产生效果。")
        else:
            print("\n> ✅ input_tokens 有实质下降 ⇒ 阈值确实减少了送入 LLM 的上下文。")

    # 逐题明细落盘，便于事后核对
    out_path = ROOT / "evals" / "results" / "token_compare.json"
    out_path.write_text(
        json.dumps({"thresholds": list(THRESHOLDS),
                    "by_min_score": {str(k): v for k, v in all_rows.items()},
                    "summary": stats},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细落盘：{out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""成本报表 —— 只读 `logs/traces.jsonl`，不新增依赖。

**为什么单独一个脚本而不是塞进 pipeline**

成本是**事后聚合**，不是请求路径的一部分。放进去会让检索多一份统计开销，
而且统计口径一改就要动主链路。离线读日志，改多少次都不碰生产代码。

**三种口径，不能混**

| 口径 | 来源 | 用途 |
|---|---|---|
| **在线** | `logs/traces.jsonl`（本脚本） | 真实请求的成本分布、异常占比 |
| **离线** | `evals/results/summary_*.json` | 可复现的基线对照（固定 17 题） |
| 单次估算 | `observe.PRICE` | 单价核对 |

本脚本输出的是**在线口径**。写进文档时必须标注，否则会和离线基线对不上。

**数据质量必须报告**

日志里混着测试产生的记录（`usage=null`、`cost=0`、`status=error`）。
如果直接算 p50，这些会被当成「免费请求」拉低均值 —— 所以：
- 区分「有效样本」（有 usage）与「无效样本」
- 分位数只算有效样本
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TRACES = Path("logs") / "traces.jsonl"


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def percentile(values: list[float], frac: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * frac)))
    return round(ordered[idx], 4)


def is_valid(rec: dict) -> bool:
    """有 usage 才算有效样本（否则 cost 字段没有意义）。"""
    return isinstance(rec.get("usage"), dict) and rec["usage"].get("total_tokens")


def main() -> int:
    records = load(TRACES)
    if not records:
        print(f"未找到日志：{TRACES}", file=sys.stderr)
        return 1

    valid = [r for r in records if is_valid(r)]
    invalid = len(records) - len(valid)

    costs = [float(r.get("cost_cny") or 0.0) for r in valid]
    total_cost = sum(costs)
    over_01 = sum(1 for c in costs if c > 0.01)

    print("# 成本报表（在线 traces 口径）\n")
    print(f"样本：日志 {len(records)} 条，其中**有效** {len(valid)} 条、"
          f"无 usage 的 {invalid} 条（多为测试记录，已排除在分位统计外）\n")

    if not valid:
        print("⚠️ 没有有效样本，无法计算成本分位。", file=sys.stderr)
        return 2

    # ── 1. 单请求成本分布 ─────────────────────────────────
    print("## 1. 单请求成本分布（有效样本）\n")
    print("| 指标 | 值 |")
    print("|---|---|")
    print(f"| 请求数 | {len(valid)} |")
    print(f"| 成本均值 | ¥{total_cost / len(valid):.4f} |")
    print(f"| 成本 p50 | ¥{percentile(costs, 0.50):.4f} |")
    print(f"| 成本 p95 | ¥{percentile(costs, 0.95):.4f} |")
    print(f"| 成本 max | ¥{max(costs):.4f} |")
    print(f"| 累计成本 | ¥{total_cost:.4f} |")
    print(f"| 超 ¥0.01 的请求 | {over_01} / {len(valid)}"
          f"（{over_01 / len(valid):.1%}）|\n")

    # ── 2. 按 status 与 mode 分组 ─────────────────────────
    print("## 2. 按状态与模式的成本差异\n")
    print("| 分组 | 维度 | 请求数 | 平均成本 | 平均 token |")
    print("|---|---|--:|--:|--:|")

    def group(key_fn, label: str) -> None:
        buckets: dict[str, list[dict]] = {}
        for r in valid:
            buckets.setdefault(str(key_fn(r)), []).append(r)
        for name, items in sorted(buckets.items()):
            avg_cost = sum(float(x.get("cost_cny") or 0) for x in items) / len(items)
            avg_tok = sum(
                int(x["usage"].get("total_tokens") or 0) for x in items
            ) / len(items)
            print(f"| {label} | {name} | {len(items)} | ¥{avg_cost:.4f} | {avg_tok:.0f} |")

    group(lambda r: r.get("status") or "?", "status")
    group(lambda r: (r.get("params") or {}).get("mode") or "?", "mode")
    print()

    # ── 3. 延迟拆分 ───────────────────────────────────────
    print("## 3. 延迟阶段拆分（均值）\n")
    stages = ("embed_ms", "search_ms", "sparse_ms", "fuse_ms", "generate_ms")
    print("| 阶段 | 均值 (ms) | 占比 |")
    print("|---|--:|--:|")
    sums = {}
    for st in stages:
        vals = [
            float((r.get("stage_ms") or {}).get(st) or 0.0) for r in valid
        ]
        sums[st] = sum(vals) / len(vals) if vals else 0.0
    total = sum(sums.values()) or 1.0
    for st, v in sums.items():
        if v > 0:
            print(f"| {st} | {v:.1f} | {v / total:.1%} |")
    print(f"\n合计均值：**{total:.1f} ms**\n")

    # ── 4. 数据质量 ───────────────────────────────────────
    print("## 4. 数据质量\n")
    print("| 项 | 值 |")
    print("|---|---|")
    print(f"| 无 usage 的记录 | {invalid} |")
    errs = sum(1 for r in records if r.get("status") == "error")
    print(f"| status=error | {errs} |")
    print(f"| 时间跨度 | {records[0].get('ts')} ~ {records[-1].get('ts')} |")
    print("\n> ⚠️ 无 usage 的记录几乎都来自测试（测试直接写了真实日志文件）。"
          "这是已知问题：测试应改用临时日志路径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
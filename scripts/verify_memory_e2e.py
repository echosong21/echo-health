"""端到端验证：记忆是否真的被用上（不只是 prompt 里塞了字）。

**为什么需要这个验证**

`test_memory.py` 里的字节级测试证明了「无记忆时 prompt 不变」，
但它证明不了「有记忆时模型真的用了」—— 那是语义问题，必须真实跑。

**⚠️ 问题选择是有讲究的（第一版就选错了）**

第一版用「那我晚饭该注意什么」做第二轮提问 —— 结果无论有没有记忆，
模型都答「材料中未提及」，因为**检索到的材料里根本没有"晚饭"的内容**。
于是验证结论是 INCONCLUSIVE，看上去像记忆失效，实际是**测试问题设计失败**。

选问题的规则：
1. **材料里必须真的有答案**（否则只能输出 no_answer，测不出差异）
2. **答案的正确性依赖用户条件**（否则有没有记忆都一样）

「我这种情况，晚饭能吃咸菜吗」满足两条：材料有咸菜的营养数据，
而"能不能吃"依赖用户是否高血压。

**验证设计：三组对照**

    ① 第一轮：问「我最近血压有点高」        → 写入记忆（user + assistant）
    ② 第二轮：问「我这种情况，能吃咸菜吗」  → **有记忆**
    ③ 对照：  同样的问题，session_id=None  → **无记忆**

**为什么必须有 ③ 对照组**

health 领域本身就会谈饮食建议，"②提到血压"完全可能是巧合（领域知识而非记忆）。
只有与"同样的输入、无记忆"对比，才能排除这种解释。

**成本**：3 次 LLM 调用 ≈ ¥0.01
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DOMAIN", "health")

# 记忆是否生效的关键词：这些词应当由 ① 的血压约束"带出来"
MEMORY_MARKERS = ("血压", "钠", "盐", "咸", "降压", "高血压")


def _hit(text: str) -> list[str]:
    return [m for m in MEMORY_MARKERS if m in (text or "")]


def main() -> int:
    from src import pipeline as pl
    from src.memory import reset_store
    from src.memory.pg_store import PgMemoryStore

    reset_store()
    store = PgMemoryStore()
    sid = f"e2e_{uuid.uuid4().hex[:8]}"

    q1 = "我最近血压有点高"
    # ⚠️ 这个问题必须「材料里有答案」且「答案依赖用户条件」
    q2 = "我这种情况，晚饭能吃咸菜吗"

    print(f"session_id = {sid}\n")

    # ── ① 第一轮：写入记忆 ───────────────────────────────
    print(f"① 第一轮（带记忆）: {q1}")
    r1 = pl.rag_query(q1, top_k=3, session_id=sid)
    print(f"   回答: {r1['answer'][:100].replace(chr(10), ' ')}...")
    print(f"   上下文 {len(r1['contexts'])} 条 | hit={_hit(r1['answer'])}\n")

    # ── 确认记忆真的写进去了 ─────────────────────────────
    mem = store.read(session_id=sid)
    print(f"   记忆已写入: episodes={len(mem['episodes'])} 条")
    for e in mem["episodes"]:
        print(f"     turn={e['turn']} {e['role']:9s} {e['text'][:40]}")
    assert len(mem["episodes"]) == 2, "应写入 user+assistant 两条"
    print()

    # ── ② 第二轮：有记忆 ─────────────────────────────────
    print(f"② 第二轮（有记忆）: {q2}")
    r2 = pl.rag_query(q2, top_k=3, session_id=sid)
    hits2 = _hit(r2["answer"])
    print(f"   回答: {r2['answer'][:200].replace(chr(10), ' ')}...")
    print(f"   命中记忆关键词: {hits2}\n")

    # ── ③ 对照：同样的输入，无记忆 ───────────────────────
    print(f"③ 对照（无记忆，同样的问题）: {q2}")
    r3 = pl.rag_query(q2, top_k=3, session_id=None)
    hits3 = _hit(r3["answer"])
    print(f"   回答: {r3['answer'][:200].replace(chr(10), ' ')}...")
    print(f"   命中记忆关键词: {hits3}\n")

    # ── 判定 ─────────────────────────────────────────────
    print("=" * 70)
    print("判定")
    print("=" * 70)
    print(f"  ②（有记忆）命中 {len(hits2)} 个: {hits2}")
    print(f"  ③（无记忆）命中 {len(hits3)} 个: {hits3}")

    if len(hits2) > len(hits3):
        print("\n  ✅ 记忆生效：带记忆的回答引入了第一轮的血压约束，而无记忆版本没有")
        verdict = "PASS"
    elif hits2 and hits3 and set(hits2) == set(hits3):
        print("\n  ⚠️ 两者命中相同的关键词 —— 无法区分是记忆生效还是领域知识")
        print("     建议：换一个答案更依赖用户条件的问题")
        verdict = "INCONCLUSIVE"
    elif not hits2 and not hits3:
        print("\n  ⚠️ 两者都未命中 —— 检查材料里是否根本没有相关内容")
        print("     （材料无答案时，有没有记忆都只能输出 no_answer）")
        verdict = "INCONCLUSIVE"
    else:
        print("\n  ⚠️ 无记忆版本命中更多，反直觉，需要人工看回答")
        verdict = "INCONCLUSIVE"

    # ── 清理 ─────────────────────────────────────────────
    with store._pool.connection() as conn:
        conn.execute("DELETE FROM episodes WHERE session_id = %s", (sid,))
        conn.commit()
    print(f"\n  已清理测试数据（session={sid}）")
    print(f"\n结论: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
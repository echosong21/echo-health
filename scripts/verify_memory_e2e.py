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
    uid = f"u_e2e_{uuid.uuid4().hex[:8]}"

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

    # ── 判定（同会话部分）───────────────────────────────
    print("=" * 70)
    print("第一部分：同一 session 两轮")
    print("=" * 70)
    print(f"  ②（有记忆）命中 {len(hits2)} 个: {hits2}")
    print(f"  ③（无记忆）命中 {len(hits3)} 个: {hits3}")

    if len(hits2) > len(hits3):
        print("  ✅ 记忆生效：带记忆的回答引入了第一轮的血压约束，而无记忆版本没有")
        verdict_same = "PASS"
    elif hits2 and hits3 and set(hits2) == set(hits3):
        print("  ⚠️ 两者命中相同的关键词 —— 无法区分是记忆生效还是领域知识")
        verdict_same = "INCONCLUSIVE"
    elif not hits2 and not hits3:
        print("  ⚠️ 两者都未命中 —— 检查材料里是否根本没有相关内容")
        verdict_same = "INCONCLUSIVE"
    else:
        print("  ⚠️ 无记忆版本命中更多，反直觉，需要人工看回答")
        verdict_same = "INCONCLUSIVE"

    # ── ④ 跨 session：稳定事实（语义记忆）───────────────
    #
    # 这一段的区别是关键的：**新开一个 session**，情景记忆（episodes）是空的，
    # 唯一可能带上的是 facts。如果这里也能生效，说明事实真的跟着用户走，
    # 而不是只跟着某个会话走。
    print()
    print("=" * 70)
    print("第二部分：跨 session 的稳定事实")
    print("=" * 70)

    from src.memory import remember

    ALLERGY_MARKERS = ("花生", "过敏", "避免", "不能吃", "慎")

    ok = remember("allergies", "对花生过敏", evidence="用户自述", user_id=uid)
    print(f"④ remember('allergies', '对花生过敏') → {ok}")
    assert ok is True, "remember 应写入成功"

    # 新 session：episodes 必然是空的
    sid_new = f"e2e_cross_{uuid.uuid4().hex[:8]}"
    mem_new = store.read(session_id=sid_new, user_id=uid)
    print(f"   新 session 的 episodes = {len(mem_new['episodes'])} 条（应为 0）")
    print(f"   新 session 的 facts    = {mem_new['facts']}（应为 allergies）")
    assert mem_new["episodes"] == [], "新 session 不该有情景记忆"

    q3 = "我这种情况，花生能吃吗"
    print(f"\n   提问（带 facts）: {q3}")
    r4 = pl.rag_query(q3, top_k=3, session_id=sid_new, user_id=uid)
    hits4 = [m for m in ALLERGY_MARKERS if m in r4["answer"]]
    print(f"   回答: {r4['answer'][:220].replace(chr(10), ' ')}...")
    print(f"   命中过敏相关词: {hits4}")

    # 清空 facts，同一问题再问
    with store._pool.connection() as conn:
        conn.execute("DELETE FROM facts WHERE user_id = %s", (uid,))
        conn.commit()
    print(f"\n   提问（清空 facts 后）: {q3}")
    r5 = pl.rag_query(q3, top_k=3, session_id=f"e2e_cross2_{uuid.uuid4().hex[:8]}",
                      user_id=f"{uid}_nofact")
    hits5 = [m for m in ALLERGY_MARKERS if m in r5["answer"]]
    print(f"   回答: {r5['answer'][:220].replace(chr(10), ' ')}...")
    print(f"   命中过敏相关词: {hits5}")

    if "过敏" in hits4 and "过敏" not in hits5:
        print("\n  ✅ 跨 session 事实生效：带 facts 时提到过敏，清空后不再提")
        verdict_cross = "PASS"
    elif len(hits4) > len(hits5):
        print("\n  ✅ 跨 session 事实有影响（命中词更多）")
        verdict_cross = "PASS"
    elif not hits4 and not hits5:
        print("\n  ⚠️ 两者都未提到过敏 —— 材料里可能没有过敏相关内容")
        verdict_cross = "INCONCLUSIVE"
    else:
        print("\n  ⚠️ 未观察到跨 session 事实的影响")
        verdict_cross = "INCONCLUSIVE"

    # ── 总判定 ───────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"同会话记忆: {verdict_same}")
    print(f"跨会话事实: {verdict_cross}")
    print("=" * 70)
    overall = "PASS" if (verdict_same == "PASS" and verdict_cross == "PASS") else "INCONCLUSIVE"
    print(f"CROSS_SESSION={verdict_cross}")
    print(f"结论: {overall}")

    # ── 清理 ─────────────────────────────────────────────
    with store._pool.connection() as conn:
        conn.execute(
            "DELETE FROM episodes WHERE session_id LIKE %s OR session_id LIKE %s",
            (f"{sid}%", f"e2e_cross%"),
        )
        conn.execute("DELETE FROM facts WHERE user_id LIKE %s", (f"{uid}%",))
        conn.commit()
    print(f"\n  已清理测试数据（session={sid}, uid={uid}）")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
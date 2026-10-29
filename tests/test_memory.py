"""记忆层测试：接口契约 / 字节级回归 / 边界 / 存储隔离。

**这个文件里最重要的一条是 `test_no_memory_prompt_is_byte_identical`。**

它守的不是记忆功能，而是**整个评测基线**：
`EVAL_REPORT` 里 17/17、top-1 70.6%、MRR 0.8284 这些数字，是拿改造前的 prompt
跑出来的。只要 prompt 多一个换行，那些数字就不可比 —— 而这种偏差肉眼看不出来。

**测试隔离策略（与文件方案不同）**

记忆存在 PostgreSQL 里，没法像文件那样重定向目录，只能靠**数据隔离**：
- 每条测试用独立的 `session_id` 前缀（`test_<uuid8>`）
- 语义事实用独立 `user_id`（`u_test_<uuid8>`）
- 测试后按前缀 DELETE，不碰真实数据

**PG 不可达时整组跳过**（`pytest.skip`）：
记忆测试不该成为 CI 的硬依赖 —— CI 环境没有 PG，不该因此变红。
"""
from __future__ import annotations

import uuid

import pytest

from src import pipeline as pl
from src.memory.base import (
    DEFAULT_USER_ID,
    MAX_EPISODES_PER_SESSION,
    MAX_FACTS_PER_USER,
    MAX_TEXT_LENGTH,
    MemoryStore,
)


# ── 字节级回归（守住评测基线）────────────────────────────────
def test_no_memory_prompt_is_byte_identical():
    """无记忆时 prompt 必须与改造前逐字节相同。

    基准是 2026-10-27 20:38 实测的：
        PROMPT_TMPL.format(system='SYS', contexts='CTX', question='Q')
        == 'SYS\\n\\n材料：\\nCTX\\n\\n问题：Q\\n'
    """
    baseline = "SYS\n\n材料：\nCTX\n\n问题：Q\n"
    actual = pl._build_prompt("SYS", "CTX", "Q", memory_block="")
    assert actual == baseline, "无记忆时 prompt 与基线不一致 ⇒ 评测基线作废"


def test_memory_block_inserted_between_system_and_materials():
    """有记忆时只多出一段，其余字节不动。"""
    out = pl._build_prompt("SYS", "CTX", "Q", memory_block="已知的用户情况：高血压=是")
    assert out == "SYS\n\n已知的用户情况：高血压=是\n\n材料：\nCTX\n\n问题：Q\n"


def test_render_memory_empty_returns_empty_string():
    """空记忆必须返回空串 —— 这是字节级兼容的前提。"""
    assert pl._render_memory(None) == ""
    assert pl._render_memory({}) == ""
    assert pl._render_memory({"episodes": [], "facts": {}}) == ""


def test_render_memory_includes_facts_and_episodes():
    mem = {
        "facts": {"allergies": "花生"},
        "episodes": [
            {"turn": 1, "role": "user", "text": "我最近血压高"},
            {"turn": 1, "role": "assistant", "text": "建议控钠"},
        ],
    }
    out = pl._render_memory(mem)
    assert "allergies=花生" in out
    assert "- user: 我最近血压高" in out
    assert "- assistant: 建议控钠" in out


def test_render_memory_caps_episode_count():
    """只渲染最近 N 条，防止长会话把 prompt 撑爆。"""
    eps = [{"turn": i, "role": "user", "text": f"第{i}轮"} for i in range(20)]
    out = pl._render_memory({"episodes": eps, "facts": {}})
    assert out.count("- user:") == pl.MEMORY_EPISODE_WINDOW


# ── 接口契约 ────────────────────────────────────────────────
def test_memory_store_is_protocol_with_two_methods():
    """Protocol 只要求 read/write，不多不少。"""
    assert hasattr(MemoryStore, "read")
    assert hasattr(MemoryStore, "write")
    for forbidden in ("search", "summarize", "forget"):
        assert not hasattr(MemoryStore, forbidden), f"不应提供 {forbidden}()"


def test_boundary_constants_are_single_source():
    """4 个边界值都定义在 base.py，值符合设计文档。"""
    assert MAX_TEXT_LENGTH == 500
    assert MAX_FACTS_PER_USER == 30
    assert MAX_EPISODES_PER_SESSION == 200
    assert DEFAULT_USER_ID == "u_default"


# ── 记忆层失败不拖死主链路 ──────────────────────────────────
def test_load_memory_block_never_raises(monkeypatch):
    """存储炸了也要返回空串，不能抛。"""

    class Boom:
        def read(self, **_kw):
            raise RuntimeError("PG 挂了")

    monkeypatch.setattr("src.memory.get_store", lambda log=None: Boom())
    assert pl._load_memory_block("s1") == ""


def test_load_memory_block_skips_when_no_session():
    """session_id 为 None 时不碰存储。"""
    assert pl._load_memory_block(None) == ""


def test_save_turns_never_raises(monkeypatch):
    """写失败必须静默，否则回答已经生成了也发不出去。"""

    class Boom:
        def write(self, **_kw):
            raise RuntimeError("PG 挂了")

    monkeypatch.setattr("src.memory.get_store", lambda log=None: Boom())
    pl._save_turns("s1", "问题", "回答")   # 不应抛


def test_save_turns_skips_empty_answer():
    """空回答不写进历史 —— 会污染下一轮上下文。"""
    pl._save_turns("s1", "问题", None)
    pl._save_turns("s1", "问题", "")


# ── PG 实存测试（PG 不可达时跳过）─────────────────────────────
@pytest.fixture
def mem_store():
    """真 PG 的 store + 独立前缀；PG 不可达则整条跳过。"""
    from src.memory.pg_store import PgMemoryStore

    try:
        store = PgMemoryStore()
        # 探活：能跑一次查询才算可用
        with store._pool.connection() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"PostgreSQL 不可达，跳过记忆实存测试（{type(exc).__name__}）")

    tag = uuid.uuid4().hex[:8]
    yield store, f"test_{tag}", f"u_test_{tag}"

    with store._pool.connection() as conn:
        conn.execute("DELETE FROM episodes WHERE session_id LIKE %s", (f"test_{tag}%",))
        conn.execute("DELETE FROM facts WHERE user_id = %s", (f"u_test_{tag}",))
        conn.commit()
    store.close()


def test_write_then_read_roundtrip(mem_store):
    store, sid, _uid = mem_store
    store.write(session_id=sid, role="user", text="我最近血压有点高")
    store.write(session_id=sid, role="assistant", text="建议控制钠摄入")

    out = store.read(session_id=sid)
    assert len(out["episodes"]) == 2
    assert [e["role"] for e in out["episodes"]] == ["user", "assistant"]
    assert out["episodes"][0]["turn"] <= out["episodes"][1]["turn"]   # 升序


def test_long_text_is_truncated(mem_store):
    store, sid, _uid = mem_store
    store.write(session_id=sid, role="user", text="长" * (MAX_TEXT_LENGTH + 300))
    out = store.read(session_id=sid)
    assert len(out["episodes"][-1]["text"]) == MAX_TEXT_LENGTH


def test_read_with_none_session_returns_no_episodes(mem_store):
    store, _sid, uid = mem_store
    store.upsert_fact("allergies", "花生", user_id=uid)
    out = store.read(session_id=None, user_id=uid)
    assert out["episodes"] == []
    assert out["facts"] == {"allergies": "花生"}


def test_episodes_rolling_cap(mem_store):
    """写超过上限后只保留最近 MAX_EPISODES_PER_SESSION 条。"""
    store, sid, _uid = mem_store
    over = MAX_EPISODES_PER_SESSION + 20
    for i in range(over):
        store.write(session_id=sid, role="user", text=f"第{i}条")

    with store._pool.connection() as conn:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE session_id = %s", (sid,)
        ).fetchone()[0]
    assert cnt == MAX_EPISODES_PER_SESSION, f"应裁剪到 {MAX_EPISODES_PER_SESSION}，实际 {cnt}"


def test_facts_cap_returns_false(mem_store):
    """事实达上限后新增返回 False，而不是静默丢弃。"""
    store, _sid, uid = mem_store
    for i in range(MAX_FACTS_PER_USER):
        assert store.upsert_fact(f"k{i}", f"v{i}", user_id=uid) is True
    assert store.upsert_fact("over_limit", "x", user_id=uid) is False

    # 已有 key 的更新不受上限限制
    assert store.upsert_fact("k0", "updated", user_id=uid) is True
    assert store.read(session_id=None, user_id=uid)["facts"]["k0"] == "updated"


def test_read_degrades_on_broken_store(mem_store):
    """存储炸了 read 返回空结构，不抛。

    ⚠️ 必须**恢复** `_pool` —— 否则 fixture 的清理阶段也会用到它而报错，
    表现出来是「测试通过但 teardown ERROR」这种容易误读的失败。
    """
    store, sid, _uid = mem_store
    original_pool = store._pool

    class Boom:
        def connection(self):
            raise RuntimeError("模拟 PG 挂掉")

    store._pool = Boom()
    try:
        out = store.read(session_id=sid)
        assert out == {"episodes": [], "facts": {}}
    finally:
        store._pool = original_pool
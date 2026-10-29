"""PostgreSQL 实现（`episodes` + `facts` 两张表）。

**为什么用连接池而不是每次新建连接**

FastAPI 是并发的。每次请求新建连接会在高并发下耗尽 PostgreSQL 的
`max_connections`（默认 100）—— 这不是性能优化，是**可用性要求**。

**为什么 read 吞异常、write 也吞**

记忆是增强项。存储挂了应该退化到「无记忆」，而不是让整个问答失败。
这与 MCP 仓的降级设计同一条原则：**增强能力不能把主链路拖死**。

> ⚠️ 一个例外：写入失败应当**记一条 warning**，否则记忆静默丢失
> 谁也不知道。日志走调用方的 logger，实现本身不 import 全局 logger。

**为什么 turn 由本实现分配而不是调用方传**

调用方（pipeline）不该关心「这是第几轮」。实现用
`COALESCE(MAX(turn), 0) + 1` 分配，且表上有
`UNIQUE (session_id, turn, role)` 兜底 —— 并发写入撞车时后者会拒绝，
不会产生两条 turn=5 的记录。

**为什么滚动裁剪用一条 DELETE 而不是定时任务**

MVP 单机不需要后台任务。写入后立即裁剪，代价是一条索引范围扫描，
在 200 条上限下可忽略。等量级上去再换异步清理。
"""
from __future__ import annotations

import json
from typing import Any

from .base import (
    DEFAULT_USER_ID,
    MAX_EPISODES_PER_SESSION,
    MAX_FACTS_PER_USER,
    MAX_TEXT_LENGTH,
    MAX_TURNS_PER_SESSION,
)


class PgMemoryStore:
    """基于 PostgreSQL 的 `MemoryStore` 实现。

    构造时可传 `dsn`（如 `dbname=echo_health user=postgres host=127.0.0.1`），
    不传则读环境变量 `MEMORY_DSN`，再缺省为本机默认库。
    """

    def __init__(self, dsn: str | None = None, *, log=None) -> None:
        from psycopg_pool import ConnectionPool

        import os

        conninfo = dsn or os.getenv("MEMORY_DSN") or (
            "dbname=echo_health user=postgres host=127.0.0.1 port=5432"
        )
        # min_size=1：单用户 MVP 不需要预热多个连接
        # open=True：构造即连接；连不上会抛，让配置错误立刻暴露
        self._pool = ConnectionPool(conninfo, min_size=1, max_size=4, open=True)
        self._log = log

    # ── 读 ───────────────────────────────────────────────────
    def read(
        self,
        *,
        session_id: str | None,
        user_id: str = DEFAULT_USER_ID,
        limit: int = MAX_TURNS_PER_SESSION,
    ) -> dict[str, Any]:
        """见 `MemoryStore.read` 的约定。**任何异常都折成空结构。**"""
        empty: dict[str, Any] = {"episodes": [], "facts": {}}
        try:
            facts = self._read_facts(user_id)
            episodes = self._read_episodes(session_id, limit) if session_id else []
            return {"episodes": episodes, "facts": facts}
        except Exception as exc:                       # noqa: BLE001
            self._warn("memory read failed, degrade to empty", exc)
            return empty

    def _read_episodes(self, session_id: str, limit: int) -> list[dict]:
        # 先按 turn 倒序取最近 N 轮，再在 Python 里翻回升序 ——
        # SQL 里直接升序会取到**最旧**的 N 条，正好相反
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT turn, role, text, ts
                  FROM episodes
                 WHERE session_id = %s
                 ORDER BY turn DESC
                 LIMIT %s
                """,
                (session_id, limit * 2),   # 一轮含 user+assistant 两条
            ).fetchall()
        return [
            {"turn": r[0], "role": r[1], "text": r[2], "ts": r[3].isoformat()}
            for r in reversed(rows)
        ]

    def _read_facts(self, user_id: str) -> dict[str, str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM facts WHERE user_id = %s ORDER BY key",
                (user_id,),
            ).fetchall()
        return {k: v for k, v in rows}

    # ── 写 ───────────────────────────────────────────────────
    def write(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        user_id: str = DEFAULT_USER_ID,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """见 `MemoryStore.write` 的约定。超长截断、超量滚动删除。"""
        if role not in ("user", "assistant"):
            raise ValueError(f"role 必须是 user/assistant，收到 {role!r}")

        clipped = (text or "")[:MAX_TEXT_LENGTH]
        try:
            with self._pool.connection() as conn, conn.transaction():
                # turn 分配与插入在同一事务里，配合 UNIQUE 约束防并发撞车
                row = conn.execute(
                    """
                    INSERT INTO episodes (session_id, user_id, turn, role, text, meta)
                    VALUES (%s, %s,
                            COALESCE((SELECT MAX(turn) FROM episodes WHERE session_id = %s), 0) + 1,
                            %s, %s, %s::jsonb)
                    RETURNING turn
                    """,
                    (session_id, user_id, session_id, role, clipped,
                     json.dumps(meta or {}, ensure_ascii=False)),
                ).fetchone()

                # 滚动裁剪：只留最近 MAX_EPISODES_PER_SESSION 条
                conn.execute(
                    """
                    DELETE FROM episodes
                     WHERE session_id = %s
                       AND id NOT IN (
                           SELECT id FROM episodes
                            WHERE session_id = %s
                            ORDER BY turn DESC, role DESC
                            LIMIT %s
                       )
                    """,
                    (session_id, session_id, MAX_EPISODES_PER_SESSION),
                )
            _ = row
        except Exception as exc:                       # noqa: BLE001
            self._warn("memory write failed", exc)

    # ── 语义事实（显式维护，不自动抽取）───────────────────────
    def upsert_fact(
        self,
        key: str,
        value: str,
        *,
        user_id: str = DEFAULT_USER_ID,
        evidence: str | None = None,
    ) -> bool:
        """写入/更新一条稳定事实。

        **刻意不做自动抽取**（不从对话里猜"我高血压"）：
        猜错的代价比不记更糟 —— 会基于错误事实给出建议。
        调用方必须是显式决定的（用户主动说、或界面上让用户填）。

        返回 False 表示已达 `MAX_FACTS_PER_USER` 上限且是新增 key。
        """
        try:
            with self._pool.connection() as conn, conn.transaction():
                exists = conn.execute(
                    "SELECT 1 FROM facts WHERE user_id = %s AND key = %s",
                    (user_id, key),
                ).fetchone()
                if not exists:
                    cnt = conn.execute(
                        "SELECT COUNT(*) FROM facts WHERE user_id = %s", (user_id,)
                    ).fetchone()[0]
                    if cnt >= MAX_FACTS_PER_USER:
                        self._warn("facts 达上限，拒绝新增", f"{cnt}/{MAX_FACTS_PER_USER}")
                        return False
                conn.execute(
                    """
                    INSERT INTO facts (user_id, key, value, evidence, updated_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (user_id, key)
                    DO UPDATE SET value = EXCLUDED.value,
                                  evidence = COALESCE(EXCLUDED.evidence, facts.evidence),
                                  updated_at = now()
                    """,
                    (user_id, key, value, evidence),
                )
            return True
        except Exception as exc:                       # noqa: BLE001
            self._warn("upsert_fact failed", exc)
            return False

    # ── 运维 ─────────────────────────────────────────────────
    def close(self) -> None:
        self._pool.close()

    def _warn(self, msg: str, detail: object) -> None:
        """记忆层的失败统一走这里：有 logger 就记，没有就静默。

        **不抛异常**是刻意的 —— 见类 docstring 的"增强能力不能拖死主链路"。
        """
        if self._log is not None:
            self._log.warning("%s: %s", msg, detail)
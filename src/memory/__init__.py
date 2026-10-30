"""记忆层。

**分层**（能力要求）：

| 层 | 存什么 | 失效策略 |
|---|---|---|
| 情景（episodic） | 发生过什么：按时间累积的交互记录 | 滑动窗口，只留最近 N 轮 |
| 语义（semantic） | 关于用户的稳定事实：过敏、忌口、慢性病、目标 | 长期，显式更新才变 |

**存储与接口分离**

`base.MemoryStore` 是 Protocol —— 只要求 `read` / `write` 两个方法。
当前实现是 `pg_store.PgMemoryStore`（PostgreSQL）。将来换存储只需加一个类，
`pipeline.py` 一行不用改 —— 这是「未来不大改」的技术根基。

**故障语义**

记忆是增强项。存储不可达时 `read` 返回空结构、`write` 静默失败，
主链路按"无记忆"继续。**增强能力不能把主链路拖死。**

⚠️ 与 MCP 仓的 `common.ops.reindex` 不同：那个是写操作，失败必须让调用方知道；
这里记忆丢失只会少一点上下文，不值得让整次问答失败。
"""
from .base import (
    DEFAULT_USER_ID,
    MAX_EPISODES_PER_SESSION,
    MAX_FACTS_PER_USER,
    MAX_TEXT_LENGTH,
    MAX_TURNS_PER_SESSION,
    MemoryStore,
)

import threading

__all__ = [
    "DEFAULT_USER_ID",
    "MAX_EPISODES_PER_SESSION",
    "MAX_FACTS_PER_USER",
    "MAX_TEXT_LENGTH",
    "MAX_TURNS_PER_SESSION",
    "MemoryStore",
]


def get_store(log=None) -> "MemoryStore":
    """取当前记忆存储（惰性单例）。

    惰性是为了**不在 import 时建连接** —— 这是本仓既有的纪律
    （`pipelines.py` 的模块级 `setup_logging` 就是反例，后来修掉了）。
    没有 PG 时 import 这个包不应该报错。

    **加锁的原因**：FastAPI 是并发的。如果两个请求同时首次调用这里，
    都会看到 `_store is None` 并各自建一个 `PgMemoryStore`（每个带一个
    ConnectionPool）—— 后者覆盖前者，前者那份连接池**永远泄漏**。
    单机跑几百次请求不会发现，但它是真实存在的资源泄漏。
    用 double-check：绝大多数调用只读一次 `_store`，不进锁。
    """
    global _store
    if _store is None:
        with _LOCK:
            if _store is None:                      # double-check
                from .pg_store import PgMemoryStore

                _store = PgMemoryStore(log=log)
    return _store


_store: "MemoryStore | None" = None
_LOCK = threading.Lock()


def reset_store() -> None:
    """测试用：清掉单例，让下一个 `get_store()` 重新建连接。"""
    global _store
    with _LOCK:
        if _store is not None:
            close = getattr(_store, "close", None)
            if callable(close):
                close()
        _store = None
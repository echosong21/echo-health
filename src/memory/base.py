"""记忆层的存储契约 —— 只有两个方法。

**为什么是 Protocol 而不是抽象基类**

`Protocol` 是结构化类型：任何实现了 `read` / `write` 的类都自动满足它，
不需要显式继承。这样：

    · 测试里可以用一个 20 行的假存储，不必继承
    · 将来换 PG / SQLite / 向量库实现，彼此不共享基类也能互换
    · 不会因为基类加了一个默认方法，所有实现被迫跟着改

**为什么只有两个方法**

`read` / `write` 是最小完备集。刻意**不**提供：

    · `search()`   —— 一旦有它，接口就锁死成"必须具备语义检索能力"。
                      将来想用纯关系库实现就得假装实现一个假搜索。
    · `summarize()` —— 压缩策略属于实现内部，暴露出来会让调用方开始依赖它。
    · `forget()`   —— 删数据的接口一旦先设计，就要处理"删了但索引还在"的
                      复杂情况。等真的需要（如合规要求）再设计。

**返回值为什么是「可扩展字典」而不是 dataclass**

    read() → {"episodes": [...], "facts": {...}}

将来加一类记忆（如 L2 的图谱节点）只需多一个 key，
老调用方用 `.get()` 读不到新 key 但**不会崩**。用 dataclass 的话，
每加一个字段就要改所有构造与解构点 —— 那就成了"大改"。

**失败语义：read 不抛异常**

记忆是增强项，不是关键路径。`read` 在存储不可达时应返回空结构，
让主链路按"无记忆"继续 —— 这与 MCP 仓的降级设计是同一条原则。
`write` 的失败由调用方决定是否记录，实现本身不应抛出中断请求流的异常。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ── 写入边界（设计文档 §3.5 的 4 个限制值，**只出现在这一处**）──────
#
# ⚠️ 这些数字不许散落到实现里。散落之后修改边界要改多处，
#    而且很容易改漏一个 —— 记忆无限膨胀就是这么来的。
MAX_TURNS_PER_SESSION = 20      # 单会话保留最近 20 轮（40 条：user + assistant）
MAX_EPISODES_PER_SESSION = 200  # 单会话硬上限，超出滚动删最旧
MAX_FACTS_PER_USER = 30         # 语义事实条数上限
MAX_TEXT_LENGTH = 500           # 单条 text 字符上限，超出截断

DEFAULT_USER_ID = "u_default"


@runtime_checkable
class MemoryStore(Protocol):
    """记忆存储的最小契约。

    实现可以是 PostgreSQL、SQLite、文件、内存 —— 调用方只认这两个方法。
    """

    def read(
        self,
        *,
        session_id: str | None,
        user_id: str = DEFAULT_USER_ID,
        limit: int = MAX_TURNS_PER_SESSION,
    ) -> dict[str, Any]:
        """取上下文。

        返回结构（**可扩展字典**）：

            {
              "episodes": [{"turn": 1, "role": "user", "text": "...", "ts": "..."}, ...],
              "facts": {"allergies": "花生", ...},
            }

        约定：
        · `session_id=None` ⇒ `episodes` 为空（只取 facts）
        · 任何存储错误 ⇒ 返回 `{"episodes": [], "facts": {}}`，**不抛异常**
        · `episodes` 按 `turn` **升序**（渲染时最自然的顺序）
        """

    def write(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        user_id: str = DEFAULT_USER_ID,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """追加一条交互记录。

        约定：
        · `role` ∈ {"user", "assistant"}
        · `turn` 由实现分配（同一 session 内递增）
        · `text` 超过 `MAX_TEXT_LENGTH` 时实现负责截断
        · 超过 `MAX_EPISODES_PER_SESSION` 时实现负责滚动删除最旧
        · `meta` 只放可枚举元信息，**不放原文**
        """


__all__ = [
    "DEFAULT_USER_ID",
    "MAX_EPISODES_PER_SESSION",
    "MAX_FACTS_PER_USER",
    "MAX_TEXT_LENGTH",
    "MAX_TURNS_PER_SESSION",
    "MemoryStore",
]
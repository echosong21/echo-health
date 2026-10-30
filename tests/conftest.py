"""测试共用夹具（trace 契约与依赖注入两组共用，统一放这里）。

**为什么需要这些夹具**

    `retrieval.py` 的检索函数签名是 `fn(client, qvec, collection, ...)` ——
    依赖注入形态，client 从参数进来。这让「不连 Milvus 也能测」成为可能，
    但前提是有一个行为正确的假 client。

    `observe.py` 的 `write_trace()` 读模块级 `TRACE_FILE` 常量，
    测试若直接调用会把假数据写进真实 `logs/`，污染观测数据。

**使用方式**

    def test_xxx(fake_client, make_hits):
        fake_client.hits_by_form["guides"] = make_hits(n=2)
        ...
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── 自动隔离：所有测试都不许写真实 logs/ ──────────────────────────
#
# **为什么需要 autouse，而不是"每个测试自觉申请"**
#
# 原先有两个污染源叠加，让 `logs/traces.jsonl` 里 41% 的记录来自测试
# （`usage=null`、`cost=0`、`status=error`）：
#
#   1. `observe.TRACE_FILE` 是模块级常量 —— 下面那个显式夹具能重定向它，
#      但**只有声明了 `tmp_trace_file` 参数的测试才用得上**。
#      后来新写的 API / 失败路径测试没声明，于是直接写进了真实文件。
#   2. `pipelines.py` 在**模块级**调用 `observe.setup_logging()` ——
#      只要 import 到 pipeline（测试必然 import），就会创建指向真实
#      `logs/app_YYYY-MM-DD.log` 的 FileHandler。
#
# 教训：**隔离不该依赖每个测试作者记得申请**。忘一个就静默污染，
# 而且污染不会自己暴露 —— 这次是靠成本报表「41% 无 usage」才发现的。
#
# ⚠️ 文件名刻意与 `tmp_trace_file` 区分（`_autouse_` 前缀），
# 避免两个夹具写到同一个文件上互相覆盖。
@pytest.fixture(autouse=True)
def _isolate_observe(monkeypatch):
    """重定向 trace 文件，并摘掉指向真实 logs/ 的文件 handler。"""
    import logging

    from src import observe

    tmp_dir = ROOT / ".pytest_tmp"
    tmp_dir.mkdir(exist_ok=True)
    target = tmp_dir / "_autouse_traces.jsonl"
    target.write_text("", encoding="utf-8")
    monkeypatch.setattr(observe, "TRACE_FILE", target)

    # 摘掉真实文件 handler；保留控制台 handler，否则排查时看不到输出
    rag_logger = logging.getLogger("rag")
    for handler in list(rag_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            rag_logger.removeHandler(handler)
            handler.close()

    yield target


# ── 夹具：临时 trace 文件 ──────────────────────────────────────────
@pytest.fixture
def tmp_trace_file(monkeypatch):
    """把 `observe.TRACE_FILE` 重定向到项目内临时目录，返回该路径。

    与 autouse 的 `_isolate_observe` 分工：那个保证**默认不污染**，
    这个给**需要断言日志内容**的测试用（返回可读路径）。

    ⚠️ **刻意不用 pytest 的 `tmp_path`**：本机 `%TEMP%\\pytest-of-*`
    存在 ACL 锁定（`PermissionError: [WinError 5]`），`tmp_path` 在该环境
    下建不出来 —— 与 `pytest-cache-files-*` 是同一类问题。
    改用项目内 `.pytest_tmp/`（已在 .gitignore），跨平台与 CI 都能写。
    """
    from src import observe

    tmp_dir = ROOT / ".pytest_tmp"
    tmp_dir.mkdir(exist_ok=True)

    target = tmp_dir / "traces.jsonl"
    target.write_text("", encoding="utf-8")      # 每个测试都从空文件开始

    monkeypatch.setattr(observe, "TRACE_FILE", target)
    return target


# ── 夹具：假 Milvus client ────────────────────────────────────────
class FakeMilvusClient:
    """只实现 `retrieval.py` 用到的那部分：`search()`。

    ⚠️ 返回形状必须与 `_to_raw` 完全一致：

        [                              # 外层按 query 分组，代码取 [0]
            [
                {"distance": 0.9, "entity": {"source", "chunk_index", "text"}},
                ...
            ]
        ]

    `filter` 参数在真实 Milvus 里是表达式 `source like "guides/%"`，
    检索层用它区分形态。假 client 必须同样按它分流，
    否则"每形态各取 per_form 条"这个核心行为测不出来。
    """

    _FORM_RE = re.compile(r'source like "([^/]+)/')

    def __init__(self, hits_by_form: dict[str, list[dict]] | None = None):
        self.hits_by_form: dict[str, list[dict]] = hits_by_form or {}
        self.calls: list[dict] = []          # 记录每次调用，便于断言"问了几路"

    def search(self, collection_name, data, limit, filter=None, output_fields=None, **kw):
        self.calls.append(
            {"collection": collection_name, "limit": limit, "filter": filter}
        )
        match = self._FORM_RE.search(filter or "")
        form = match.group(1) if match else None
        hits = self.hits_by_form.get(form, [])
        return [hits[:limit]]                # 外层再包一层：按 query 分组


@pytest.fixture
def fake_client():
    """空的假 client；测试里直接往 `hits_by_form` 塞数据。"""
    return FakeMilvusClient()


# ── 夹具：造命中 ──────────────────────────────────────────────────
@pytest.fixture
def make_hits():
    """工厂：生成 Milvus 命中列表（分数递减）。

    默认造 3 条，分数 0.9 / 0.8 / 0.7 —— 便于断言排序行为。
    """

    def _make(
        n: int = 3,
        form: str = "guides",
        score_start: float = 0.9,
        step: float = 0.1,
    ) -> list[dict]:
        return [
            {
                "distance": round(score_start - i * step, 4),
                "entity": {
                    "source": f"{form}/doc_{i}.md",
                    "chunk_index": i,
                    "text": f"这是 {form} 的第 {i} 段材料。",
                },
            }
            for i in range(n)
        ]

    return _make


# ── 夹具：假 embedding ────────────────────────────────────────────
class FakeEmbeddings:
    """只实现 `embed_query` / `embed_documents`，返回固定维度向量。

    真实 embedding 要调百炼，测试里既慢又要花钱；
    这里返回常量向量即可满足"向量形状正确"的断言。
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1] * self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


@pytest.fixture
def fake_emb():
    return FakeEmbeddings()


# ── 夹具：领域配置 ────────────────────────────────────────────────
@pytest.fixture(scope="session")
def health_domain():
    """加载 health 领域配置（session 级，读盘一次即可）。"""
    from src import domain as domain_mod

    return domain_mod.load("health")
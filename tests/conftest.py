"""测试共用夹具（R3 与 R5 共用，按 D3 纪律统一放这里）。

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


# ── 夹具：临时 trace 文件 ──────────────────────────────────────────
@pytest.fixture
def tmp_trace_file(monkeypatch):
    """把 `observe.TRACE_FILE` 重定向到项目内临时目录，返回该路径。

    observe 用的是模块级常量，`write_trace` 每次读取它，
    所以 monkeypatch 属性即可重定向，**不需要改 observe 的实现**。

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
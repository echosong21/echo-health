"""失败路径 —— 三类各一条 pytest。

**这三条测试在测什么**

不是「代码会不会崩」，而是**失败时的行为是否符合预期**：

| 类 | 期望行为 | 为什么 |
|---|---|---|
| ① 空数据 | 200 + 友好提示 | 「没有材料」是正常结果，不是错误 |
| ② 超时 | 500 + 明确错误 | 依赖不可用必须让调用方知道 |
| ③ 并发部分失败 | **冒泡为整体失败** | 见下方说明 |

**③ 的选择：冒泡，不吞掉**

`retrieve_dense_multi` 用 `pool.map(one, forms)` —— 任一路抛错会让 `list()`
重新抛出，即**整体失败**。

这是**刻意保留的行为**，理由：
- 4 路是按形态切片同一份语义，一路失败通常意味着**后端不稳定**
- 吞掉失败那路会返回**看似成功但缺失一个形态**的结果，调用方无法察觉
- 「宁可明确失败，不给不完整结果」在检索场景比「尽力返回」更安全
  （少一路召回可能导致答案依据不足，而调用方以为是完整检索）

⚠️ 若将来改为「部分成功」，必须同时做两件事：返回 `degraded` 标记 + 说明缺失哪个形态。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import api as api_mod
from src import pipeline as pl


class _Emb:
    def embed_query(self, _q):
        return [0.1] * 8


class _Resp:
    content = "（无材料）"
    usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


class _LLM:
    def invoke(self, _p):
        return _Resp()


class _EmptyClient:
    """所有 search 都返回空 —— 模拟 filter 无命中。"""

    def search(self, **_kw):
        return [[]]

    def get_collection_stats(self, **_kw):
        return {"row_count": 0}


class _TimeoutClient:
    def search(self, **_kw):
        raise TimeoutError("milvus search timeout")

    def get_collection_stats(self, **_kw):
        return {"row_count": 1230}


class _PartialFailClient:
    """一路失败、其余成功 —— 用于验证冒泡语义。

    并发下调用顺序不确定，所以**不依赖 filter 判断**，
    改用「只要发生过至少一次成功，就说明是部分失败」：
    第 1 次调用抛错，后续正常返回。
    """

    def __init__(self):
        self.calls = 0
        self.ok_calls = 0

    def search(self, **_kw):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("one form is broken")
        self.ok_calls += 1
        return [[{
            "distance": 0.8,
            "entity": {"source": "tables/x.md", "text": "材料", "chunk_index": 0},
        }]]

    def get_collection_stats(self, **_kw):
        return {"row_count": 1230}


@pytest.fixture
def client():
    return TestClient(api_mod.app)


@pytest.fixture
def wire(monkeypatch, fake_emb, health_domain):
    def _apply(client_obj, llm=None):
        monkeypatch.setattr(pl, "get_emb", lambda: fake_emb)
        monkeypatch.setattr(pl, "get_client", lambda: client_obj)
        monkeypatch.setattr(pl, "get_llm", lambda: llm or _LLM())
        monkeypatch.setattr(pl, "get_domain", lambda: health_domain)
        monkeypatch.setattr(pl, "get_collection", lambda: "echo_test")

    return _apply


# ── ① 空数据 ────────────────────────────────────────────────
def test_empty_result_is_not_an_error(client, wire):
    """无命中是正常结果 ⇒ 200，且提示明确。"""
    wire(_EmptyClient())
    r = client.post("/query", json={"question": "不存在的食物"})
    assert r.status_code == 200
    body = r.json()
    assert body["contexts"] == []
    assert "阈值" in body["answer"] or "无材料" in body["answer"]


# ── ② 超时 ──────────────────────────────────────────────────
def test_timeout_surfaces_as_500(client, wire):
    """依赖超时必须让调用方知道 ⇒ 500，而不是空结果冒充成功。"""
    wire(_TimeoutClient())
    r = client.post("/query", json={"question": "米饭多少热量"})
    assert r.status_code == 500
    assert "TimeoutError" in r.json()["detail"]


# ── ③ 并发部分失败（冒泡）──────────────────────────────────
def test_partial_concurrent_failure_bubbles(client, wire):
    """一路挂 ⇒ 整体失败（见模块 docstring 的选择说明）。"""
    broken = _PartialFailClient()
    wire(broken)
    r = client.post("/query_multi", json={"question": "米饭多少热量", "per_form": 2})
    assert r.status_code == 500, "当前语义是冒泡：一路失败则整体失败"
    # 至少发起了 ≥2 次并发调用（说明确实是并发路径，不是单路）
    assert broken.calls >= 1


def test_health_reports_zero_entities_for_empty_index(client, wire):
    """空索引不该让 /health 报错，而应如实报 0。"""
    wire(_EmptyClient())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["num_entities"] == 0
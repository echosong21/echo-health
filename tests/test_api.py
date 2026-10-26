"""API 层测试 —— 3 端点各覆盖「正常 + 错误」。

**为什么不碰真实 Milvus**

`rag_query` 的依赖全部经 getter 取得（依赖注入改造的收益），
所以 monkeypatch 掉 `get_emb` / `get_client` / `get_llm` / `get_domain` 即可离线跑。
若这些仍是模块级对象，就必须起 Milvus 才能测 —— 那正是 R5 要解决的问题。

**状态码的选择理由（见 `src/api.py` docstring）**

- 空 question → **400**（schema 通过但语义无效）
- top_k=0 → **422**（schema 校验失败，FastAPI 自动）
- 依赖抛错 → **500**（内部错误）
- `/health` 探不到 Milvus → **503**（依赖不可用，非服务故障）
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import api as api_mod
from src import pipeline as pl


# ── 替身 ────────────────────────────────────────────────────
class _Client:
    def __init__(self, rows: int = 1230, stats_fail: bool = False, search_fail: bool = False):
        self.rows = rows
        self.stats_fail = stats_fail
        self.search_fail = search_fail
        self.search_calls = 0

    def get_collection_stats(self, collection_name):          # noqa: ARG002
        if self.stats_fail:
            raise RuntimeError("milvus down")
        return {"row_count": self.rows}

    def search(self, **_kwargs):
        if self.search_fail:
            raise TimeoutError("search timeout")
        self.search_calls += 1
        return [[{
            "distance": 0.81,
            "entity": {"source": "tables/food.md", "text": "米饭每100克含116千卡", "chunk_index": 0},
        }]]


class _Emb:
    def embed_query(self, _q):
        return [0.1] * 8


class _Resp:
    content = "米饭每 100 克含能量 116 千卡。"
    usage_metadata = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


class _LLM:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def invoke(self, _prompt):
        if self.fail:
            raise RuntimeError("llm down")
        return _Resp()


@pytest.fixture
def patch_all(monkeypatch, fake_emb, health_domain):
    """把 pipeline 的四个依赖换成替身，返回可调参的 client。"""
    client = _Client()

    def _apply(llm_fail: bool = False, stats_fail: bool = False, search_fail: bool = False):
        client.stats_fail = stats_fail
        client.search_fail = search_fail
        monkeypatch.setattr(pl, "get_emb", lambda: fake_emb)
        monkeypatch.setattr(pl, "get_client", lambda: client)
        monkeypatch.setattr(pl, "get_llm", lambda: _LLM(fail=llm_fail))
        monkeypatch.setattr(pl, "get_domain", lambda: health_domain)
        monkeypatch.setattr(pl, "get_collection", lambda: "echo_test")
        return client

    return _apply


@pytest.fixture
def client():
    return TestClient(api_mod.app)


# ── /health ─────────────────────────────────────────────────
def test_health_ok(client, patch_all):
    patch_all()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # 监测面板契约：必须有这两个字段
    assert body["collection"] == "echo_test"
    assert body["num_entities"] == 1230


def test_health_returns_503_when_milvus_down(client, patch_all):
    patch_all(stats_fail=True)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"


# ── /query ──────────────────────────────────────────────────
def test_query_ok(client, patch_all):
    patch_all()
    r = client.post("/query", json={"question": "米饭多少热量", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert "116" in body["answer"]
    assert len(body["contexts"]) >= 1
    assert set(body["stage_ms"]) >= {"embed_ms", "search_ms", "generate_ms"}
    assert body["safety"]["input"] == "pass"


def test_query_rejects_blank_question(client, patch_all):
    """空字符串通过 schema 但语义无效 ⇒ 400，不是 422。"""
    patch_all()
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400
    assert "空白" in r.json()["detail"]


def test_query_rejects_bad_top_k(client, patch_all):
    """top_k=0 违反 Field(ge=1) ⇒ FastAPI 自动 422。"""
    patch_all()
    r = client.post("/query", json={"question": "x", "top_k": 0})
    assert r.status_code == 422


def test_query_returns_500_when_llm_fails(client, patch_all):
    patch_all(llm_fail=True)
    r = client.post("/query", json={"question": "米饭多少热量"})
    assert r.status_code == 500
    assert "检索失败" in r.json()["detail"]


# ── /query_multi ────────────────────────────────────────────
def test_query_multi_ok(client, patch_all):
    patch_all()
    r = client.post("/query_multi", json={"question": "米饭多少热量", "per_form": 2})
    assert r.status_code == 200
    body = r.json()
    assert "116" in body["answer"]
    assert body["per_form"] == 2


def test_query_multi_rejects_blank(client, patch_all):
    patch_all()
    r = client.post("/query_multi", json={"question": ""})
    assert r.status_code == 400


def test_unknown_path_is_404(client, patch_all):
    patch_all()
    assert client.get("/nope").status_code == 404

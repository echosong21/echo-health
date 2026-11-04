"""API 层测试 —— 3 端点各覆盖「正常 + 错误」。

**为什么不碰真实 Milvus**

`rag_query` 的依赖全部经 getter 取得（依赖注入改造的收益），
所以 monkeypatch 掉 `get_emb` / `get_client` / `get_llm` / `get_domain` 即可离线跑。
若这些仍是模块级对象，就必须起 Milvus 才能测 —— 这正是本改造要解决的问题。

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
    """把 pipeline 的四个依赖换成替身，返回可调参的 client。

    ⚠️ 还要打桩 `api._probe_pg` —— 它**直接探 5432 端口**（不走 pipeline），
    不打桩的话测试结果会依赖「本机有没有起 PostgreSQL」，那是环境耦合。
    """
    client = _Client()

    def _apply(llm_fail: bool = False, stats_fail: bool = False, search_fail: bool = False,
               pg_up: bool = True):
        client.stats_fail = stats_fail
        client.search_fail = search_fail
        monkeypatch.setattr(pl, "get_emb", lambda: fake_emb)
        monkeypatch.setattr(pl, "get_client", lambda: client)
        monkeypatch.setattr(pl, "get_llm", lambda: _LLM(fail=llm_fail))
        monkeypatch.setattr(pl, "get_domain", lambda: health_domain)
        monkeypatch.setattr(pl, "get_collection", lambda: "echo_test")
        # 依赖探针打桩（真实实现会连 5432 / Milvus）
        monkeypatch.setattr(api_mod, "_probe_pg", lambda timeout=1.0: "up" if pg_up else "down")
        # ⚠️ _probe_milvus 必须跟随 stats_fail —— `_classify_failure` 用它
        # 区分「依赖挂了(503)」和「我的 bug(500)」，如果固定成 up，
        # 依赖类失败会被误判成 500。
        monkeypatch.setattr(
            api_mod, "_probe_milvus",
            lambda: ("down", None) if stats_fail else ("up", "echo_test"),
        )
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
    """`/query` 走**规划模式**，stage_ms 的字段与单路检索不同。

    ⚠️ 规划模式的阶段是 `plan_ms` / `retrieve_per_task_ms` / `generate_ms`，
    **没有** `embed_ms` / `search_ms` —— 因为每个子任务的 embed+search 是
    合在一起计时的（分开计会因多线程累加竞争而失真）。
    """
    patch_all()
    r = client.post("/query", json={"question": "米饭多少热量", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert "116" in body["answer"]
    assert len(body["contexts"]) >= 1
    assert set(body["stage_ms"]) >= {"plan_ms", "generate_ms"}
    assert body["safety"]["input"] == "pass"
    # 规划模式的附加字段
    assert "subtasks" in body and len(body["subtasks"]) >= 1
    assert body["mode"] == "planned"


def test_query_multi_focus_reports_subtasks(client, patch_all):
    """多焦点问题应拆成 ≥2 个子任务，并把拆分结果回给调用方。"""
    patch_all()
    r = client.post("/query", json={"question": "米饭的热量多少，适合高血压吗"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["subtasks"]) >= 2, f"应至少拆 2 个子任务，实际：{body['subtasks']}"


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


# ── /health 分级（依赖性质不同，状态码不同）──────────────────
def test_health_degraded_when_pg_down(client, patch_all):
    """Milvus 正常、PG 挂掉 ⇒ **200 + degraded**，不是 503。

    这是本设计的核心契约：**状态码由必需品（Milvus）决定**。
    记忆层有熔断降级，PG 挂了服务照常答 —— 对外报 503 等于把降级能力作废。

    ⚠️ 这条测试是 C1 验收项「≥1 条 pytest 覆盖 /health 的 degraded 分支」。
    """
    patch_all(pg_up=False)
    r = client.get("/health")
    assert r.status_code == 200, "可降级依赖挂掉不应影响整体状态码"
    body = r.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["milvus"] == "up"
    assert body["dependencies"]["postgres"] == "down"
    assert "记忆层" in body["detail"]


def test_health_unavailable_when_milvus_down_even_if_pg_up(client, patch_all):
    """Milvus 挂 ⇒ 503，**即使 PG 正常** —— 必需品缺失优先。"""
    patch_all(stats_fail=True, pg_up=True)
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["dependencies"]["milvus"] == "down"
    assert body["dependencies"]["postgres"] == "up"


def test_health_reports_both_dependencies(client, patch_all):
    """两个依赖必须**分别报告** —— 只回一个总状态无法定位是谁挂了。"""
    patch_all()
    body = client.get("/health").json()
    assert set(body["dependencies"]) == {"milvus", "postgres"}


# ── /tools ──────────────────────────────────────────────────
def test_tools_endpoint_lists_all(client, patch_all):
    patch_all()
    r = client.get("/tools")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(api_mod.TOOLS) == 7
    names = [t["name"] for t in body["tools"]]
    assert "common.discovery.health_check" in names
    assert len(set(names)) == len(names), "工具名不能重复"


def test_tools_have_required_fields(client, patch_all):
    """每个工具必须有 name / scope / side_effect / brief 四个字段。"""
    patch_all()
    for t in client.get("/tools").json()["tools"]:
        assert set(t) == {"name", "scope", "side_effect", "brief"}, f"{t} 字段不全"
        assert t["side_effect"] in {"read", "write"}


def test_tools_mirror_matches_registry():
    """🔴 静态镜像必须与真实注册表一致。

    `/tools` 用静态镜像（运行时不能跨仓读文件），代价是**可能漂移**。
    这条测试就是防漂移的守卫 —— 注册表加了工具而 `api.TOOLS` 没同步，这里会红。

    ⚠️ **路径不写死**：注册表在另一个仓库，目录名属于本地开发布局，不该出现在
    可公开的代码里。改为**从本文件向上逐级搜索** `*/domain-rag-mcp/registry/registry.yaml`。
    找不到就 skip，而不是失败 —— 单独 clone 本仓跑测试不该红得莫名其妙。
    """
    import os
    from pathlib import Path

    import yaml

    # ① 显式指定优先（CI 或特殊布局用）
    candidates = []
    if env_path := os.environ.get("MCP_TOOL_REGISTRY"):
        candidates.append(Path(env_path))
    # ② 向上搜索（开发布局下能找到，不写死目录名）
    for ancestor in Path(__file__).resolve().parents:
        candidates.extend(sorted(ancestor.glob("*/domain-rag-mcp/registry/registry.yaml")))
        if candidates:
            break

    registry = next((p for p in candidates if p.exists()), None)
    if registry is None:
        pytest.skip("未找到工具注册表（可用 MCP_TOOL_REGISTRY 指定），跳过一致性检查")

    data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    real = [t["name"] for t in data["tools"]]
    mirror = [t["name"] for t in api_mod.TOOLS]
    assert mirror == real, f"镜像与注册表不一致\n镜像：{mirror}\n注册表：{real}"


# ── 依赖故障 ⇒ 503（不是 500）────────────────────────────────
def test_query_returns_503_when_dependency_unavailable(client, patch_all):
    """依赖层故障（连接/超时）⇒ **503**，不是 500。

    区分标准：503 = 依赖不可用（不是我的错），500 = 我自己的 bug。
    调用方据此决定「等会儿重试」还是「报 bug」。

    ⚠️ 这是 C1 验收项「依赖不可用时 /query 返回 503 而非 500」。
    """

    def _boom(*_a, **_kw):
        raise ConnectionError("milvus unreachable")

    patch_all()
    import src.pipeline as _pl
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(_pl, "rag_query_planned", _boom)
        r = client.post("/query", json={"question": "米饭多少热量"})
    assert r.status_code == 503, f"依赖故障应回 503，实际 {r.status_code}"
    assert "依赖不可用" in r.json()["detail"]


def test_query_returns_500_on_non_dependency_error(client, patch_all):
    """非依赖类异常仍回 500 —— 不能把自家 bug 说成「依赖挂了」。"""
    patch_all()

    def _boom(*_a, **_kw):
        raise ValueError("unexpected internal bug")

    import src.pipeline as _pl
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(_pl, "rag_query_planned", _boom)
        r = client.post("/query", json={"question": "米饭多少热量"})
    assert r.status_code == 500

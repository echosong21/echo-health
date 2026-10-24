"""retrieval 层的纯逻辑单测（用假 client，不连 Milvus）。

**为什么这些测试以前写不出来**

    改造前 `retrieval.py` 与 Milvus 强绑，只有起真实服务才能验证。
    依赖注入（`client` 作为函数第一参数）之后，假 client 就能覆盖
    分流、限量、合并、排序这些**纯逻辑**。

    ⚠️ 真实 Milvus 的行为（`filter` 表达式语法、索引类型、COSINE 距离）
    仍需要集成测试，不在这里的覆盖范围内 —— 假 client 对这些一无所知。

**这些测试同时是 R5 改造的回归防线**：

    如果 C1 重构把「每形态各取 per_form 条」改成「全局取 per_form 条」，
    `test_retrieve_dense_multi_queries_each_form` 会立刻失败。
"""
from __future__ import annotations

import json

from src import retrieval as rt


# ── 适配层 ────────────────────────────────────────────────────────
def test_to_raw_maps_milvus_shape(make_hits):
    """Milvus 的 `distance`/`entity` 结构 → 内部统一的 score/source/text。"""
    raw = rt._to_raw(make_hits(n=2, form="faq", score_start=0.88, step=0.05))

    assert [r["score"] for r in raw] == [0.88, 0.83]
    assert raw[0]["source"] == "faq/doc_0.md"
    assert raw[0]["chunk_index"] == 0
    assert raw[0]["text"].startswith("这是 faq")


# ── 单路召回 ──────────────────────────────────────────────────────
def test_retrieve_dense_single_respects_top_k(fake_client, make_hits):
    """single 模式：只取 top_k 条，limit 要如实传给 client。"""
    fake_client.hits_by_form[None] = make_hits(n=5)

    raw, ms = rt.retrieve_dense_single(fake_client, [0.1] * 4, "t", top_k=3)

    assert len(raw) == 3
    assert ms >= 0
    assert fake_client.calls[0]["limit"] == 3
    assert fake_client.calls[0]["filter"] is None      # single 不做形态过滤


# ── 多路召回（核心）───────────────────────────────────────────────
def test_retrieve_dense_multi_queries_each_form(fake_client, make_hits):
    """🔴 核心行为：每个形态各发一次请求、各取 per_form 条，然后合并。

    这是依赖注入改造的直接收益 —— 改造前无法脱离 Milvus 验证。
    """
    forms = ["guides", "faq"]
    for f in forms:
        fake_client.hits_by_form[f] = make_hits(n=4, form=f)

    merged, per_form_ms, wall_ms = rt.retrieve_dense_multi(
        fake_client, [0.1] * 4, "t", per_form=2, forms=forms,
    )

    # ① 每形态各发一次
    assert len(fake_client.calls) == 2
    # ② filter 带上了正确的形态前缀
    assert sorted(c["filter"] for c in fake_client.calls) == [
        'source like "faq/%"',
        'source like "guides/%"',
    ]
    # ③ 每形态 2 条 → 合并 4 条
    assert len(merged) == 4
    assert set(per_form_ms) == set(forms)
    assert wall_ms >= 0


def test_retrieve_dense_multi_respects_per_form_limit(fake_client, make_hits):
    """per_form 是「每形态上限」，不是「总数上限」。"""
    fake_client.hits_by_form["guides"] = make_hits(n=10, form="guides")

    merged, _, _ = rt.retrieve_dense_multi(
        fake_client, [0.1] * 4, "t", per_form=1, forms=["guides"],
    )

    assert len(merged) == 1
    assert fake_client.calls[0]["limit"] == 1


# ── 观测层 ────────────────────────────────────────────────────────
def test_write_trace_goes_to_tmp_file(tmp_trace_file):
    """trace 写到临时文件，不污染真实 `logs/traces.jsonl`。"""
    from src import observe

    observe.write_trace({"trace": "abc123", "status": "ok"})

    lines = tmp_trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace"] == "abc123"


# ── 夹具自身 ──────────────────────────────────────────────────────
def test_fake_embeddings_returns_fixed_dimension(fake_emb):
    """假 embedding 只保证形状正确，不保证语义 —— 这正是它该做的事。"""
    vec = fake_emb.embed_query("米饭多少热量")

    assert len(vec) == fake_emb.dim
    assert fake_emb.queries == ["米饭多少热量"]


def test_health_domain_is_loadable(health_domain):
    """领域配置能读出关键字段（collection 决定检索打哪个集合）。"""
    assert health_domain["collection"] == "echo_health_v2"
    assert health_domain["prompt"]["no_answer"]
    assert health_domain["forms"]
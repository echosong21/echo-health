"""检索层：向量召回、稀疏召回与混合检索，供**服务与评测共用**。

**为什么抽成独立模块**
    服务（`src/pipeline.py`）和评测（`evals/run_eval.py`）都需要检索。如果两边各写一份，
    **它们的行为会悄悄分叉** —— 评测出来的结论就不再代表线上行为，而这正是评测要避免的。
    因此两条路径都调用本模块。

**三种召回方式**
    `retrieve_dense_single` —— 单路向量召回（原 `--mode single` 的行为，保持不动）
    `retrieve_dense_multi`  —— 按形态多路向量召回。**并发**执行，替代原来的串行 4 次请求
                              （原来的串行是 N+1 式的浪费：4 次请求的延迟被逐个相加）
    `retrieve_hybrid`       —— 向量 + 稀疏（BM25）→ RRF 融合

**⚠️ 关于混合模式的分数尺度**
    RRF 分数（约 0.01–0.03）与余弦分（0–1）**不在同一尺度**，因此混合模式下的 `score`
    是 RRF 分数，**不能用原来的 `min_score` 去过滤**。每条结果仍保留 `vec_score`
    （若该块被向量路召回），需要按相似度过滤时用它。
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from src import bm25 as bm25_mod  # noqa: E402
from src import fusion as fusion_mod  # noqa: E402
from src import domain as domain_mod  # noqa: E402


_forms_cache: list[str] | None = None


def _form_keys() -> list[str]:
    """当前领域的形态列表（来自 `domains/*.yaml`，能力层不写死）。

    ⚠️ 原先是模块级 `FORMS = _form_keys()`，而它同时被当作**函数默认参数**：

        def retrieve_dense_multi(..., forms: list[str] = FORMS)

    Python 的默认参数在**函数定义时**求值，所以 `FORMS` 必须在 import 阶段
    就有值 —— 这迫使整个领域配置在模块级加载（`import` 即读盘 + 校验）。

    改成「惰性 + 缓存」后：import 不再触碰 `domains/`，首次真正调用才加载，
    且只加载一次（领域配置在进程生命周期内不变）。
    """
    global _forms_cache
    if _forms_cache is None:
        _forms_cache = domain_mod.form_keys(domain_mod.load())
    return _forms_cache

# 形态内融合时两路的权重。
#
# ⚠️ 实测（2026-10-20）：**给 dense 加权对评测指标没有影响** ——
#    1:1 / 2:1 / 3:1 / 5:1 四组的命中率、top1、MRR **完全相同**（17/17、12/17、0.7737）。
#
# 真实原因（经实测确认；此前一度归因错误，故在此写明）：
#    dense 与 BM25 的候选**几乎不重合** —— 68 组形态配对里 **0 组同序**，
#    候选交集仅 33/408 ≈ 8%，平均每组只重合 0.49/3 条。
#    两路不重合时，RRF 分数由 (w_d)/(k+rank_d) 与 (w_s)/(k+rank_s) 两类项混合而成；
#    而 dense 在**靠前名次**上占优 ⇒ **前若干名始终由 dense 占据**，
#    加权改变的是**靠后部分**的相对位置，top-N 的构成与顺序都不变。
#
# ⇒ 教训：**加权只在"两路的靠前名次会互相竞争"时才起作用**。
#    若某一路在前部全面占优，调权重只是放大已有优势，看不出效果。
#
# 加权实现本身是正确的 —— 用完全分歧的两路验证（dense=A,B,C / sparse=C,B,A）：
#    权重 1.0 → A,C,B；权重 2.0 / 5.0 → A,B,C。
ROUTE_WEIGHTS = {"dense": 1.0, "sparse": 1.0}


def _to_raw(hits: list) -> list[dict]:
    """把 Milvus 返回的命中统一成内部结构。"""
    return [
        {
            "score": round(h["distance"], 4),
            "source": h["entity"]["source"],
            "chunk_index": h["entity"]["chunk_index"],
            "text": h["entity"]["text"],
        }
        for h in hits
    ]


def retrieve_dense_single(client, qvec, collection: str, top_k: int) -> tuple[list[dict], float]:
    """单路向量召回（与改造前行为一致，用于保持基线可比）。"""
    t0 = time.perf_counter()
    hits = client.search(
        collection_name=collection, data=[qvec], limit=top_k,
        output_fields=["text", "source", "chunk_index"],
    )[0]
    return _to_raw(hits), round((time.perf_counter() - t0) * 1000, 1)


def retrieve_dense_multi(
    client, qvec, collection: str, per_form: int,
    forms: list[str] | None = None, max_workers: int = 4,
) -> tuple[list[dict], dict, float]:
    """按形态多路向量召回 —— **并发**。

    替代原来的 4 次串行请求：串行时总延迟是 4 次的**加和**，并发后接近**最大值**。
    语义完全不变（仍是每形态各取 per_form 条），只是不再等待彼此。

    返回 `(merged, per_form_ms, wall_ms)`。
    """
    forms = _form_keys() if forms is None else forms

    def one(form: str):
        t = time.perf_counter()
        hits = client.search(
            collection_name=collection, data=[qvec], limit=per_form,
            filter=f'source like "{form}/%"',
            output_fields=["text", "source", "chunk_index"],
        )[0]
        return form, _to_raw(hits), round((time.perf_counter() - t) * 1000, 1)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(one, forms))
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)

    merged: list[dict] = []
    per_form_ms: dict[str, float] = {}
    for form, raw, ms in results:
        per_form_ms[form] = ms
        merged.extend(raw)
    return merged, per_form_ms, wall_ms


def retrieve_hybrid(
    client, qvec, query: str, collection: str, per_form: int,
    bm25_bundle: tuple, forms: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """混合检索：向量（并发）+ 稀疏（BM25）→ RRF 融合。

    `bm25_bundle` 是 `(bm25, items, form_index)`（来自 `bm25.build_or_load`）。

    返回 `(fused, stage_ms)`，`stage_ms` 含 dense / sparse / fuse 三段耗时 ——
    **三段必须分开记录**，否则无法判断混合检索把时间花在了哪里。
    """
    bm25, items, form_index = bm25_bundle
    forms = _form_keys() if forms is None else forms
    stage: dict[str, float] = {}

    dense, per_form_ms, wall_ms = retrieve_dense_multi(client, qvec, collection, per_form, forms)
    stage["dense_ms"] = wall_ms
    stage["dense_per_form_ms"] = per_form_ms

    t0 = time.perf_counter()
    sparse_routes: dict[str, list[dict]] = {}
    for form in forms:
        hits, _ms, _tokens = bm25_mod.search(
            query, bm25, items, top_k=per_form, forms=[form], form_index=form_index,
        )
        sparse_routes[form] = hits
    stage["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── 两层融合（这是实测逼出来的设计，见下方说明）────────────────────
    #
    # ⚠️ 第一版是"4 形态 × 2 检索器 = 8 路一起 RRF"，实测**排名质量反而变差**：
    #      期待来源居首 70.6% → 47.1%，MRR 0.8284 → 0.6536
    #    其中表格类受损最重（T1/T2/T3 从第 1 位掉到第 7/8/7 位）。
    #    根因：RRF 对**每一路的第 1 名一视同仁**，而各形态的可信度并不相同 ——
    #    tables 的路内第 1 名相似度只有约 0.55，faq 的却有 0.72，
    #    两者却拿到同一个 RRF 分 ⇒ 弱形态的"路内第一"挤掉了强形态的正确结果。
    #
    # ⇒ 改成两层：
    #    层 1（形态内）：dense + sparse 用 RRF 融合 —— **保留 sparse 的召回补充**
    #    层 2（形态间）：按该形态的**最高向量相似度**排序 —— **保证强形态优先**
    #
    # 这样既拿到了混合检索多召回的好处，又不会被弱形态的路内第一带偏。
    t0 = time.perf_counter()
    form_fused: dict[str, list[dict]] = {}
    form_top_score: dict[str, float] = {}
    for form in forms:
        d = [x for x in dense if x["source"].startswith(form + "/")]
        form_fused[form] = fusion_mod.rrf_fuse(
            {f"dense:{form}": d, f"sparse:{form}": sparse_routes[form]},
            weights=ROUTE_WEIGHTS,
        )
        form_top_score[form] = max([x["score"] for x in d], default=0.0)

    # 形态间：向量相似度高的形态整体优先（相似度是**绝对**可比的质量信号）
    form_order = sorted(forms, key=lambda f: form_top_score[f], reverse=True)

    vec_scores = {(d["source"], d["chunk_index"]): d["score"] for d in dense}
    out: list[dict] = []
    for form in form_order:
        for item in form_fused[form]:
            doc = dict(item["doc"])
            doc["score"] = item["score"]           # RRF 分数（形态内可比）
            doc["vec_score"] = vec_scores.get(item["key"])
            doc["ranks"] = item["ranks"]
            doc["form_rank"] = form_order.index(form) + 1
            out.append(doc)
    stage["fuse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    stage["form_order"] = form_order

    stage["n_dense"] = len(dense)
    stage["n_sparse"] = sum(len(v) for v in sparse_routes.values())
    stage["n_fused"] = len(out)
    return out, stage
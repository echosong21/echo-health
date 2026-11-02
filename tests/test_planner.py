"""规划层测试。

**最核心的一条是 `test_multi_focus_question_triggers_two_retrievals`** ——
它断言「同一个问题下，plan 至少 2 个子任务被检索」。

计划里明确要求：这是「规划真的发生了」的**唯一硬证据**。没有它，
规划层可以「看起来在拆」但实际只检索一次（比如拆了却没用）。

**为什么用假检索器而不是真 Milvus**

`plan_and_retrieve` 的 `retrieve_one` 是注入的 —— 这是刻意的设计：
规划逻辑的正确性不应该依赖「Milvus 能不能连上」。真检索留给端到端脚本。
"""
from __future__ import annotations

import pytest

from src import planner


# ── 拆分规则 ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "question,expected",
    [
        # 并列连接词
        ("高血压和糖尿病该注意什么", 2),
        ("维生素C和铁的吸收情况以及它们的相互作用", 2),
        ("高血压、糖尿病、高血脂分别该注意什么", 2),
        # 双疑问焦点
        ("大白菜热量多少，适合高血压吗", 2),
        # 不该拆的
        ("成年人每天盐摄入量应该低于多少", 1),
        ("儿童肥胖怎么办", 1),
        ("红烧肉", 1),
        # 「和」是动词，不是连接词
        ("和面要注意什么", 1),
    ],
)
def test_split_counts(question, expected):
    assert len(planner.split(question)) == expected


def test_short_question_never_splits():
    """长度 < 8 字的问题一律不拆 —— 这是四条回退条件之一。"""
    assert planner.split("红烧肉") == ["红烧肉"]


def test_single_focus_never_splits():
    assert planner.split("成年人每天盐摄入量应该低于多少") == ["成年人每天盐摄入量应该低于多少"]


def test_high_top1_score_skips_planning():
    """单轮 top-1 已高于阈值就不拆（拆了反而稀释召回）。"""
    assert planner.should_plan("高血压和糖尿病该注意什么", top1_score=0.9) is False
    assert planner.should_plan("高血压和糖尿病该注意什么", top1_score=0.5) is True


def test_subtask_count_capped():
    """拆出的子任务数不超过上限 3 —— 再多会稀释每个子任务的召回质量。"""
    q = "高血压、糖尿病、高血脂、高尿酸、脂肪肝分别该注意什么"
    assert len(planner.split(q)) <= planner.MAX_SUBTASKS


def test_split_returns_original_when_cannot_split():
    """拆不出 ≥2 段时必须**原样返回** [原问题]，不能返回空列表。

    返回空列表会让下游「无子任务可检索」→ 直接崩。
    """
    for q in ("红烧肉", "儿童肥胖怎么办", "和面要注意什么"):
        assert planner.split(q) == [q]


# ── 端到端（含检索 + 融合）─────────────────────────────────
def _fake_retriever(calls: list[str], *, top_k: int = 2):
    """构造一个记录调用的假检索器。

    **它记录每次被调用的查询词** —— 这就是「规划真的发生了」的证据：
    如果只调用 1 次，说明拆分没生效或没被用上。
    """

    def retrieve(question: str) -> list[dict]:
        calls.append(question)
        # 用查询词做 source，保证不同子任务返回不同块（避免融合时被当成同一条）
        return [
            {
                "source": f"fake/{question}.md",
                "chunk_index": i,
                "text": f"{question} 的第 {i} 条",
                "score": 0.9 - i * 0.1,
            }
            for i in range(top_k)
        ]

    return retrieve


def test_multi_focus_question_triggers_two_retrievals():
    """🔴 核心断言：多焦点问题下，**至少 2 个子任务被实际检索**。

    这是「规划真的发生了」的唯一硬证据 —— 没有这条，
    规划层可以「看起来拆了」但实际只检索一次。
    """
    calls: list[str] = []
    out = planner.plan_and_retrieve("高血压和糖尿病该注意什么", _fake_retriever(calls))

    assert len(calls) >= 2, f"应至少检索 2 个子任务，实际只检索 {len(calls)} 次：{calls}"
    assert len(out["subtasks"]) >= 2
    # 融合后的结果里必须能看到来自不同子任务的条目
    tasks_in_result = {d["task"] for d in out["results"]}
    assert len(tasks_in_result) >= 2, "融合结果应包含来自多个子任务的条目"


def test_single_focus_question_retrieves_once():
    """单焦点问题只检索一次 —— 不能为了「看起来在规划」而多查。"""
    calls: list[str] = []
    planner.plan_and_retrieve("成年人每天盐摄入量应该低于多少", _fake_retriever(calls))
    assert len(calls) == 1, f"单焦点问题应只检索 1 次，实际 {len(calls)} 次：{calls}"


def test_rrf_fusion_marks_task_and_rank():
    """融合结果要带 `task` / `task_rank` —— 否则无法解释「这条是谁顶上来的」。"""
    calls: list[str] = []
    out = planner.plan_and_retrieve("高血压和糖尿病该注意什么", _fake_retriever(calls))

    for doc in out["results"]:
        assert "task" in doc, "每条结果必须标明来自哪个子任务"
        assert "task_rank" in doc, "每条结果必须标明在该子任务内的名次"
        assert isinstance(doc["task_rank"], int) and doc["task_rank"] >= 1


def test_results_sorted_by_rrf_score():
    """融合后按 RRF 分数降序 —— 分数不可比时靠排名，这是 RRF 的全部意义。"""
    calls: list[str] = []
    out = planner.plan_and_retrieve("高血压和糖尿病该注意什么", _fake_retriever(calls))
    scores = [d["score"] for d in out["results"]]
    assert scores == sorted(scores, reverse=True)


def test_per_task_n_reports_raw_counts():
    """`per_task_n` 要能看出每个子任务原始召回了几条（排查「某个子任务零召回」用）。"""
    calls: list[str] = []
    out = planner.plan_and_retrieve("高血压和糖尿病该注意什么", _fake_retriever(calls, top_k=3))
    assert set(out["per_task_n"].keys()) == set(out["subtasks"])
    assert all(n == 3 for n in out["per_task_n"].values())


def test_empty_retrieval_does_not_crash():
    """某个子任务零召回时不能崩 —— 融合要能处理空路。"""

    def empty_retriever(question: str) -> list[dict]:
        return []

    out = planner.plan_and_retrieve("高血压和糖尿病该注意什么", empty_retriever)
    assert out["results"] == []
    assert len(out["subtasks"]) >= 2


def test_duplicate_docs_across_tasks_are_merged():
    """不同子任务命中同一个块时，RRF 会把它们合并成一条（而不是出现两次）。

    这正是 RRF 相对于「直接拼接」的价值：`(source, chunk_index)` 相同的
    文档在多路里出现，其分数会累加，而不是产生重复条目。
    """

    def same_doc_retriever(question: str) -> list[dict]:
        return [{"source": "guides/盐钠钾.md", "chunk_index": 0, "text": "同一块", "score": 0.7}]

    out = planner.plan_and_retrieve("高血压和糖尿病该注意什么", same_doc_retriever)
    # 两个子任务各返回同一条 ⇒ 融合后应去重成 1 条
    assert len(out["results"]) == 1, f"应去重为 1 条，实际 {len(out['results'])} 条"
    # 且它在该条上的 ranks 应该有 2 个来源
    assert len(out["results"][0]["ranks"]) >= 2
"""规划层：把多焦点问题拆成子任务，各自检索，再合并。

**解决什么问题**

单轮检索只能接收**一个查询**。「高血压和糖尿病该注意什么」这种多焦点问题，
单轮只能命中其中一半 —— 不是检索坏了，是**问题本身有多个检索意图**。

**为什么拆分是规则驱动，不是 LLM 驱动**

LLM 拆分覆盖句式更广，但**不可判定**：同一个问题两次运行可能拆出不同子任务，
导致检索路径不同、答案不同。这在评审时写不出断言，在线上无法解释
「为什么这次答得不一样」。

规则型覆盖窄，但**同输入必得同输出**，这让「规划真的发生了」可以用测试断言。
窄的部分用「回退率」监控 —— 回退率高了说明规则保守，那时才值得引入 LLM。

**为什么合并用 RRF 而不是原始 score**

不同子任务的向量相似度**不在同一尺度**：「红烧肉热量」与「高血压饮食原则」
的分数不可比，按分数混排等于假设它们可比，这个假设是错的。

RRF 只看排名，天然规避。

⚠️ **但 `retrieval.py` 有一个反面实测**（L163-174）：它曾把「4 形态 × 2 检索器」
共 8 路一起 RRF，结果排名质量**反而变差**（top-1 70.6% → 47.1%，MRR 0.8284 → 0.6536）。
根因是**RRF 对每一路的第 1 名一视同仁，而各路可信度并不相同**。

本模块的子任务之间**理论上存在同类风险**（「高血压饮食原则」可能比
「大白菜热量」更容易命中的好结果）。差别在于：
  · 8 路那次是**异构来源**（向量 vs BM25，尺度与可靠性差异极大）
  · 这里的分支是**同构来源**（都是同一个向量检索器，只是查询不同）
但风险不完全为零 —— 所以 `plan_and_retrieve` 保留了每条结果的
`task_rank` 与 `vec_score`，**一旦发现子任务间质量差异显著，可以退化成
「按子任务顺序分组 + 组内 RRF」**（见 `DESIGN.md` §5 的「何时该换」）。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

# ── 拆分参数（与 DESIGN.md §5 一一对应，改这里要同步改文档）──────
MAX_SUBTASKS = 3
MAX_BY_CONJUNCTION = 2

# 并列连接词。注意「和」在中文里高频但歧义多（「和面」「温和」），
# 所以只在两侧都有实际内容且整体疑问结构成立时才拆 —— 见 `_split_by_conjunction`。
CONJUNCTIONS = ("以及", "还有", "、", "和")

# 回退阈值：单轮 top-1 已高于这个分数就不拆（拆了反而稀释）
FALLBACK_TOP1_SCORE = 0.85

# 问题长度下限：更短的问题几乎不可能有多焦点
MIN_QUESTION_LEN = 8


def _clean(part: str) -> str:
    return part.strip(" \t　，。？?！!；;：:")


def _split_by_conjunction(question: str) -> list[str]:
    """按并列连接词拆。

    ⚠️ 「和」的歧义处理：「和面」「温和」「共和国」里的「和」不是连接词。
    这里用一条保守规则：**拆开后每一段都必须有实际内容**（长度 ≥ 2），
    否则放弃这次拆分。宁可漏拆，不可错拆 —— 错拆会引入与问题无关的检索。

    ⚠️ 阈值为什么是 2 而不是 4：实测「高血压和糖尿病该注意什么」这类句子，
    两侧主语都是 3 字（高血压 / 糖尿病），阈值设 4 会把两侧都筛掉、
    退回成「不拆」。设 2 既能保住这个常见句式，又挡得住「和面」这种
    （拆出来第一段是空串，长度 0 被过滤后只剩 1 段 ⇒ 自动不拆）。

    ⚠️ 超出上限时**合并多余项**而不是丢弃：用户问了三件事，只检索两件
    等于漏答一件。合并后第二段是「糖尿病、高血脂分别该注意什么」——
    向量检索对词组友好，能同时命中两类的文档。
    """
    for conj in CONJUNCTIONS:
        if conj not in question:
            continue
        parts = [_clean(p) for p in question.split(conj)]
        parts = [p for p in parts if len(p) >= 2]
        if len(parts) < 2:
            continue
        if len(parts) > MAX_BY_CONJUNCTION:
            # 保留前 N-1 段，其余用顿号合并进最后一段
            head = parts[: MAX_BY_CONJUNCTION - 1]
            tail = "、".join(parts[MAX_BY_CONJUNCTION - 1:])
            parts = [*head, tail]
        return parts
    return []


def _split_by_dual_focus(question: str) -> list[str]:
    """双疑问焦点：同一句里同时问「数值」和「判断」。

    例：「大白菜热量多少，适合高血压吗」→
        ["大白菜热量多少", "大白菜适合高血压吗"]

    ⚠️ 这条规则比连接词更保守：只在**同一主语**下出现两类焦点时才拆。
    否则会把「热量多少合适」这种单焦点问题误拆。
    """
    # 数值型焦点
    has_value = bool(re.search(r"(多少|几克|几卡|多少卡|含量|热量|蛋白质|脂肪)", question))
    # 判断型焦点
    has_judge = bool(re.search(r"(适合|能不能|可以吃|该不该|有没有|是否)", question))
    if not (has_value and has_judge):
        return []

    # 用逗号/问号切，看能不能自然分成两段
    parts = [_clean(p) for p in re.split(r"[，,；;]|\？|\?", question)]
    parts = [p for p in parts if len(p) >= 4]
    if len(parts) < 2:
        return []

    # 主语取第一段的前缀（到第一个焦点词之前），用于补全第二段
    head = parts[0]
    head = re.split(r"(多少|几克|几卡|含量|热量|蛋白质|脂肪)", head)[0].strip()
    second = parts[1]
    if head and not second.startswith(head):
        second = f"{head}{second}"
    return [parts[0], _clean(second)][:MAX_SUBTASKS]


def should_plan(question: str, top1_score: float | None = None) -> bool:
    """是否值得拆分。**四条回退条件，任一成立就不拆。**

    这四条都是可判定的 —— 计划里明令不许出现「视情况」「看复杂度」。
    """
    if top1_score is not None and top1_score >= FALLBACK_TOP1_SCORE:
        return False        # 单轮已经很好，拆了反而稀释
    if len(question.strip()) < MIN_QUESTION_LEN:
        return False        # 太短，不可能多焦点
    return True             # 真正的判定在 split() —— 拆不出 ≥2 段就自动回退


def split(question: str) -> list[str]:
    """拆成子任务。**拆不出 ≥2 个就返回 [原问题]**（长度 1 = 不拆）。

    这是本模块唯一的对外契约：**返回 1..N 个子问题**。
    内部是规则还是 LLM、同步还是异步，调用方不需要知道 ——
    将来换成 LLM 拆分只改这个函数内部。
    """
    if not should_plan(question):
        return [question]

    for splitter in (_split_by_conjunction, _split_by_dual_focus):
        parts = splitter(question)
        if len(parts) >= 2:
            return parts[:MAX_SUBTASKS]
    return [question]


def plan_and_retrieve(
    question: str,
    retrieve_one,
    *,
    top1_score: float | None = None,
    max_workers: int = 3,
) -> dict:
    """拆分 → 并行检索 → RRF 合并。

    参数 `retrieve_one`：`(子问题) -> list[dict]`，每个 dict 至少含
    `source` / `chunk_index` / `score`。**由调用方注入**，
    这样本函数不依赖 Milvus、可以在测试里用假检索器跑。

    返回：
        {
          "subtasks": [子问题...],
          "results":  [合并后的结果...],
          "per_task_n": {子问题: 原始条数},
        }

    ⚠️ 为什么 `retrieve_one` 是注入的而不是内部 new 一个 client：
    `retrieval.py` 已经有线程池与连接管理，这里再造一个会重复。
    注入也让单测不需要起 Milvus。
    """
    subtasks = split(question)

    # 只有一个子任务 ⇒ 根本没拆，直接单轮（省掉一次线程池开销）
    if len(subtasks) == 1:
        raw = retrieve_one(question)
        return {
            "subtasks": subtasks,
            "results": raw,
            "per_task_n": {question: len(raw)},
        }

    def one(sub: str) -> tuple[str, list[dict]]:
        return sub, retrieve_one(sub)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        per_task = dict(pool.map(one, subtasks))

    # 每条结果标上它来自哪个子任务、在该子任务内的名次
    route_docs: dict[str, list[dict]] = {}
    for sub, docs in per_task.items():
        tagged = []
        for rank, doc in enumerate(docs, start=1):
            d = dict(doc)
            d["task"] = sub
            d["task_rank"] = rank
            tagged.append(d)
        route_docs[sub] = tagged

    from src import fusion

    fused = fusion.rrf_fuse(route_docs)
    results = []
    for item in fused:
        doc = dict(item["doc"])
        doc["score"] = item["score"]        # RRF 分数（跨子任务可比）
        doc["ranks"] = item["ranks"]        # 各子任务里的名次，便于解释谁把它顶上来
        results.append(doc)

    return {
        "subtasks": subtasks,
        "results": results,
        "per_task_n": {k: len(v) for k, v in per_task.items()},
    }
"""安全拦截层：把 `domains/*.yaml` 的 `safety` 段变成**可执行的检查**。

**为什么单独一层**

    规则写进提示词只是"建议"，模型可以不听；写成代码是"闸门"，不过就拦下。
    本模块不依赖模型、不猜语义 —— 它只做**确定性的字符串判定**，
    因此可以被单元测试完全覆盖，也能在每次请求里以近乎零成本执行。

**三道闸**

    check_input(question)         输入侧：命中 `must_refer` ⇒ 强制插入转介语
    check_output(answer)          输出侧：命中 `forbidden_claims` ⇒ 整条拒答
    check_plan_delta(prev, new)   计划侧：变动幅度超过 `max_plan_delta_per_day` ⇒ 拒绝

**刻意的保守性（宁可误杀）**

    子串匹配会误伤否定句（`"无需就医"` 会命中 `"并非无需就医"`）。
    安全场景下，**误杀（拒答一个合法问题）比漏放（给出违规建议）代价小得多**，
    所以本版不做语义判断 —— 这个局限写进 DESIGN 文档，不假装不存在。

**规则词不在本文件里**

    `治愈` / `胸痛` / `无需就医` 全部来自领域配置。本模块只执行规则，
    换领域时规则随之改变，代码无需改动。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import domain as domain_mod


@dataclass
class SafetyVerdict:
    """一次安全判定的结果。

    `allowed=False` 时，调用方**必须**按 `action` 处理：
    - `block`：整条拒答，用 `message` 作为回复
    - `refer`：在原回答前插入 `message`（转介语）
    """

    allowed: bool
    action: str = "pass"                    # pass | block | refer
    reasons: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "reasons": self.reasons,
            "matched": self.matched,
            "message": self.message,
        }


# ── 输入侧：转介 ────────────────────────────────────────────────
def check_input(question: str, cfg: dict | None = None) -> SafetyVerdict:
    """检查用户问题是否触发转介规则（`safety.must_refer`）。

    命中即返回 `action="refer"`，调用方应把 `message`（即 `action` 字段的文案）
    插入到最终回复的最前面 —— **不管模型生成了什么**。
    """
    cfg = cfg or domain_mod.load()
    rules = cfg.get("safety", {}).get("must_refer") or []
    text = (question or "").lower()

    for rule in rules:
        when = (rule.get("when") or "").lower()
        action = rule.get("action") or ""
        if not when or not action:
            continue
        # `when` 支持用顿号/逗号分隔多个触发词（如"胸痛、呼吸困难、意识改变"）
        raw_triggers = [t.strip() for t in when.replace("，", "、").replace(",", "、").split("、")
                        if t.strip()]
        # ⚠️ 剥掉引导词：`when` 里常写成"提到胸痛、呼吸困难"，
        #    按顿号切分后第一段会带"提到"，导致真实信号词匹配不上（已实测踩到）。
        triggers = []
        for t in raw_triggers:
            t = re.sub(r"^(提到|出现|伴随|发生|有|如果|当)", "", t).strip()
            if t:
                triggers.append(t)
        hits = [t for t in triggers if t in text]
        if hits:
            return SafetyVerdict(
                allowed=False,
                action="refer",
                reasons=[f"命中转介条件：{when}"],
                matched=hits,
                message=action,
            )
    return SafetyVerdict(allowed=True)


# ── 输出侧：拦截 ────────────────────────────────────────────────
def check_output(answer: str, cfg: dict | None = None) -> SafetyVerdict:
    """检查模型回答是否包含禁用断言（`safety.forbidden_claims`）。

    命中即 `action="block"` —— 调用方应**整条拒答**，而不是改写后返回。
    改写会引入新的、未经验证的表达；拒答是确定性的。
    """
    cfg = cfg or domain_mod.load()
    claims = cfg.get("safety", {}).get("forbidden_claims") or []
    text = answer or ""

    hits = [c for c in claims if c and c in text]
    if hits:
        return SafetyVerdict(
            allowed=False,
            action="block",
            reasons=[f"命中禁用断言：{'、'.join(hits)}"],
            matched=hits,
            message="该回答涉及超出安全边界的表述，已被拦截。请以专业医疗意见为准。",
        )
    return SafetyVerdict(allowed=True)


# ── 计划侧：变动幅度 ────────────────────────────────────────────
def check_plan_delta(prev_plan: list[str], new_plan: list[str],
                     cfg: dict | None = None) -> SafetyVerdict:
    """检查计划重排的变动幅度是否超过 `safety.max_plan_delta_per_day`。

    只比较**同一长度**的两份计划（按位置比对）。长度不一致视为结构性变更，
    需要人工确认 —— 这一层不做"聪明的对齐"。
    """
    cfg = cfg or domain_mod.load()
    limit = cfg.get("safety", {}).get("max_plan_delta_per_day")

    if limit is None:
        return SafetyVerdict(allowed=True, reasons=["未配置变动上限，跳过检查"])

    if len(prev_plan) != len(new_plan):
        return SafetyVerdict(
            allowed=False,
            action="block",
            reasons=[f"计划天数变化（{len(prev_plan)} → {len(new_plan)}），需人工确认"],
            message="计划结构发生变化，需人工确认后再执行。",
        )

    changed = [f"第{i + 1}项：{a} → {b}"
               for i, (a, b) in enumerate(zip(prev_plan, new_plan)) if a != b]

    if len(changed) > int(limit):
        return SafetyVerdict(
            allowed=False,
            action="block",
            reasons=[f"变动 {len(changed)} 处，超过上限 {limit} 处"],
            matched=changed,
            message=f"本次变动 {len(changed)} 处，超过单日上限 {limit} 处，已拒绝执行。",
        )
    return SafetyVerdict(allowed=True, reasons=[f"变动 {len(changed)} 处，未超上限 {limit}"],
                         matched=changed)


# ── 组合：把三道闸串起来 ────────────────────────────────────────
def apply_all(question: str, answer: str | None = None,
              cfg: dict | None = None) -> dict:
    """一次完整的安全检查（供 `pipeline` 调用）。

    返回：
        {"input": SafetyVerdict, "output": SafetyVerdict | None, "final_answer": str}
    `final_answer` 已按闸门结果处理完毕，调用方可直接输出。
    """
    cfg = cfg or domain_mod.load()

    vin = check_input(question, cfg)
    out = {"input": vin, "output": None}

    if answer is None:
        out["final_answer"] = vin.message if not vin.allowed else ""
        return out

    vout = check_output(answer, cfg)
    out["output"] = vout

    if not vout.allowed:
        final = vout.message
    elif not vin.allowed:
        final = f"{vin.message}\n\n{answer}"
    else:
        final = answer

    out["final_answer"] = final
    return out


if __name__ == "__main__":
    # 自检：直接用 health 领域跑几组典型样例
    import json

    cfg = domain_mod.load("health")
    print("=" * 78)
    print("安全检查自检（health 领域）")
    print("=" * 78)

    cases = [
        ("胸痛该怎么办", "建议先休息，观察症状。"),
        ("今天吃什么好", "可以治愈你的疾病。"),
        ("今天吃什么好", "建议多吃蔬菜，少吃油炸食品。"),
    ]
    for q, a in cases:
        r = apply_all(q, a, cfg)
        print(f"\n问题：{q}")
        print(f"回答：{a}")
        print(f"  输入闸：{'通过' if r['input'].allowed else r['input'].action}"
              f"{'｜' + str(r['input'].matched) if r['input'].matched else ''}")
        print(f"  输出闸：{'通过' if r['output'].allowed else r['output'].action}"
              f"{'｜' + str(r['output'].matched) if r['output'].matched else ''}")
        print(f"  最终输出：{r['final_answer'][:60]}")

    print("\n" + "=" * 78)
    print("计划变动闸（上限 2 处）")
    print("=" * 78)
    prev = ["早餐燕麦", "午餐鸡胸", "晚餐沙拉"]
    same = ["早餐燕麦", "午餐鸡胸", "晚餐沙拉"]
    ok = ["早餐燕麦", "午餐鱼", "晚餐沙拉"]
    bad = ["早餐牛肉", "午餐鱼", "晚餐火锅"]
    for label, new in [("未变", same), ("变 1 处", ok), ("变 3 处", bad)]:
        v = check_plan_delta(prev, new, cfg)
        print(f"  {label}：{'通过' if v.allowed else '拦截'}｜{v.reasons}")
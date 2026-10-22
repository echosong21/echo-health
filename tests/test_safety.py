"""安全拦截层的单元测试。

**这些断言的价值不在于"跑通过"，而在于把行为钉死**：
安全规则一旦悄悄失效，系统看起来一切正常 —— 只是开始给出不该给的回答。
所以每一条规则都配了正反两个用例（该拦的拦住、不该拦的别拦），
以及两个自检时真实踩到过的 bug 的回归用例。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import domain as domain_mod  # noqa: E402
from src.safety import rules  # noqa: E402


@pytest.fixture(scope="module")
def health():
    return domain_mod.load("health")


@pytest.fixture(scope="module")
def law():
    return domain_mod.load("law")


# ── 输入闸：转介 ────────────────────────────────────────────────
def test_refer_triggered_by_emergency_signal(health):
    """出现急症信号时必须触发转介。"""
    v = rules.check_input("我胸口疼、呼吸困难，还能按计划运动吗？", health)
    assert not v.allowed
    assert v.action == "refer"
    assert v.message  # 必须带出转介文案


def test_refer_not_triggered_for_normal_question(health):
    """普通问题不应被误拦。"""
    v = rules.check_input("今天晚饭吃什么比较好？", health)
    assert v.allowed
    assert v.action == "pass"


def test_refer_strips_leading_verb(health):
    """回归用例：`when` 里的引导词（"提到胸痛"）必须被剥掉，否则永远匹配不上。

    这是自检时真实踩到的 bug —— "胸痛该怎么办" 当时被判为"通过"。
    """
    v = rules.check_input("胸痛该怎么办", health)
    assert not v.allowed, "引导词未剥离 ⇒ 真实信号词匹配失败"
    assert v.matched


def test_law_domain_has_no_refer_rules(law):
    """law 领域的 must_refer 为空 ⇒ 不应拦截任何输入（保持改造前行为）。"""
    for q in ["胸痛该怎么办", "呼吸困难", "意识改变"]:
        assert rules.check_input(q, law).allowed


# ── 输出闸：禁用断言 ─────────────────────────────────────────────
def test_block_forbidden_claim(health):
    """出现禁用断言必须整条拦截。"""
    v = rules.check_output("这个方案可以治愈你的疾病。", health)
    assert not v.allowed
    assert v.action == "block"
    assert "可以治愈" in v.matched


def test_allow_normal_answer(health):
    """正常建议不应被误拦。"""
    v = rules.check_output("建议多吃蔬菜，少吃油炸食品，保证睡眠。", health)
    assert v.allowed


def test_block_message_is_safe(health):
    """拦截后的回复本身必须是安全的（不能又带出违规表述）。"""
    v = rules.check_output("无需就医，多喝水就行。", health)
    assert not v.allowed
    assert "无需就医" in v.matched
    # 拦截文案不能再包含任何禁用断言
    assert all(c not in v.message for c in health["safety"]["forbidden_claims"])


def test_law_domain_blocks_nothing(law):
    """law 领域禁止断言为空 ⇒ 任何回答都放行（与改造前一致）。"""
    assert rules.check_output("可以治愈你的疾病。", law).allowed


# ── 计划闸：变动幅度 ────────────────────────────────────────────
def test_plan_delta_within_limit(health):
    prev = ["早餐燕麦", "午餐鸡胸", "晚餐沙拉"]
    new = ["早餐燕麦", "午餐鱼", "晚餐沙拉"]
    v = rules.check_plan_delta(prev, new, health)
    assert v.allowed
    assert len(v.matched) == 1


def test_plan_delta_exceeds_limit(health):
    prev = ["早餐燕麦", "午餐鸡胸", "晚餐沙拉"]
    new = ["早餐牛肉", "午餐鱼", "晚餐火锅"]
    v = rules.check_plan_delta(prev, new, health)
    assert not v.allowed
    assert v.action == "block"
    assert len(v.matched) == 3


def test_plan_delta_identical_plan_passes(health):
    plan = ["早餐燕麦", "午餐鸡胸", "晚餐沙拉"]
    assert rules.check_plan_delta(plan, list(plan), health).allowed


def test_plan_delta_length_change_requires_review(health):
    """天数变化属结构性变更，不应被静默接受。"""
    v = rules.check_plan_delta(["a", "b"], ["a", "b", "c"], health)
    assert not v.allowed


def test_plan_delta_skipped_when_unset(law):
    """未配置上限（law）⇒ 跳过检查而不是报错。"""
    v = rules.check_plan_delta(["a"], ["b"], law)
    assert v.allowed


# ── 组合：apply_all ────────────────────────────────────────────
def test_apply_all_refer_prepends_message(health):
    """转介语必须出现在最终输出的**最前面**。"""
    r = rules.apply_all("胸痛怎么办", "建议休息观察。", health)
    assert r["final_answer"].startswith(r["input"].message)


def test_apply_all_block_replaces_answer(health):
    """命中禁用断言时，最终输出被整条替换，原回答不泄露。"""
    r = rules.apply_all("吃什么好", "这个可以治愈你的病。", health)
    assert "可以治愈" not in r["final_answer"]
    assert r["final_answer"] == r["output"].message


def test_apply_all_clean_case_passthrough(health):
    """两侧都通过时，原回答原样返回。"""
    answer = "建议多吃蔬菜。"
    r = rules.apply_all("今天吃什么好", answer, health)
    assert r["final_answer"] == answer


def test_apply_all_question_only(health):
    """只给问题（未生成回答）时，也应能做输入侧判定。"""
    r = rules.apply_all("胸痛怎么办", None, health)
    assert r["output"] is None
    assert r["final_answer"] == r["input"].message


# ── 配置驱动：同一份代码，不同领域不同行为 ──────────────────────────
def test_same_code_different_domain_behavior(health, law):
    """真正的"配置驱动"验证：**同一份代码**在两个领域下行为不同。

    （不用"规则词是否出现在源码里"来判断 —— 那种字符串匹配会被注释、
     自检样例等干扰，证明力也弱。行为差异才是配置生效的确切证据。）
    """
    answer = "这个方案可以治愈你的疾病。"

    # health 配置了禁用断言 ⇒ 拦截
    assert rules.check_output(answer, health).allowed is False
    # law 没有配置 ⇒ 放行
    assert rules.check_output(answer, law).allowed is True

    # 急症转介同理
    assert rules.check_input("胸痛", health).allowed is False
    assert rules.check_input("胸痛", law).allowed is True
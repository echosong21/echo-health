"""领域配置加载：把「换领域 = 换一个 yaml」变成可执行的契约。

**这一层要解决的唯一问题**

    换领域时，哪些东西必须跟着变？答案集中在 `domains/*.yaml` 里：
    形态名、提示词、免责声明、语料参数、集合名、安全规则。
    而**模型、端点、密钥、向量维度**属于部署环境，留在 `.env`，不进 yaml。

    ⇒ 因此能力层（本目录下的检索/生成代码）**不需要知道"法律"或"健康"**，
       它只读 `load()` 返回的结构。这就是"能力层与领域层解耦"的落地方式。

**设计约束**

    ① **fail fast**：缺字段/字段非法立刻抛 `DomainConfigError`，不猜默认值蒙混过去。
    ② **向后兼容**：`DOMAIN` 未设时默认加载 `law`，与 `rag-service` 的现状一一对应。
    ③ **只做加载与校验**，不含业务逻辑 —— 安全拦截属 `safety/rules.py`（P2）。

用法：
    python -c "from src.domain import load; d = load('health'); print(d['collection'])"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOMAINS_DIR = ROOT / "domains"

# 领域未指定时的默认值 —— 与 rag-service 现状保持一致，保证行为不回归
DEFAULT_KEY = os.getenv("DOMAIN", "law")

# 允许的免责声明注入位置
VALID_POSITIONS = ("prepend", "append")


class DomainConfigError(Exception):
    """领域配置缺失或非法。**刻意不做静默回退** —— 配错了就该立刻暴露。"""


def list_domains() -> list[str]:
    """列出可用领域（`domains/` 下的 yaml 文件名，按字母序）。"""
    if not DOMAINS_DIR.is_dir():
        return []
    return sorted(p.stem for p in DOMAINS_DIR.glob("*.yaml"))


def load(key: str | None = None) -> dict:
    """加载并校验领域配置。

    `key` 省略时取环境变量 `DOMAIN`（默认 `law`）。
    返回已完成默认值填充的 dict；任何缺失或非法都抛 `DomainConfigError`。
    """
    key = key or DEFAULT_KEY
    path = DOMAINS_DIR / f"{key}.yaml"
    if not path.exists():
        available = ", ".join(list_domains()) or "（domains/ 下没有任何 yaml）"
        raise DomainConfigError(f"找不到领域配置：{path}；可用领域：{available}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _fill_defaults(raw)
    _validate(cfg, path)
    cfg["_source"] = str(path)          # 便于日志与排障时回答"用的是哪个配置"
    return cfg


# ── 默认值填充 ────────────────────────────────────────────────
def _fill_defaults(cfg: dict) -> dict:
    """只填**无歧义**的默认值。凡是"填错会悄悄改变行为"的，一律不填、交给校验。"""
    cfg.setdefault("forms", [])
    cfg.setdefault("disclaimers", [])
    cfg.setdefault("safety", {})
    cfg.setdefault("eval", {})

    corpus = cfg.setdefault("corpus", {})
    corpus.setdefault("root", "./corpus")
    corpus.setdefault("chunk_size", 600)        # 与能力层既有默认一致
    corpus.setdefault("chunk_overlap", 80)
    corpus.setdefault("per_form_limit", {})

    prompt = cfg.setdefault("prompt", {})
    prompt.setdefault("no_answer", "材料中未提及")

    safety = cfg["safety"]
    safety.setdefault("forbidden_claims", [])
    safety.setdefault("must_refer", [])
    safety.setdefault("max_plan_delta_per_day", None)
    safety.setdefault("scope_note", "")

    return cfg


# ── 校验（对应契约里的 6 条 + 若干结构检查）──────────────────────
def _validate(cfg: dict, src: Path) -> None:
    def fail(msg: str):
        raise DomainConfigError(f"[{src.name}] {msg}")

    domain = cfg.get("domain") or {}
    if not domain.get("key"):
        fail("缺少 domain.key")
    if not domain.get("language"):
        fail("缺少 domain.language（分词与提示词都依赖它）")
    if domain["key"] != src.stem:
        fail(f"domain.key（{domain['key']}）与文件名（{src.stem}）不一致")

    forms = cfg["forms"]
    if not forms:
        fail("forms 不能为空（它决定多路召回的'路'）")
    for i, form in enumerate(forms):
        if not form.get("key"):
            fail(f"forms[{i}] 缺少 key")
        # 形态 key 会作为语料子目录名与 source 前缀，必须是安全标识符
        if not form["key"].replace("_", "").isalnum():
            fail(f"forms[{i}].key（{form['key']}）必须是字母/数字/下划线"
                 "（它会用作目录名与 source 前缀）")

    if not cfg.get("collection"):
        fail("缺少 collection（换领域必换集合）")

    prompt = cfg["prompt"]
    if not prompt.get("system"):
        fail("缺少 prompt.system")
    if "{no_answer}" not in prompt["system"]:
        fail("prompt.system 必须包含 {no_answer} 占位符")
    if not prompt.get("no_answer"):
        fail("缺少 prompt.no_answer")

    for i, d in enumerate(cfg["disclaimers"]):
        if not d.get("text"):
            fail(f"disclaimers[{i}] 缺少 text")
        if d.get("position") not in VALID_POSITIONS:
            fail(f"disclaimers[{i}].position 必须是 {VALID_POSITIONS} 之一，"
                 f"实际为 {d.get('position')!r}")

    safety = cfg["safety"]
    if not isinstance(safety.get("forbidden_claims"), list):
        fail("safety.forbidden_claims 必须是列表（可为空列表）")

    corpus = cfg["corpus"]
    if not isinstance(corpus.get("per_form_limit"), dict):
        fail("corpus.per_form_limit 必须是 dict（形态 → 文件数上限）")


# ── 给能力层用的访问器（避免各处直接摸 dict 键名）────────────────
def form_keys(cfg: dict) -> list[str]:
    """形态 key 列表，顺序即多路召回的形态顺序。"""
    return [f["key"] for f in cfg["forms"]]


def system_prompt(cfg: dict) -> str:
    """把 `{no_answer}` 占位符替换为兜底话术后的系统提示。"""
    return cfg["prompt"]["system"].format(no_answer=cfg["prompt"]["no_answer"])


def disclaimer_texts(cfg: dict, position: str) -> list[str]:
    """取指定位置的免责声明文本（`prepend` / `append`）。"""
    return [d["text"] for d in cfg["disclaimers"] if d.get("position") == position]


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else None
    c = load(key)
    print(f"领域      : {c['domain']['key']} ({c['domain'].get('display_name', '')})")
    print(f"来源      : {c['_source']}")
    print(f"形态      : {form_keys(c)}")
    print(f"集合      : {c['collection']}")
    print(f"免责声明  : prepend={len(disclaimer_texts(c, 'prepend'))} "
          f"append={len(disclaimer_texts(c, 'append'))}")
    print(f"安全规则  : forbidden={len(c['safety']['forbidden_claims'])} "
          f"must_refer={len(c['safety']['must_refer'])}")
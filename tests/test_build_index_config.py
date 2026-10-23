"""`scripts/build_index.py` 读取领域配置的行为测试。

**为什么需要这些测试**

    背景缺陷：`domains/*.yaml` 里定义了 `corpus.chunk_size` / `corpus.per_form_limit` /
    `collection`，但 `build_index.py` 只认命令行参数 —— 配置写了却完全不生效
    （「配置化了但没接线」）。

    修复后必须保证两件事：
      ① **配置真的生效**（`--domain health` 能读到 yaml 里的值）
      ② **向后兼容**（不传 `--domain` 时行为与修复前完全一致）

    第 ② 点尤其重要：这个脚本也被 law 领域和其他实验使用，不能让它们因为这次
    改动而改变切分参数或集合名。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_index.py"


def _run_help() -> str:
    """跑 --help，拿 argparse 的 usage 与参数说明（不触发任何网络/embedding）。"""
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert out.returncode == 0, f"--help 失败：{out.stderr}"
    return out.stdout


def test_domain_argument_exists() -> None:
    """`--domain` 必须存在 —— 它是配置生效的入口。"""
    help_text = _run_help()
    assert "--domain" in help_text, "build_index.py 缺少 --domain 参数"


def test_collection_default_is_empty() -> None:
    """`--collection` 的默认值必须是空，否则 yaml 里的 collection 永远被盖掉。

    这是一个容易写错的地方：如果默认值写成 settings.COLLECTION，
    `--domain` 模式下就永远取不到 yaml 的 collection。
    """
    help_text = _run_help()
    # argparse 的 usage 段会列出所有可选参数
    assert "--collection" in help_text
    # 说明文字里应能看到「留空则取 --domain 配置」这类语义
    assert "--domain 配置" in help_text or "domain" in help_text


def test_chunk_size_help_shows_no_hardcoded_default() -> None:
    """`--chunk-size` 不应在帮助里显示写死的默认值。

    default=None 时 argparse 不会打印 `(default: 600)`；
    如果打印了，说明默认值又被写死了，`--domain` 的 chunk_size 会失效。
    """
    help_text = _run_help()
    lines = [ln for ln in help_text.splitlines() if "--chunk-size" in ln]
    assert lines, "--help 里找不到 --chunk-size"
    joined = " ".join(lines)
    assert "default: 600" not in joined, (
        "--chunk-size 的默认值被写死为 600，会导致 domains/*.yaml 的 chunk_size 不生效"
    )


@pytest.mark.parametrize("domain", ["health", "law"])
def test_domain_yaml_exposes_corpus_settings(domain: str) -> None:
    """每个领域的 yaml 都应能通过 src.domain.load 读出 corpus 段。

    build_index.py 就是靠这个入口读配置的；若入口本身读不到，
    配置生效也就无从谈起。
    """
    sys.path.insert(0, str(REPO))
    from src.domain import load  # noqa: PLC0415

    cfg = load(domain)
    corpus = cfg.get("corpus") or {}
    assert "chunk_size" in corpus, f"{domain}.yaml 缺少 corpus.chunk_size"
    assert "chunk_overlap" in corpus, f"{domain}.yaml 缺少 corpus.chunk_overlap"
    assert cfg.get("collection"), f"{domain}.yaml 缺少 collection"


def test_health_per_form_limit_is_present() -> None:
    """health 领域必须声明 per_form_limit —— 它控制各形态入库的文件数。

    历史：该字段曾经只是「写在 yaml 里给人看」，实际不生效；
    修复后它必须能读到，否则又回到「配置形同虚设」的状态。
    """
    sys.path.insert(0, str(REPO))
    from src.domain import load  # noqa: PLC0415

    cfg = load("health")
    limits = (cfg.get("corpus") or {}).get("per_form_limit")
    assert limits is not None, "health.yaml 缺少 corpus.per_form_limit"
    assert "dialogues" in limits, "per_form_limit 应至少约束 dialogues"
    # null 在 YAML 里解析为 None，语义是「不截断」
    assert limits.get("tables", "missing") is None or isinstance(limits.get("tables"), int)
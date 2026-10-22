"""模块导入的回归守卫。

**这条测试要防的是什么**

    `pipeline.py` 曾在**模块级**构造 `OpenAIEmbeddings` / `ChatOpenAI` / `MilvusClient`。
    后果：`import src.pipeline` 会立刻校验配置、构造网络客户端 ——
    在没有 `.env` 的环境里（CI、全新 clone、IDE 索引）直接抛
    `ValidationError: model must be a valid string`，模块**根本无法导入**。

    更麻烦的是这个缺陷**很难被功能测试发现**：只要本地有 `.env`，一切正常，
    问题只会在"干净环境"里暴露 —— 也就是 CI 和别人的机器上。

**为什么用静态检查而不是"导入一下试试"**

    "导入试试"只在**恰好没有 .env** 时才有效；本地有 `.env` 就测不出来。
    而"模块级不得出现客户端构造"是**与环境污染无关**的确定判据：
    只要有人把构造挪回模块级，这条测试在任何机器上都会红。

    两者都保留：静态检查负责**在错误的写法出现时立刻报警**，
    导入检查负责**兜住"能导入"这一最低要求**。
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 这些构造会建立网络客户端 / 校验配置，**不允许出现在模块级**
SIDE_EFFECT_CONSTRUCTORS = (
    "OpenAIEmbeddings(",
    "ChatOpenAI(",
    "MilvusClient(",
    "load_dotenv(",
)


def _module_level_lines(path: Path) -> list[tuple[int, str]]:
    """返回文件中**顶格**（不在函数/类体内）的非空、非注释行。

    判据足够简单：顶格 = 模块级。函数体、类体、if 块内部都会缩进。
    （`if __name__ == "__main__":` 内部同样缩进，因此不会被误判。）
    """
    out = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw[0] not in (" ", "\t"):
            out.append((i, raw.rstrip()))
    return out


@pytest.mark.parametrize("path", [
    ROOT / "src" / "pipeline.py",
    ROOT / "src" / "retrieval.py",
    ROOT / "src" / "fusion.py",
    ROOT / "src" / "rerank.py",
    ROOT / "src" / "domain.py",
])
def test_no_module_level_side_effects(path):
    """模块级不得构造客户端 —— 否则 `import` 就会依赖运行时配置。"""
    offenders = [
        f"L{lineno}: {line.strip()}"
        for lineno, line in _module_level_lines(path)
        for ctor in SIDE_EFFECT_CONSTRUCTORS
        if ctor in line
    ]
    assert not offenders, (
        f"{path.name} 在模块级出现了副作用构造（会破坏无配置环境下的导入）：\n  "
        + "\n  ".join(offenders)
        + "\n⇒ 请改为在函数内部按需创建（参考 pipeline.get_emb/get_llm/get_client）。"
    )


@pytest.mark.parametrize("module", [
    "src.domain",
    "src.retrieval",
    "src.fusion",
    "src.bm25",
    "src.rerank",
    "src.pipeline",
    "src.safety.rules",
])
def test_module_is_importable(module):
    """最低要求：模块能被导入且不抛异常。"""
    importlib.invalidate_caches()
    assert importlib.import_module(module) is not None


def test_pipeline_exposes_lazy_getters():
    """`pipeline` 必须提供懒加载入口（而不是模块级对象）。

    这条是"接口约定"测试：即使有人删掉 get_* 又改回模块级对象，
    静态检查会红；如果他两个都留着，这条会提醒调用方该用哪个。
    """
    from src import pipeline

    for name in ("get_emb", "get_llm", "get_client"):
        fn = getattr(pipeline, name, None)
        assert callable(fn), f"pipeline.{name} 应存在且可调用"
        # 调用前不应已创建实例；这里只断言是函数，不实际调用（会连网）
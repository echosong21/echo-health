import os,sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1]/ ".env")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MILVUS_URL = os.getenv("MILVUS_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL")

# ── 可选：重排层 ──────────────────────────────────────────────
# 重排用的是 DashScope 原生接口（不是 OpenAI 兼容端点），因此需要业务空间 ID 与地域。
# 这几个键**不加入 assert_env() 的强制校验** —— 重排是可降级功能，缺配置时应静默降级
# 为「不重排」，而不是让整个服务起不来。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or API_KEY
WORKSPACE_ID = os.getenv("WORKSPACE_ID")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")
RERANK_REGION = os.getenv("RERANK_REGION", "cn-beijing")
# ⚠️ 默认关闭：实测重排接口有约 3.6s 的**固定**延迟（与候选条数、模型、冷启动均无关），
# 开启会让端到端延迟翻倍。因此默认不启用，需要时显式打开并接受该代价。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").strip().lower() in (
    "1", "true", "yes", "on"
)



def assert_env():

    assert API_KEY,"缺少API_KEY"
    assert MILVUS_URL, "缺少MILVUS_URL"
    assert EMBED_MODEL, "缺少EMBED_MODEL"
    assert EMBED_BASE_URL, "缺少EMBED_BASE_URL"
    assert MODEL, "缺少 MODEL"
    assert BASE_URL, "缺少 BASE_URL"


if __name__ == '__main__':
    assert_env()
    print("配置就绪 ✅")
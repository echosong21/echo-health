"""HTTP 包装层 —— 统一入口 v0：把三个能力串到一个服务上。

**为什么状态码要分开语义**

`422` 是 FastAPI 对**请求体不合 schema** 的自动响应（字段缺失、类型不对）；
但「`question` 传了空字符串」是**通过了 schema 但语义无效** ——
这种情况用 `400` 更准，否则调用方无法区分「我发错了格式」和「我发的内容没意义」。

| 码 | 何时 |
|---|---|
| 200 | 正常返回（含 `degraded` —— 见下） |
| 400 | 语义无效（如 question 全空白） |
| 422 | 请求体不合 schema（FastAPI 自动） |
| 500 | 内部错误（不该发生的 bug） |
| 503 | **必需品依赖不可用**（Milvus 探不到） |

**`/health` 为什么要分级而不是一刀切 503**

本系统有两类依赖，故障后果完全不同：

| 依赖 | 性质 | 挂了之后 | 状态码 |
|---|---|---|---|
| **Milvus** | **必需品** —— 没有检索就没有答案 | 服务**真的不能工作** | **503** |
| **PostgreSQL**（记忆层） | **可降级** —— 有熔断，照常答 | 服务**仍能工作**，只是少了上下文 | **200 + degraded** |

**判断规则一句话：整体状态码由「必需品」决定；可降级依赖只影响 body 不影响 200。**

如果 PG 挂了就返 503，等于对外宣称「我挂了」，把记忆层那套熔断降级的能力
主动作废 —— 那是自相矛盾的。反过来，Milvus 挂了必须 503，
否则负载均衡会把流量继续送到一个答不出话的实例上。

**为什么 `/health` 要返回 collection 与 num_entities**

监测面板与负载均衡都要靠它判断「这个实例背后有没有数据」。
只回一个 `{"status":"ok"}` 无法区分「服务活着但索引空了」。

**为什么 `/tools` 用静态镜像而不是跨仓读文件**

工具注册表在另一个仓库。运行时读它的文件意味着部署时必须把两个仓库
放在同一台机器的固定相对路径上 —— 生产环境不可能（两个服务可能在不同容器）。

所以：**运行时读本模块内的静态镜像，测试里断言镜像与真实注册表一致**。
跨仓读只发生在测试，不发生在部署。

⚠️ 本模块**不做**缓存、限流、鉴权、多租户 —— 那些在反向代理或 MCP 层解决，
放在这里会让「检索服务」变成一个什么都管的巨型组件。
"""
from __future__ import annotations

import socket
from contextlib import closing

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(3, ge=1, le=20, description="返回条数")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="相似度阈值")


class MultiQueryRequest(QueryRequest):
    per_form: int = Field(2, ge=1, le=10, description="每个形态各取几条")


app = FastAPI(
    # ⚠️ title / description 会出现在 /docs，必须中性、不含内部用语
    title="Domain RAG Retrieval Service",
    description="多形态向量检索与问答接口",
    version="0.2.0",
)


# ── 工具镜像（来源：MCP 工具注册表）────────────────────────────
#
# ⚠️ 这是**静态镜像**，不是运行时读取。原因见模块 docstring。
# tests/test_api.py::test_tools_mirror_matches_registry 会断言它与注册表一致 ——
# 注册表加了工具而这里没同步，测试会红。
TOOLS = [
    {
        "name": "common.discovery.list_domains",
        "scope": "common.discovery",
        "side_effect": "read",
        "brief": "列出可用的知识领域，含每种的检索形态与集合名",
    },
    {
        "name": "common.discovery.health_check",
        "scope": "common.discovery",
        "side_effect": "read",
        "brief": "检查向量库 / embedding / 重排服务与熔断器的当前状态",
    },
    {
        "name": "common.knowledge.search",
        "scope": "common.knowledge",
        "side_effect": "read",
        "brief": "按语义在指定领域内检索，返回片段列表（不生成回答）",
    },
    {
        "name": "common.knowledge.search_multi",
        "scope": "common.knowledge",
        "side_effect": "read",
        "brief": "按数据形态并发召回多路再 RRF 融合，返回统一排序的片段列表",
    },
    {
        "name": "common.knowledge.answer",
        "scope": "common.knowledge",
        "side_effect": "read",
        "brief": "检索 + 生成 + 安全闸一体的回答；命中禁用断言则整条拒答",
    },
    {
        "name": "common.knowledge.answer_planned",
        "scope": "common.knowledge",
        "side_effect": "read",
        "brief": "先判断是否多焦点，是则拆子任务分别检索再融合生成",
    },
    {
        "name": "common.knowledge.explain",
        "scope": "common.knowledge",
        "side_effect": "read",
        "brief": "解释一次检索的决策：拆了哪些子任务、每路召回什么、融合后谁被顶上来",
    },
    {
        "name": "common.ops.get_index_status",
        "scope": "common.ops.read",
        "side_effect": "read",
        "brief": "查询某集合的行数、索引类型与状态",
    },
    {
        "name": "common.ops.reindex",
        "scope": "common.ops.write",
        "side_effect": "write",
        "brief": "重建某领域的索引；默认只预演，需显式传 dry_run=false 才真的执行",
    },
    {
        "name": "common.safety.get_rules",
        "scope": "common.safety",
        "side_effect": "read",
        "brief": "返回该领域的禁止断言、免责声明、计划变动上限与急症信号",
    },
    {
        "name": "common.safety.check_input",
        "scope": "common.safety",
        "side_effect": "read",
        "brief": "检查用户输入是否需转介急症，或触及禁止断言",
    },
    {
        "name": "common.safety.check_output",
        "scope": "common.safety",
        "side_effect": "read",
        "brief": "检查生成的回答是否含禁止断言；命中则整条拒答（不改写）",
    },
]


def _classify_failure(exc: Exception) -> int:
    """把失败分成「依赖不可用」还是「内部错误」。

    **为什么不能只靠异常类型判断**

    `MilvusException` / `PgError` 这些库的异常**不继承** `ConnectionError` ——
    它们有自己的层级。按类型白名单永远会漏（每换一个依赖就要加一个类型）。

    **改用探针**：失败时问一句「现在依赖还活着吗？」
      · 依赖已 down ⇒ 503（这个失败是它造成的）
      · 依赖还 up  ⇒ 500（那是我自己的 bug）

    ⚠️ 代价：失败路径多一次探测。但失败路径本来就不该追求快，
    而且探针是端口/统计调用，量级是毫秒。

    ⚠️ 边界：依赖「能连但答错」的情况会判成 500 —— 这是刻意的。
    那说明依赖活着，问题在我们的使用方式（或数据），属于内部错误。
    """
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return 503                     # 连接层故障，直接归因依赖
    milvus, _ = _probe_milvus()
    return 503 if milvus != "up" else 500


def _require_question(question: str) -> str:
    """语义校验：schema 通过不代表内容有意义。"""
    text = (question or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="question 不能为空白")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="question 超过 2000 字上限")
    return text


def _probe_milvus() -> tuple[str, str | None]:
    """探 Milvus。返回 (状态, 集合名)。

    ⚠️ 连通但集合为空也算 up —— 那表示「能工作但答不了」，不是 down。
    """
    from src import pipeline as pl

    try:
        collection = pl.get_collection()
        pl.get_client().get_collection_stats(collection_name=collection)
        return "up", collection
    except Exception:                                            # noqa: BLE001
        return "down", None


def _probe_pg(timeout: float = 1.0) -> str:
    """探 PostgreSQL（记忆层）。

    **为什么用 socket 而不是 `store.read()`**

    记忆层刻意做成「失败不抛异常、返回空结构」——那是降级设计，不是缺陷。
    所以用它自己的 API 探测**永远看不出挂没挂**（永远返回空、永远"成功"）。
    这里直接探端口，绕开这层语义。

    ⚠️ 只表示「端口开着」，不表示 SQL 能跑。更严的探针要执行 `SELECT 1`，
    但那要连密码与库名 —— 健康检查不该引入这些依赖，端口探测对这个场景够用。
    """
    host, port = "127.0.0.1", 5432
    try:
        from src import settings

        url = getattr(settings, "PG_URL", "") or getattr(settings, "DATABASE_URL", "")
        if "@" in url:
            netloc = url.split("@", 1)[1].split("/", 1)[0]
            if ":" in netloc:
                host, port_s = netloc.rsplit(":", 1)
                port = int(port_s)
            elif netloc:
                host = netloc
    except Exception:                                            # noqa: BLE001
        pass                                                     # 取不到配置就用默认

    try:
        with closing(socket.create_connection((host, port), timeout=timeout)):
            return "up"
    except Exception:                                            # noqa: BLE001
        return "down"


@app.get("/health")
def health() -> JSONResponse:
    """存活 + 依赖分级探针。

    状态码由**必需品**（Milvus）决定：
      · Milvus down         → 503 unavailable
      · Milvus up, PG down  → 200 degraded（服务仍能答，只是没记忆）
      · 全 up               → 200 ok
    """
    milvus, collection = _probe_milvus()
    pg = _probe_pg()
    deps = {"milvus": milvus, "postgres": pg}

    if milvus != "up":
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "dependencies": deps,
                "detail": "检索后端不可用，服务无法回答问题",
            },
        )

    body: dict = {"status": "ok", "dependencies": deps, "collection": collection}
    if pg != "up":
        body["status"] = "degraded"
        body["detail"] = "记忆层不可用，问答不受影响（已熔断降级）"

    # 附带行数，便于区分「活着但索引空了」
    try:
        from src import pipeline as pl

        stats = pl.get_client().get_collection_stats(collection_name=collection)
        body["num_entities"] = int(stats.get("row_count") or 0)
    except Exception:                                            # noqa: BLE001
        body["num_entities"] = None

    return JSONResponse(status_code=200, content=body)


@app.get("/tools")
def tools() -> dict:
    """暴露工具清单（静态镜像，来源见模块 docstring）。"""
    return {
        "count": len(TOOLS),
        "tools": TOOLS,
        "note": "静态镜像，与 MCP 工具注册表由测试保证一致",
    }


@app.post("/query")
def query(req: QueryRequest) -> dict:
    """规划模式：多焦点问题自动拆子任务检索。

    **与 `/query_multi` 的分工**

        /query       按【语义子问题】拆（高血压 / 糖尿病该注意什么）
        /query_multi 按【数据形态】拆（guides / faq / tables / dialogues）

    两个维度正交，都不做对方的事。
    """
    from src import pipeline as pl

    question = _require_question(req.question)
    try:
        out = pl.rag_query_planned(question, top_k=req.top_k, min_score=req.min_score)
    except HTTPException:
        raise
    except Exception as exc:                                    # noqa: BLE001
        code = _classify_failure(exc)
        detail = (
            f"依赖不可用：{type(exc).__name__}" if code == 503
            else f"检索失败：{type(exc).__name__}"
        )
        raise HTTPException(status_code=code, detail=detail) from exc

    return {
        "answer": out["answer"],
        "contexts": out["contexts"],
        "stage_ms": out["stage_ms"],
        "usage": out["usage"],
        "safety": out["safety"],
        "mode": out.get("mode", "planned"),
        "subtasks": out.get("subtasks", [question]),
        "per_task_n": out.get("per_task_n", {}),
    }


@app.post("/query_multi")
def query_multi(req: MultiQueryRequest, hybrid: bool = False) -> dict:
    """多形态并发召回 + 生成。"""
    from src import pipeline as pl

    question = _require_question(req.question)
    try:
        out = pl.rag_query_multi(
            question, per_form=req.per_form,
            min_score=req.min_score, top_k=req.top_k, hybrid=hybrid,
        )
    except HTTPException:
        raise
    except Exception as exc:                                    # noqa: BLE001
        code = _classify_failure(exc)
        detail = (
            f"依赖不可用：{type(exc).__name__}" if code == 503
            else f"检索失败：{type(exc).__name__}"
        )
        raise HTTPException(status_code=code, detail=detail) from exc

    return {
        "answer": out["answer"],
        "contexts": out["contexts"],
        "stage_ms": out["stage_ms"],
        "usage": out["usage"],
        "safety": out["safety"],
        "per_form": out["per_form"],
    }
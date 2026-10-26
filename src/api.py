"""HTTP 包装层 —— 把检索能力暴露为 3 个端点。

**为什么状态码要分开语义**

`422` 是 FastAPI 对**请求体不合 schema** 的自动响应（字段缺失、类型不对）；
但「`question` 传了空字符串」是**通过了 schema 但语义无效** ——
这种情况用 `400` 更准，否则调用方无法区分「我发错了格式」和「我发的内容没意义」。

| 码 | 何时 |
|---|---|
| 200 | 正常返回 |
| 400 | 语义无效（如 question 全空白） |
| 422 | 请求体不合 schema（FastAPI 自动） |
| 500 | 内部错误（检索/生成抛异常） |
| 503 | 依赖不可用（`/health` 探不到 Milvus） |

**为什么 `/health` 要返回 collection 与 num_entities**

监测面板与负载均衡都要靠它判断「这个实例背后有没有数据」。
只回一个 `{"status":"ok"}` 无法区分「服务活着但索引空了」。

⚠️ 本模块**不做**缓存、限流、鉴权 —— 那些在反向代理或 MCP 层解决，
放在这里会让「检索服务」变成一个什么都管的巨型组件。
"""
from __future__ import annotations

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
    version="0.1.0",
)


def _require_question(question: str) -> str:
    """语义校验：schema 通过不代表内容有意义。"""
    text = (question or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="question 不能为空白")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="question 超过 2000 字上限")
    return text


@app.get("/health")
def health() -> JSONResponse:
    """存活 + 数据可用性。"""
    from src import pipeline as pl

    try:
        collection = pl.get_collection()
        stats = pl.get_client().get_collection_stats(collection_name=collection)
        num = int(stats.get("row_count") or 0)
    except Exception as exc:                                    # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": type(exc).__name__},
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "collection": collection,
            "num_entities": num,
        },
    )


@app.post("/query")
def query(req: QueryRequest) -> dict:
    """单路检索 + 生成。"""
    from src import pipeline as pl

    question = _require_question(req.question)
    try:
        out = pl.rag_query(question, top_k=req.top_k, min_score=req.min_score)
    except HTTPException:
        raise
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"检索失败：{type(exc).__name__}") from exc

    return {
        "answer": out["answer"],
        "contexts": out["contexts"],
        "stage_ms": out["stage_ms"],
        "usage": out["usage"],
        "safety": out["safety"],
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
        raise HTTPException(status_code=500, detail=f"检索失败：{type(exc).__name__}") from exc

    return {
        "answer": out["answer"],
        "contexts": out["contexts"],
        "stage_ms": out["stage_ms"],
        "usage": out["usage"],
        "safety": out["safety"],
        "per_form": out["per_form"],
    }
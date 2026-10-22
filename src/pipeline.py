import argparse
import sys,time
from datetime import datetime
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from pymilvus import MilvusClient
from config import settings
from src import retrieval as rt
from src import domain as domain_mod
from src import observe
from src.safety import rules as safety_rules

try:
    _DOMAIN = domain_mod.load()
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(f"领域配置加载失败（检查 domains/*.yaml）：{exc}") from exc

# ⚠️ 集合名来自领域配置 —— 换领域时本文件无需改动
COLLECTION = _DOMAIN["collection"]

# ── BM25 索引懒加载 ──────────────────────────────────────────────────
# 只有混合检索模式才需要；懒加载避免其他模式白等约 2 秒的索引加载。
BM25_CACHE_KEY = "935426c666faa59d"
_bm25_bundle = None


def _get_bm25_bundle():
    global _bm25_bundle
    if _bm25_bundle is None:
        import pickle

        from src import bm25 as bm
        cache = bm.CACHE_DIR / f"vectors_{BM25_CACHE_KEY}.pkl"
        items = pickle.loads(cache.read_bytes())["items"]
        bm25, docs, _stats, form_index = bm.build_or_load(items, BM25_CACHE_KEY)
        # 预热两处惰性初始化（jieba 词典 + 打分侧），否则第一个请求会慢约 1 秒
        bm.warmup(bm25)
        _bm25_bundle = (bm25, docs, form_index)
    return _bm25_bundle

# ── 懒加载：模块导入时**不**建立任何运行时连接 ──────────────────────
# ⚠️ 原先在模块级直接构造 OpenAIEmbeddings / ChatOpenAI / MilvusClient，
#    导致 `import src.pipeline` 立刻校验配置并建连接 —— 没有 .env 时直接报
#    ValidationError，模块根本无法被导入（也就无法被单测覆盖）。
#    ⇒ 改为首次使用时构造：导入零副作用，缺配置时在调用处报错，位置更准确。
_emb = None
_llm = None
_client = None


def get_emb():
    """OpenAIEmbeddings 单例（首次调用时创建）。"""
    global _emb
    if _emb is None:
        _emb = OpenAIEmbeddings(
            model=settings.EMBED_MODEL,
            base_url=settings.EMBED_BASE_URL,
            api_key=settings.API_KEY,
            check_embedding_ctx_length=False,
        )
    return _emb


def get_llm():
    """ChatOpenAI 单例（首次调用时创建）。"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.MODEL,
            base_url=settings.BASE_URL.removesuffix("/chat/completions"),
            api_key=settings.API_KEY,
        )
    return _llm


def get_client():
    """MilvusClient 单例（首次调用时创建）。"""
    global _client
    if _client is None:
        _client = MilvusClient(uri=settings.MILVUS_URL)
    return _client

# 拼装模板（能力层固定）：只负责把「领域提示词 + 材料 + 问题」拼起来。
# 领域提示词本身来自 domains/*.yaml 的 prompt.system，不写在这里。
PROMPT_TMPL = """{system}

材料：
{contexts}

问题：{question}
"""


def _finalize_answer(answer: str | None) -> str | None:
    """按领域配置注入免责声明（**强制注入**，Q41 硬要求）。

    - `disclaimers` 为空（如 law 领域）⇒ 原样返回，**行为与改造前完全一致**；
    - 非空（如 health 领域）⇒ 按 `position` 注入到答案前后。
    """
    if answer is None:
        return None
    pre = domain_mod.disclaimer_texts(_DOMAIN, "prepend")
    app = domain_mod.disclaimer_texts(_DOMAIN, "append")
    parts = [*pre, answer, *app]
    return "\n\n".join(parts) if (pre or app) else answer
logger = observe.setup_logging()
def rag_query(question: str, top_k: int = 3, min_score: float = 0.0) -> dict:
    """min_score=0.0 ⇒ 不过滤，行为与改造前完全一致"""
    stage_ms, contexts, usage, answer, raw = {}, [], None, None, []
    trace_id = observe.new_trace_id()
    status, error = "ok", None
    try:
        t0 = time.perf_counter()
        qvec = get_emb().embed_query(question)
        stage_ms["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        hits = get_client().search(collection_name=COLLECTION, data=[qvec],
                             limit=top_k,
                             output_fields=["text", "source", "chunk_index"])[0]
        stage_ms["search_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        raw = [{"score": round(h["distance"], 4),
                "source": h["entity"]["source"],
                "text": h["entity"]["text"]} for h in hits]
        contexts = [c for c in raw if c["score"] >= min_score]      # ← 阈值过滤

        t0 = time.perf_counter()
        if contexts:                                                # ← 空上下文保护
            materials = "\n\n".join(f"[{i+1}] 来源：{c['source']}\n{c['text']}"
                                    for i, c in enumerate(contexts))
            resp = get_llm().invoke(PROMPT_TMPL.format(system=domain_mod.system_prompt(_DOMAIN), contexts=materials, question=question))
            answer, usage = resp.content, resp.usage_metadata
        else:
            answer = f"（{len(raw)} 条候选全部低于阈值 {min_score}，无材料可用）"
        stage_ms["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    except Exception as e:
        status, error = "error", repr(e)
        logger.exception("trace=%s 失败：%s", trace_id, e)
        raise
    finally:
        cost = observe.calc_cost(usage)
        span = observe.build_retrieval_span(contexts, raw_n=len(raw))
        observe.write_trace({
            "trace_id": trace_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "params": {"top_k": top_k, "min_score": min_score},     # ← 参数进 trace
            "stage_ms": stage_ms, "usage": usage, "cost_cny": cost,
            "retrieval": span, "status": status, "error": error,
        })
        logger.info("trace=%s status=%s top_k=%s min_score=%s n_raw=%s n_kept=%s stage_ms=%s cost=%.6f元",
                    trace_id, status, top_k, min_score, len(raw), len(contexts), stage_ms, cost)

    _safe = safety_rules.apply_all(question, _finalize_answer(answer), _DOMAIN)
    return {"answer": _safe["final_answer"], "contexts": contexts, "usage": usage, "stage_ms": stage_ms,
            "top_k": top_k, "min_score": min_score, "n_raw": len(raw),
            "safety": {"input": _safe["input"].action,
                       "output": _safe["output"].action if _safe["output"] else None}}
FORMS = domain_mod.form_keys(_DOMAIN)   # 来自领域配置；检索逻辑在 src/retrieval.py
def rag_query_multi(question: str, per_form: int = 2, min_score: float = 0.0,
                    top_k: int = 3, hybrid: bool = False) -> dict:
    """按形态多路召回：每个形态各取 per_form 条，合并后统一过滤/生成。

    `hybrid=True` 时额外走 BM25 稀疏召回并做 RRF 融合。
    ⚠️ 默认 **False**：实测混合检索的排名质量不如纯向量多路，且延迟约 2.7 倍
    （详见 `DESIGN.md` 第 9 节）。
    """
    stage_ms, contexts, usage, answer, raw = {}, [], None, None, []
    trace_id = observe.new_trace_id()
    status, error = "ok", None
    try:
        t0 = time.perf_counter()
        qvec = get_emb().embed_query(question)
        stage_ms["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        if hybrid:
            raw, stage = rt.retrieve_hybrid(get_client(), qvec, question, COLLECTION,
                                            per_form, _get_bm25_bundle())
            stage_ms["search_ms"] = round(
                stage["dense_ms"] + stage["sparse_ms"] + stage["fuse_ms"], 1)
            # 三段分开记录，否则无法判断混合检索把时间花在哪
            for key in ("dense_ms", "sparse_ms", "fuse_ms"):
                stage_ms[key] = stage[key]
            stage_ms["n_dense"], stage_ms["n_sparse"] = stage["n_dense"], stage["n_sparse"]
            contexts = raw                     # 已按融合顺序排好，不再重排
        else:
            # 按形态并发召回，替代原来的 4 次串行请求：
            # 语义完全不变（仍是每形态各 per_form 条），只是不再等待叠加。
            merged, per_form_ms, wall_ms = rt.retrieve_dense_multi(
                get_client(), qvec, COLLECTION, per_form)
            stage_ms["search_ms"] = wall_ms
            stage_ms["search_per_form_ms"] = per_form_ms
            raw = sorted(merged, key=lambda c: c["score"], reverse=True)
            contexts = [c for c in raw if c["score"] >= min_score]
        t0 = time.perf_counter()
        if contexts:
            materials = "\n\n".join(
                f"[{i + 1}] 来源：{c['source']}\n{c['text']}" for i, c in enumerate(contexts)
            )
            resp = get_llm().invoke(PROMPT_TMPL.format(system=domain_mod.system_prompt(_DOMAIN), contexts=materials, question=question))
            answer, usage = resp.content, resp.usage_metadata
        else:
            answer = f"（{len(raw)} 条候选全部低于阈值 {min_score}，无材料可用）"
        stage_ms["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        status, error = "error", repr(e)
        logger.exception("trace=%s 失败：%s", trace_id, e)
        raise
    finally:
        cost = observe.calc_cost(usage)
        span = observe.build_retrieval_span(contexts, raw_n=len(raw))
        observe.write_trace({
            "trace_id": trace_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "params": {"mode": "hybrid" if hybrid else "multi",
                       "per_form": per_form, "top_k": top_k, "min_score": min_score},  # ← 参数进 trace
            "stage_ms": stage_ms, "usage": usage, "cost_cny": cost,
            "retrieval": span, "status": status, "error": error,
        })
        logger.info("trace=%s status=%s per_form=%s mode=%s min_score=%s n_raw=%s n_kept=%s stage_ms=%s cost=%.6f元",
                    trace_id, status, per_form, "hybrid" if hybrid else "multi",
                    min_score, len(raw), len(contexts), stage_ms, cost)

    _safe = safety_rules.apply_all(question, _finalize_answer(answer), _DOMAIN)
    return {"answer": _safe["final_answer"], "contexts": contexts, "usage": usage, "stage_ms": stage_ms,
            "per_form": per_form, "mode": "hybrid" if hybrid else "multi",
            "min_score": min_score, "n_raw": len(raw),
            "safety": {"input": _safe["input"].action,
                       "output": _safe["output"].action if _safe["output"] else None}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 查询（可调参实验）")
    parser.add_argument("-q", "--question", default="生产经营单位的安全生产责任制是如何规定的？")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--per-form", type=int, default=2)
    parser.add_argument("--hybrid", action="store_true",
                        help="启用混合检索（dense + BM25 + RRF）。"
                             "默认关闭：实测排名质量不如纯向量多路，且延迟约 2.7 倍")

    args = parser.parse_args()
    if args.mode == "multi":
        out = rag_query_multi(args.question, per_form=args.per_form,
                              min_score=args.min_score, hybrid=args.hybrid)
        print(f"【问题】{args.question}")
        print(f"【参数】per_form={out['per_form']}  min_score={out['min_score']}"
              f"  |  候选 {out['n_raw']} → 保留 {len(out['contexts'])}")
        print(f"【回答】{out['answer']}\n")
        for i, c in enumerate(out["contexts"], 1):
            print(f"  [{i}] score={c['score']} | {c['source']}")
        print(f"【阶段耗时】{out['stage_ms']}")
        print(f"【token】{out['usage']}")
    else:
        out = rag_query(args.question, top_k=args.top_k, min_score=args.min_score)
        print(f"【问题】{args.question}")
        print(f"【参数】top_k={out['top_k']}  min_score={out['min_score']}"
              f"  |  候选 {out['n_raw']} → 保留 {len(out['contexts'])}")
        print(f"【回答】{out['answer']}\n")
        for i, c in enumerate(out["contexts"], 1):
            print(f"  [{i}] score={c['score']} | {c['source']}")
        print(f"【阶段耗时】{out['stage_ms']}")
        print(f"【token】{out['usage']}")
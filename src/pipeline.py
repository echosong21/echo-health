import argparse
import sys,time
from datetime import datetime
from pathlib import Path

# `python src/pipeline.py` 直接运行时，sys.path[0] 是 src/，找不到 config 包 ——
# 所以这行必须留在模块级（下面的 `from config import settings` 依赖它）。
# pytest 侧由 pytest.ini 的 `pythonpath = .` 覆盖，不依赖这行；
# 重复插入只是一个无害的重复项。
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

# ⚠️ 重型依赖（langchain_openai / pymilvus）**刻意不在模块级导入**：
#    它们只在 get_emb / get_llm / get_client 里按需导入，
#    这样「只想测 domain / safety 逻辑」的环境不必安装它们。
#    （输出编码 reconfigure 也移到 `__main__`，避免 import 时改全局 stdout。）
from config import settings
from src import retrieval as rt
from src import domain as domain_mod
from src import observe
from src.safety import rules as safety_rules

_domain: dict | None = None


def get_domain() -> dict:
    """领域配置（首次调用时加载并校验）。

    ⚠️ 改成懒加载的理由：

        原实现在**模块级**执行 `domain_mod.load()` 并校验，于是
        `import src.pipeline` 就会去读 `domains/*.yaml`；缺 yaml 时抛
        `RuntimeError`，**整个模块无法导入** —— 连带 `domain` / `safety`
        的单测也无法被 pytest 收集。

        懒加载后：导入零副作用，缺配置在真正用到时才报错、位置更准确
        （与 `get_emb` / `get_llm` / `get_client` 保持同一风格）。
    """
    global _domain
    if _domain is None:
        try:
            _domain = domain_mod.load()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"领域配置加载失败（检查 domains/*.yaml）：{exc}"
            ) from exc
    return _domain


def get_collection() -> str:
    """当前领域的集合名（换领域时本文件无需改动）。"""
    return get_domain()["collection"]

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
        from langchain_openai import OpenAIEmbeddings  # 函数内导入，见文件头说明

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
        from langchain_openai import ChatOpenAI  # 函数内导入，见文件头说明

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
        from pymilvus import MilvusClient  # 函数内导入，见文件头说明

        _client = MilvusClient(uri=settings.MILVUS_URL)
    return _client

# ── 提示词拼装（能力层固定）────────────────────────────────────────
#
# 领域提示词本身来自 domains/*.yaml 的 prompt.system，不写在这里。
#
# ⚠️ **为什么用函数拼而不是 str.format 模板**
#
# 原实现是 `PROMPT_TMPL.format(system=..., contexts=..., question=...)`。
# 加记忆时最自然的想法是模板里加一个 `{memory_block}` 槽位 —— **那是错的**：
#   · 槽位为空时，模板里的定界换行**仍然存在**
#   · 槽位非空时，渲染出来的记忆块又自带换行
# 两种情况换行数对不齐，结果是「无记忆时多一个空行」或「有记忆时少一个换行」。
# 这类偏差肉眼极难发现，但会让 EVAL_REPORT 的历史指标（17/17、70.6%、0.8284）
# 失去可比性 —— 那些数字是拿旧 prompt 跑出来的。
#
# 所以改成**条件拼接**：只有 memory_block 非空时才插入，其余字节原样不动。

MEMORY_EPISODE_WINDOW = 6   # 渲染进 prompt 的最近条数（约 3 轮）


def _render_memory(mem: dict | None, max_episodes: int = MEMORY_EPISODE_WINDOW) -> str:
    """把记忆渲染成 prompt 片段。

    **空记忆必须返回空字符串** —— 这是 `_build_prompt` 保持字节级兼容的前提。
    """
    if not mem:
        return ""
    facts = mem.get("facts") or {}
    episodes = (mem.get("episodes") or [])[-max_episodes:]
    if not facts and not episodes:
        return ""

    lines: list[str] = []
    if facts:
        lines.append("已知的用户情况：" + "；".join(f"{k}={v}" for k, v in facts.items()))
    if episodes:
        lines.append("近期对话：")
        lines.extend(f"- {e['role']}: {e['text']}" for e in episodes)
    return "\n".join(lines)


def _build_prompt(system: str, contexts: str, question: str, memory_block: str = "") -> str:
    """组装最终 prompt。

    ⚠️ **`memory_block` 为空时，输出必须与改造前逐字节相同**：

        'SYS\\n\\n材料：\\nCTX\\n\\n问题：Q\\n'

    这条由 `tests/test_memory.py::test_no_memory_prompt_is_byte_identical` 守住。
    """
    head = system
    if memory_block:
        head += f"\n\n{memory_block}"
    return f"{head}\n\n材料：\n{contexts}\n\n问题：{question}\n"


def _load_memory_block(session_id: str | None) -> str:
    """读记忆并渲染。**任何异常都折成空串** —— 记忆失败不能让问答失败。"""
    if not session_id:
        return ""
    try:
        from src.memory import get_store

        return _render_memory(get_store().read(session_id=session_id))
    except Exception:                                   # noqa: BLE001
        return ""


def _save_turns(session_id: str | None, question: str, answer: str | None) -> None:
    """把这一轮写进记忆。**失败静默**，且空回答不写（会污染下一轮上下文）。"""
    if not session_id or not answer:
        return
    try:
        from src.memory import get_store

        store = get_store()
        store.write(session_id=session_id, role="user", text=question)
        store.write(session_id=session_id, role="assistant", text=answer)
    except Exception:                                   # noqa: BLE001
        pass


def _finalize_answer(answer: str | None) -> str | None:
    """按领域配置注入免责声明（**强制注入**，Q41 硬要求）。

    - `disclaimers` 为空（如 law 领域）⇒ 原样返回，**行为与改造前完全一致**；
    - 非空（如 health 领域）⇒ 按 `position` 注入到答案前后。
    """
    if answer is None:
        return None
    pre = domain_mod.disclaimer_texts(get_domain(), "prepend")
    app = domain_mod.disclaimer_texts(get_domain(), "append")
    parts = [*pre, answer, *app]
    return "\n\n".join(parts) if (pre or app) else answer
logger = observe.setup_logging()
def rag_query(question: str, top_k: int = 3, min_score: float = 0.0,
              session_id: str | None = None) -> dict:
    """min_score=0.0 ⇒ 不过滤，行为与改造前完全一致。

    `session_id` 非空时才读写记忆；为 None 时**不碰记忆层**，
    prompt 与改造前逐字节相同（见 `_build_prompt` 的说明）。
    """
    stage_ms, contexts, usage, answer, raw = {}, [], None, None, []
    trace_id = observe.new_trace_id()
    status, error = "ok", None
    memory_block = _load_memory_block(session_id)
    try:
        t0 = time.perf_counter()
        qvec = get_emb().embed_query(question)
        stage_ms["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        hits = get_client().search(collection_name=get_collection(), data=[qvec],
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
            resp = get_llm().invoke(_build_prompt(
                system=domain_mod.system_prompt(get_domain()),
                contexts=materials, question=question, memory_block=memory_block))
            answer, usage = resp.content, resp.usage_metadata
        else:
            answer = f"（{len(raw)} 条候选全部低于阈值 {min_score}，无材料可用）"
        stage_ms["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    except Exception as e:
        status, error = "error", repr(e)
        logger.exception("trace=%s 失败：%s", trace_id, e)
        raise
    finally:
        _save_turns(session_id, question, answer)
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

    _safe = safety_rules.apply_all(question, _finalize_answer(answer), get_domain())
    return {"answer": _safe["final_answer"], "contexts": contexts, "usage": usage, "stage_ms": stage_ms,
            "top_k": top_k, "min_score": min_score, "n_raw": len(raw),
            "safety": {"input": _safe["input"].action,
                       "output": _safe["output"].action if _safe["output"] else None}}


def rag_query_multi(question: str, per_form: int = 2, min_score: float = 0.0,
                    top_k: int = 3, hybrid: bool = False,
                    session_id: str | None = None) -> dict:
    """按形态多路召回：每个形态各取 per_form 条，合并后统一过滤/生成。

    `hybrid=True` 时额外走 BM25 稀疏召回并做 RRF 融合。
    ⚠️ 默认 **False**：实测混合检索的排名质量不如纯向量多路，且延迟约 2.7 倍
    （详见 `DESIGN.md` 第 9 节）。

    `session_id` 非空时才读写记忆；为 None 时 prompt 与改造前逐字节相同。
    """
    stage_ms, contexts, usage, answer, raw = {}, [], None, None, []
    trace_id = observe.new_trace_id()
    status, error = "ok", None
    memory_block = _load_memory_block(session_id)
    try:
        t0 = time.perf_counter()
        qvec = get_emb().embed_query(question)
        stage_ms["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        if hybrid:
            raw, stage = rt.retrieve_hybrid(get_client(), qvec, question, get_collection(),
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
                get_client(), qvec, get_collection(), per_form)
            stage_ms["search_ms"] = wall_ms
            stage_ms["search_per_form_ms"] = per_form_ms
            raw = sorted(merged, key=lambda c: c["score"], reverse=True)
            contexts = [c for c in raw if c["score"] >= min_score]
        t0 = time.perf_counter()
        if contexts:
            materials = "\n\n".join(
                f"[{i + 1}] 来源：{c['source']}\n{c['text']}" for i, c in enumerate(contexts)
            )
            resp = get_llm().invoke(_build_prompt(
                system=domain_mod.system_prompt(get_domain()),
                contexts=materials, question=question, memory_block=memory_block))
            answer, usage = resp.content, resp.usage_metadata
        else:
            answer = f"（{len(raw)} 条候选全部低于阈值 {min_score}，无材料可用）"
        stage_ms["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        status, error = "error", repr(e)
        logger.exception("trace=%s 失败：%s", trace_id, e)
        raise
    finally:
        _save_turns(session_id, question, answer)
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

    _safe = safety_rules.apply_all(question, _finalize_answer(answer), get_domain())
    return {"answer": _safe["final_answer"], "contexts": contexts, "usage": usage, "stage_ms": stage_ms,
            "per_form": per_form, "mode": "hybrid" if hybrid else "multi",
            "min_score": min_score, "n_raw": len(raw),
            "safety": {"input": _safe["input"].action,
                       "output": _safe["output"].action if _safe["output"] else None}}


if __name__ == "__main__":
    # 只在 CLI 直接运行时改输出编码 —— 放在模块级会污染 import 本模块的进程。
    sys.stdout.reconfigure(encoding="utf-8")

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
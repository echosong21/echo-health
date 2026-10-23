#!/usr/bin/env python3
"""建库脚本：语料 → 切分 → 向量 → Milvus。

把一批文本语料切分成块、批量向量化、写入 Milvus 集合，并做一次检索自检。

用法：
    python scripts/build_index.py --corpus ./corpus
    python scripts/build_index.py --corpus ./corpus --chunk-size 300 --chunk-overlap 60
    python scripts/build_index.py --corpus ./corpus --index-type HNSW \
        --index-params '{"M": 16, "efConstruction": 200}'
    python scripts/build_index.py --corpus ./corpus --dry-run     # 只切分+向量化，不写库

参数优先级：命令行 > 代码默认值（见 DEFAULTS）。
默认值写在代码里，是为了让仓库 clone 下来即可复现 —— 环境变量不进版本库，
把参数只放在环境变量里会导致实验无法被他人复现。

语料约定：`--corpus` 目录下每个**子目录名**即一个数据形态标签（如 docs/ faq/），
该标签会写进每块的 `source` 字段（形如 `docs/xxx.md`），供检索侧按形态过滤。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

def get_settings():
    """延迟加载配置对象。

    为什么不在模块级 `from config import settings`：
    那样会在 **import 时**就构造配置，缺少 `.env`（CI 环境、全新 clone、
    或只是跑一下 `--help`）会直接抛异常，连参数帮助都看不到。
    改成函数后，只有真正需要 embedding / Milvus 的代码路径才加载配置。
    """
    from config import settings as _settings  # noqa: PLC0415

    return _settings

# ── 默认参数（写在代码里 ⇒ clone 即可复现）────────────────────────────
DEFAULTS = {
    "chunk_size": 600,
    "chunk_overlap": 80,
    "metric_type": "COSINE",
    "index_type": "",        # 空字符串 ⇒ 使用服务端默认索引
    "index_params": {},
    "embed_batch": 20,       # 批量向量化的批大小
    "insert_batch": 500,     # 批量写入的批大小
}

# 默认可读的文本扩展名。带二进制体（如 PDF）的语料需先解析成文本，否则会读出乱码。
TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}

# 自检用的查询（确认建完就能检索到东西）
SELF_CHECK_QUERIES = [
    "生产经营单位的安全生产责任制是如何规定的？",
    "订婚后退婚，彩礼能不能要回来？",
]


def discover_forms(corpus: Path, only: list[str] | None = None) -> list[str]:
    """语料目录下的子目录名即形态标签。

    跳过以 `_` 或 `.` 开头的目录 —— 它们是工作目录（处理脚本、缓存等），不是语料。
    """
    if not corpus.is_dir():
        raise SystemExit(f"语料目录不存在：{corpus}")
    forms = sorted(
        p.name
        for p in corpus.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    )
    if only:
        missing = [f for f in only if f not in forms]
        if missing:
            raise SystemExit(f"指定的形态在语料目录下不存在：{missing}")
        forms = only
    if not forms:
        raise SystemExit(f"语料目录下没有任何子目录：{corpus}")
    return forms


def build_items(
    corpus: Path,
    forms: list[str],
    per_form_limit: dict[str, int],
    chunk_size: int,
    chunk_overlap: int,
    extensions: set[str] | None = None,
) -> list[dict]:
    """读取语料并按形态切分，返回 [{text, source, chunk_index}, ...]。"""
    exts = extensions or TEXT_EXTENSIONS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    items: list[dict] = []
    for form in forms:
        folder = corpus / form
        all_files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
        files = [p for p in all_files if p.suffix.lower() in exts]
        skipped = len(all_files) - len(files)
        limit = per_form_limit.get(form)
        if limit is not None:
            files = files[:limit]
        before = len(items)
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                print(f"  [skip] {path.name}: {exc}")
                continue
            for i, chunk in enumerate(splitter.split_text(text)):
                items.append(
                    {
                        "text": chunk,
                        "source": f"{form}/{path.name}",
                        "chunk_index": i,
                    }
                )
        note = f"（跳过 {skipped} 份非文本文件）" if skipped else ""
        print(f"[1] {form:<12s} {len(files):4d} 份 → +{len(items) - before:6d} 块"
              f"（累计 {len(items)}）{note}")
    return items


def embed_items(items: list[dict], batch: int) -> list[list[float]]:
    """分批向量化，带进度与 ETA（百炼单批上限 20）。"""
    settings = get_settings()          # 懒加载：只有真要跑 embedding 时才需要配置
    emb = OpenAIEmbeddings(
        model=settings.EMBED_MODEL,
        base_url=settings.EMBED_BASE_URL,
        api_key=settings.API_KEY,
        check_embedding_ctx_length=False,  # 兼容 OpenAI 协议端点，必须关闭
    )
    vectors: list[list[float]] = []
    t0 = time.time()
    for start in range(0, len(items), batch):
        vectors.extend(
            emb.embed_documents([it["text"] for it in items[start:start + batch]])
        )
        done = len(vectors)
        if done % (batch * 25) == 0 or done == len(items):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = (len(items) - done) / rate if rate else 0
            print(f"[2] {done}/{len(items)}  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
    print(f"[2] 向量完成：{len(vectors)} 条，维度 {len(vectors[0])}，"
          f"总耗时 {time.time() - t0:.0f}s")
    return vectors


def make_index_params(index_type: str, params: dict, metric_type: str):
    """构造 IndexParams；index_type 为空时返回 None（用服务端默认索引）。"""
    if not index_type:
        return None
    from pymilvus.milvus_client.index import IndexParams

    ip = IndexParams()
    ip.add_index(
        field_name="vector",
        index_type=index_type,
        metric_type=metric_type,
        params=params or {},
    )
    return ip


def write_to_milvus(
    collection: str,
    dim: int,
    metric_type: str,
    index_type: str,
    index_params: dict,
    items: list[dict],
    vectors: list[list[float]],
    insert_batch: int,
) -> None:
    from pymilvus import MilvusClient

    settings = get_settings()          # 懒加载：只有真要写 Milvus 时才需要配置
    client = MilvusClient(uri=settings.MILVUS_URL)
    print(f"[3] Milvus 已连接：{settings.MILVUS_URL}｜"
          f"version = {client.get_server_version()}")

    if client.has_collection(collection):
        client.drop_collection(collection)
        print(f"    已删除旧集合 {collection}"
              f"（换 embedding 模型必须重建：向量空间不可混用）")

    ip = make_index_params(index_type, index_params, metric_type)
    client.create_collection(
        collection_name=collection,
        dimension=dim,
        metric_type=metric_type,
        index_params=ip,
    )
    print(f"    已创建集合 {collection}｜dim={dim}｜metric={metric_type}｜"
          f"index={index_type or '服务端默认'}｜params={index_params or {}}")

    t1 = time.time()
    for start in range(0, len(items), insert_batch):
        end = min(start + insert_batch, len(items))
        rows = [
            {
                "id": i,
                "vector": vectors[i],
                "text": items[i]["text"],
                "source": items[i]["source"],
                "chunk_index": items[i]["chunk_index"],
            }
            for i in range(start, end)
        ]
        client.insert(collection_name=collection, data=rows)
        print(f"[3] insert {end}/{len(items)}  ({time.time() - t1:.0f}s)", flush=True)

    client.flush(collection)
    print(f"[3] stats = {client.get_collection_stats(collection)}")
    print(f"[3] count(*) = "
          f"{client.query(collection_name=collection, filter='', output_fields=['count(*)'])}")

    for query in SELF_CHECK_QUERIES:
        emb = OpenAIEmbeddings(
            model=settings.EMBED_MODEL,
            base_url=settings.EMBED_BASE_URL,
            api_key=settings.API_KEY,
            check_embedding_ctx_length=False,
        )
        hits = client.search(
            collection_name=collection,
            data=[emb.embed_query(query)],
            limit=3,
            output_fields=["source", "chunk_index"],
        )
        print(f"\n[4] 自检查询：{query}")
        for rank, hit in enumerate(hits[0], 1):
            ent = hit["entity"]
            print(f"    [{rank}] score={hit['distance']:.4f} | {ent['source']} "
                  f"| chunk#{ent['chunk_index']}")


def parse_per_form_limit(raw: list[str]) -> dict[str, int]:
    """--per-form-limit docs=100 tables=10 → {'docs': 100, 'tables': 10}"""
    result: dict[str, int] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--per-form-limit 需要 FORM=N 形式，收到：{item}")
        form, _, n = item.partition("=")
        result[form.strip()] = int(n)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="建库：语料 → 切分 → 向量 → Milvus")
    ap.add_argument("--corpus", required=True, help="语料根目录（其子目录名即形态标签）")
    ap.add_argument("--domain", default="",
                    help="领域配置名（如 health）。指定后从 domains/<name>.yaml 读取 "
                         "collection / chunk_size / chunk_overlap / per_form_limit；"
                         "命令行上显式给出的值优先于配置文件")
    ap.add_argument("--collection", default="",
                    help="集合名；留空则取 --domain 配置，再退回 settings.COLLECTION")
    # ⚠️ 默认值必须为 None，否则无法区分「用户没传」和「用户传了等于默认的值」，
    #    那样 --domain 里的 chunk_size 就永远被 argparse 的默认值盖掉。
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--metric-type", default=DEFAULTS["metric_type"])
    ap.add_argument("--index-type", default=DEFAULTS["index_type"],
                    help="索引类型（如 HNSW / IVF_FLAT）；留空则用服务端默认索引")
    ap.add_argument("--index-params", default="",
                    help='索引参数 JSON，如 \'{"M": 16, "efConstruction": 200}\'')
    ap.add_argument("--embed-batch", type=int, default=DEFAULTS["embed_batch"])
    ap.add_argument("--insert-batch", type=int, default=DEFAULTS["insert_batch"])
    ap.add_argument("--per-form-limit", action="append", default=[],
                    help="限制某形态的文件数（可重复），如 --per-form-limit tables=10")
    ap.add_argument("--forms", default="",
                    help="只处理指定形态（逗号分隔）；默认处理全部子目录")
    ap.add_argument("--extensions", default="",
                    help="只处理这些扩展名（逗号分隔）；默认见 TEXT_EXTENSIONS")
    ap.add_argument("--dry-run", action="store_true", help="只切分+向量化，不写 Milvus")
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    only = [f.strip() for f in args.forms.split(",") if f.strip()] or None
    forms = discover_forms(corpus, only)
    index_params = json.loads(args.index_params) if args.index_params else {}

    # ── 领域配置 → 补齐未显式传入的参数 ────────────────────────
    # 优先级：命令行 > domains/<name>.yaml > DEFAULTS
    domain_cfg: dict = {}
    if args.domain:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.domain import load as _load_domain  # noqa: PLC0415

        domain_cfg = _load_domain(args.domain)
    domain_corpus: dict = domain_cfg.get("corpus") or {}
    _source: dict[str, str] = {}

    def _resolve(cli_value, cfg_key: str, default, label: str):
        """返回 (值, 来源说明)。命令行显式给出时优先。"""
        if cli_value is not None:
            return cli_value, "命令行"
        if cfg_key in domain_corpus and domain_corpus[cfg_key] is not None:
            return domain_corpus[cfg_key], f"{args.domain}.yaml"
        return default, "默认值"

    chunk_size, _source["chunk_size"] = _resolve(
        args.chunk_size, "chunk_size", DEFAULTS["chunk_size"], "chunk_size")
    chunk_overlap, _source["chunk_overlap"] = _resolve(
        args.chunk_overlap, "chunk_overlap", DEFAULTS["chunk_overlap"], "chunk_overlap")
    # 到这里说明要真正干活了（--help 已在 argparse 阶段退出），此时才需要配置。
    settings = get_settings()
    if args.collection:
        collection, _source["collection"] = args.collection, "命令行"
    elif domain_cfg.get("collection"):
        collection, _source["collection"] = domain_cfg["collection"], f"{args.domain}.yaml"
    elif hasattr(settings, "COLLECTION"):
        collection, _source["collection"] = settings.COLLECTION, "settings"
    else:
        collection, _source["collection"] = "law_docs", "默认值"

    # per_form_limit：yaml 打底，命令行逐形态覆盖
    per_form_limit = dict(domain_corpus.get("per_form_limit") or {})
    _pfl_source = f"{args.domain}.yaml" if per_form_limit else ""
    _cli_limits = parse_per_form_limit(args.per_form_limit)
    if _cli_limits:
        per_form_limit.update(_cli_limits)
        _pfl_source = (_pfl_source + " + 命令行") if _pfl_source else "命令行"

    extensions = (
        {"." + e.strip().lstrip(".").lower() for e in args.extensions.split(",") if e.strip()}
        if args.extensions else None
    )

    print(f"语料目录：{corpus}")
    if args.domain:
        print(f"领域配置：domains/{args.domain}.yaml")
    print(f"形态：{forms}")
    print(f"每形态文件上限：{per_form_limit or '不限'}"
          + (f"（来源：{_pfl_source}）" if _pfl_source else ""))
    print(f"切分：chunk_size={chunk_size}（{_source['chunk_size']}）"
          f" chunk_overlap={chunk_overlap}（{_source['chunk_overlap']}）")
    print(f"集合：{collection}（{_source['collection']}）")
    print(f"embedding 模型：{settings.EMBED_MODEL}")
    print(f"索引：{args.index_type or '服务端默认'}｜params={index_params or {}}")
    print()

    items = build_items(corpus, forms, per_form_limit, chunk_size, chunk_overlap, extensions)
    if not items:
        raise SystemExit("没有可入库的块（检查语料目录与 --forms）")
    print(f"[1] 语料合计：{len(items)} 块\n")

    vectors = embed_items(items, args.embed_batch)

    if args.dry_run:
        print("[3] --dry-run：跳过 Milvus 写入与自检")
        return 0

    write_to_milvus(
        collection=collection,
        dim=len(vectors[0]),
        metric_type=args.metric_type,
        index_type=args.index_type,
        index_params=index_params,
        items=items,
        vectors=vectors,
        insert_batch=args.insert_batch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
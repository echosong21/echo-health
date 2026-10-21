import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRACE_FILE = LOG_DIR / "traces.jsonl"

# ── 成本单价（元 / 百万 token）⚠️ 待以百炼控制台账单核实后校准 ──
PRICE = {
    "input": 2.0,
    "output": 8.0,
}


def setup_logging(level=logging.INFO) -> logging.Logger:
    """分级日志 → 落 logs/app_YYYY-MM-DD.log（控制台也输出一份）"""
    logfile = LOG_DIR / f"app_{datetime.now():%Y-%m-%d}.log"
    logger = logging.getLogger("rag")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
def new_trace_id() -> str:
    """8 位短 id：把日志行与 jsonl 记录对应起来"""
    return uuid.uuid4().hex[:8]


def calc_cost(usage: dict | None) -> float:
    """token → 人民币（元），保留 6 位小数"""
    if not usage:
        return 0.0
    cost = (
        usage.get("input_tokens",0) * PRICE["input"]
        + usage.get("output_tokens",0) * PRICE["output"]
    ) / 1_000_000
    return round(cost,6)


def write_trace(record: dict) -> None:
    """追加写 jsonl（一行一条）"""
    with open(TRACE_FILE,"a",encoding="utf-8")as f:
        f.write(json.dumps(record,ensure_ascii=False) + "\n")


def build_retrieval_span(contexts: list[dict], raw_n: int | None = None) -> dict:
    """E 埋点：分数分布 + 来源层级（raw_n = 阈值过滤前的条数）"""
    scores = [c["score"] for c in contexts]
    layers = sorted({c["source"].split("/")[0] for c in contexts})
    return {
        "n": len(scores),
        "n_raw": raw_n if raw_n is not None else len(scores),
        "n_kept": len(scores),
        "scores": scores,
        "top1": scores[0] if scores else None,
        "min": min(scores) if scores else None,
        "avg": round(sum(scores) / len(scores), 4) if scores else None,
        "layers": layers,
    }
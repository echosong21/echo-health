"""日志可串联性的量化验收用例。

**这条用例断言什么**

    一次请求产生的**全部日志行**都含同一个 `trace_id`，
    且 `traces.jsonl` 的每一行都能 `json.loads` 解析。

**为什么需要它**

    原始判据是「日志能串起一次完整请求」—— 这句话**无法客观判定**：
    人眼觉得串起来了，换个人可能不这么看，评审时争不出结论。

    加上「trace_id 一致」+「逐行 JSON 可解析」两条硬断言后，
    它从一个说法变成一条**会红也会绿的测试**。

**为什么能离线跑（不连 Milvus / embedding / LLM）**

    `rag_query` 的四个外部依赖全部经 getter 取得：

        get_emb() / get_client() / get_llm() / get_domain()

    测试里 monkeypatch 掉即可 —— 这是依赖注入改造的直接收益。
    改造前它们由模块级对象提供，只能起真实服务才能测。
"""
from __future__ import annotations

import json


# ── 替身：LLM ────────────────────────────────────────────────────
class _FakeLLMResponse:
    def __init__(self, content: str, usage: dict):
        self.content = content
        self.usage_metadata = usage


class _FakeLLM:
    """只实现 `invoke()`，返回固定内容与 usage，并记录被调用次数。"""

    def __init__(self, content: str = "米饭每 100 克含能量 116 千卡。"):
        self.content = content
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        return _FakeLLMResponse(
            self.content,
            {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )


# ── 替身：logger ─────────────────────────────────────────────────
class _RecordingLogger:
    """把 `info` / `exception` 的参数按 `%` 格式化后存下来，供断言。

    ⚠️ 替换它不只是为了断言，也是**防止测试污染真实 `logs/app_*.log`**：
    `pipeline.logger` 是模块级对象，不替换的话每次跑测试都会往真实日志追加。
    """

    def __init__(self):
        self.records: list[str] = []

    def info(self, msg, *args, **kwargs):
        self.records.append(msg % args if args else msg)

    def exception(self, msg, *args, **kwargs):
        self.records.append(msg % args if args else msg)


# ── 主用例 ───────────────────────────────────────────────────────
def test_trace_id_consistent_and_json_parsable(
    tmp_trace_file, fake_client, make_hits, fake_emb, health_domain, monkeypatch,
):
    """一次请求的全部日志行含同一 trace_id；traces.jsonl 每行可解析。"""
    from src import pipeline as pl

    # ① 四个外部依赖全部换成替身（离线 + 确定性）
    monkeypatch.setattr(pl, "get_emb", lambda: fake_emb)
    fake_client.hits_by_form[None] = make_hits(n=2, form="guides")
    monkeypatch.setattr(pl, "get_client", lambda: fake_client)

    fake_llm = _FakeLLM()
    monkeypatch.setattr(pl, "get_llm", lambda: fake_llm)

    # 固定用 health 领域，避免依赖 DOMAIN 环境变量
    monkeypatch.setattr(pl, "get_domain", lambda: health_domain)

    # ② logger 换成记录器（同时避免污染真实日志）
    recorder = _RecordingLogger()
    monkeypatch.setattr(pl, "logger", recorder)

    # ③ 跑一次真实链路（single 模式）
    result = pl.rag_query("米饭每100克含多少热量？", top_k=2)

    # ④ 链路本身要正常，否则后面的断言没有意义
    assert result["answer"], "回答不能为空"
    assert fake_llm.calls == 1, "应恰好调用一次 LLM"
    assert len(result["contexts"]) == 2, "应保留 2 条上下文"

    # ⑤ traces.jsonl：本次请求应只写 1 条，且可解析、trace_id 唯一
    raw_lines = tmp_trace_file.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1, f"一次请求应只写 1 条 trace，实际 {len(raw_lines)} 条"

    records = [json.loads(line) for line in raw_lines]  # 解析失败会直接抛错
    trace_ids = {r["trace_id"] for r in records}
    assert len(trace_ids) == 1, f"trace_id 应唯一，实际 {trace_ids}"

    # ⑥ 应用日志：每一行都要带同一个 trace=
    trace_id = trace_ids.pop()
    assert recorder.records, "至少应产生 1 行应用日志"
    missing = [m for m in recorder.records if f"trace={trace_id}" not in m]
    assert not missing, f"以下日志行缺少 trace={trace_id}：{missing}"

    # ⑦ trace 的关键字段齐备（这是「串起来」的实质：能定位到阶段耗时与成本）
    record = records[0]
    for field in ("trace_id", "ts", "question", "stage_ms", "usage", "cost_cny", "status"):
        assert field in record, f"trace 缺少字段 {field}"
    assert record["status"] == "ok"
    assert set(record["stage_ms"]) == {"embed_ms", "search_ms", "generate_ms"}
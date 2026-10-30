# 变更日志

本项目所有重要变更都记录在此。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

---

## [0.5.0] — 2026-10-30

### 新增

- **分层记忆**（`src/memory/`）—— 情景 / 语义两层，PostgreSQL 存储：
  - `base.py`：`MemoryStore` Protocol，**只有 `read` / `write` 两个方法**
  - `pg_store.py`：连接池、rolling 裁剪、facts 扁平 KV
  - `scripts/schema_memory.sql`：`episodes` / `facts` 两张表
  - `memory/README.md`：三行边界、四类限制值、部署前必改项
- **语义记忆写入口**（`src/memory/__init__.py` 的 `remember()`）——
  显式登记事实（过敏 / 忌口 / 慢性病 / 目标），**不做模型自动抽取**：
  抽取不可判定（无法写成断言），且猜错的代价远大于不记
- **多轮上下文**（`src/pipeline.py`）—— `rag_query` / `rag_query_multi`
  新增 `session_id` 与 `user_id`，默认值下行为与改造前完全一致
- **跨 session 端到端验收**（`scripts/verify_memory_e2e.py`）——
  两部分：同会话两轮（情景）＋ 跨会话事实（语义），带无记忆对照组
- **评测支持指定金标集**（`evals/run_eval.py` 的 `--golden`）——
  原先路径写死在模块常量里，只有一份 law 的 17 题
- **健康域冒烟基线**（`evals/health_smoke.json`）—— 10 条，覆盖
  guides / faq / tables / dialogues 四形态 + 跨形态

### 变更

- `rag_query` / `rag_query_multi` 的 `user_id` 默认空串，空时回落 `DEFAULT_USER_ID`。
  补这个参数之前，`remember()` 用别 uid 写入的事实**在问答时读不到** ——
  写端和读端用了不同的用户身份，单用户场景测不出来，一换用户就全面失效
- `DESIGN.md` 新增「分层记忆」节（四块齐全，含三轮措辞实验与可判定切换条件）

### 修复

- **`src/memory/__init__.py` 的模块 docstring 失效** —— `import threading`
  被放在文件第一行，挡在 docstring 前面，Python 只把第一个字符串字面量当
  docstring，所以 `help(src.memory)` 之前看不到任何说明
- **公开仓内的内部编号与框架术语**（10 处）—— 注释与 docstring 里残留了
  内部任务编号、评估框架名称与时段概念，这些不属于公开仓库。同时修了泄露
  扫描工具的两个缺陷：规则表缺 5 类；**扫单个文件路径会静默返回 OK**
  （等于没扫就报通过，比漏报更危险）

### 已知限制

- **`--auth=trust`（本地免密）**：`memory/README.md` 已写明部署前必须改
  `scram-sha-256` + 强密码 + 只监听 127.0.0.1
- **记忆失败静默降级**：`read` 返回空结构、`write` 记 warning 不抛 ——
  "记忆没生效"不会报错，只能靠日志发现
- **facts 无自动化写入者**：只有显式 `remember()`，没有任何链路会在用户说话时
  自动提取事实（刻意如此）

---

## [0.4.0] — 2026-10-27

### 新增

- **HTTP 层**（`src/api.py`）—— 三个端点，状态码语义分开：
  - `GET /health`：返回 `collection` 与 `num_entities`，用于区分
    「服务活着但索引空了」与「服务正常」
  - `POST /query` / `POST /query_multi`：单路与多形态召回
  - 400（语义无效）/ 422（schema 违规）/ 500（内部错误）/ 503（依赖不可用）
- **成本报表**（`scripts/cost_report.py`）—— 只读 `logs/traces.jsonl`：
  成本分位、超阈值占比、按状态与模式分组、延迟阶段拆分、数据质量检查
- **设计取舍文档**（`DESIGN.md`）—— 索引 / 切分与多形态 / 降级路径三节，
  每节给出替代方案与**可判定切换条件**
- **失败路径测试**（`tests/test_failure_paths.py`）—— 空数据 / 超时 /
  并发部分失败三类，并说明「并发部分失败选择冒泡而非吞掉」的理由
- **token 对比脚本**（`scripts/token_compare.py`）—— 真实走完「检索 → 生成」，
  对比不同 `min_score` 下的 `input_tokens` 与上下文条数。
  `run_eval.py` 只测检索、不调 LLM，因此回答不了成本问题，本脚本补这一环
- `run_eval.py` 新增 `--min-score` 参数（原先是硬编码常量）

### 变更

- `requirements.txt` 补齐 `fastapi` / `httpx` / `uvicorn` ——
  此前新增的 import 未同步依赖清单，本地能跑但 CI 在干净环境下失败

### 修复

- **测试不再写入真实 `logs/`**（`tests/conftest.py` 改 autouse 隔离）——
  两个污染源叠加曾让 `traces.jsonl` 里 41% 的记录来自测试：
  ① `observe.TRACE_FILE` 是模块级常量，显式夹具只有声明了参数的测试才生效；
  ② `pipelines.py` 在模块级调 `setup_logging()`，import 就创建真实 FileHandler。
  验证：跑全量 59 passed 前后 `traces.jsonl` 均 49 行 / 24719 B，完全不变
- **审计日志参数脱敏**中的同类问题：中文自由文本曾被判为枚举而原样落盘

### 成本治理结果（`min_score` 阈值）

同一批 17 题，`min_score` 从 `0.0` 提到 `0.5`：

| 指标 | 0.0 | 0.5 | 变化 |
|---|--:|--:|--:|
| 上下文均条数 | 12.00 | 5.71 | **−52%** |
| input_tokens 均 | 3674 | 1685 | **−54.1%** |
| 命中率 | 17/17 | 17/17 | 不变 |
| top-1 | 70.6% | 70.6% | 不变 |
| MRR | 0.8284 | 0.8284 | 不变 |

**⇒ 砍掉一半以上的上下文，质量三项完全不变。**

⚠️ `0.7` 不可用：命中率掉到 52.9%（漏 8 题），但 MRR 反升到 0.9444 ——
是精准率与召回率的权衡，健康场景漏答比答错更糟。

### 已知缺口

- `run_eval.py` 不记录 `usage`（它只测检索、不调 LLM）——
  成本类问题须用 `scripts/token_compare.py` 单独测，两者口径不同不可混比

---

## [0.2.0] — 2026-10-22

### 新增

- **安全拦截层**（`src/safety/rules.py`）—— 三道独立检查，规则全部来自领域配置：
  - **输入闸**：出现急症信号（`safety.must_refer`）时，强制在回答最前面插入转介语。
  - **输出闸**：命中禁用断言（`safety.forbidden_claims`）时**整条拒答**，而不是改写 —— 
    改写一个不安全的回答会引入新的、未经核验的表达。
  - **计划闸**：计划变动超过 `safety.max_plan_delta_per_day` 时拒绝执行。
- **导入守卫测试**（`tests/test_importable.py`）—— 静态检查「模块级不得构造客户端」，
  外加每个模块的可导入性检查。
- **安全闸接入主链路**：两条返回路径都过 `apply_all()`，判定结果写入请求 trace。

### 修复

- **消除模块级副作用**。`pipeline.py` 原先在**导入时**就构造 `OpenAIEmbeddings` /
  `ChatOpenAI` / `MilvusClient`，导致在没有 `.env` 的环境（CI、全新 clone、IDE 索引）中
  `import src.pipeline` 直接抛 `ValidationError`。现在改为由 `get_emb()` /
  `get_llm()` / `get_client()` **懒加载**，配置错误在首次使用时才暴露，而不是在导入时。

### 变更

- 测试数量从 2 个增加到 **33 个**（20 个安全规则 + 13 个导入守卫）。

---

## [0.1.0] — 2026-10-21

### 新增

- **能力层**：多形态稠密检索（并发）、基于 jieba 分词的 BM25 稀疏检索、两层 RRF 融合，
  以及可选的 rerank 阶段（**默认关闭**）。
- **领域配置化**（`domains/*.yaml`）：语料形态、提示词措辞、免责声明、安全规则、
  切分参数与集合名，全部放在按领域划分的 YAML 文件里。`src/domain.py` 是唯一入口，
  带 6 条 fail-fast 校验。
- **评测框架**（`evals/run_eval.py`）：命中率、期待来源居首率、MRR、判定分布。
- **可观测性**：逐请求 trace，含延迟分解与 token 成本。
- **CI**（lint + test），走 GitHub Actions。

### 说明

- 法律领域（`domains/law.yaml`）作为参照实现。用它跑出的检索指标与改造前基线**完全一致**
  （17/17 命中、期待来源居首 70.6%、MRR 0.8284）—— 配置化改的是**设置放在哪**，
  不是行为本身。

---

## 已知限制

- **健康领域尚未端到端可用。** 它的语料仍在整理中；`domains/health.yaml` 能加载、
  能通过校验，但还没有可检索的索引内容。
- **安全规则已生效，但尚未评测。** 安全闸有单元测试，但还没有拿真实的
  "诱导越界回答"样本集测过。
- **免责声明是静态注入**，按位置（`prepend` / `append`）拼接，没有按内容动态选择。
- **切分参数未按领域调优。** 600/80 是从参照实现继承的默认值；不同切分的成本与质量
  权衡记录在检索笔记里。

---

## 版本说明

项目处于 1.0 之前，接口仍在变动。领域 YAML 的 schema 在 1.0 之前预计会有破坏性变更。
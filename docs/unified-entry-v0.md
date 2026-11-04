# 统一入口 v0 · 三端点实测留痕

> **用途**：交付证据。三个端点各一次**真实调用**的原始响应，不是设计描述。
> **采集时间**：2026-11-04 17:00 前后
> **环境**：本机 `uvicorn src.api:app --port 8901`；Milvus（云服务器隧道）与 PostgreSQL 均未启动
> **服务版本**：`0.2.0`

---

## 为什么这份证据里有两个 503

**这不是缺陷，恰好是本设计的两个不同语义**：

| 端点 | 状态码 | 含义 |
|---|---|---|
| `/health` | **503** | Milvus（**必需品**）不可用 ⇒ 服务确实不能工作 |
| `/query` | **503** | 依赖层故障 ⇒ **不是内部错误**，调用方应稍后重试 |

**如果 `/health` 在 PostgreSQL 挂掉时也返 503，那才是错的** —— 记忆层有熔断降级，
PG 挂了照常答。本次环境里两个依赖都停了，所以看不到 `degraded` 分支；
那个分支由 `tests/test_api.py::test_health_degraded_when_pg_down` 覆盖。

---

## ① `GET /health` → HTTP 503

```json
{
  "status": "unavailable",
  "dependencies": {
    "milvus": "down",
    "postgres": "down"
  },
  "detail": "检索后端不可用，服务无法回答问题"
}
```

**读法**：
- `dependencies` **分别报告**两个依赖 —— 只回一个总状态无法定位是谁挂了
- 状态码由**必需品**（Milvus）决定；`postgres: down` 不影响 503 的判定，
  它只影响 body

---

## ② `GET /tools` → HTTP 200，count=7

```
· common.discovery.list_domains      [read]  列出可用的知识领域，含每种的检索形态与集合名
· common.discovery.health_check      [read]  检查向量库 / embedding / 重排服务与熔断器的当前状态
· common.ops.get_index_status        [read]  查询某集合的行数、索引类型与状态
· common.ops.reindex                 [write] 重建某领域的索引；默认只预演，需显式传 dry_run=false 才真的执行
· common.safety.get_rules            [read]  返回该领域的禁止断言、免责声明、计划变动上限与急症信号
· common.safety.check_input          [read]  检查用户输入是否需转介急症，或触及禁止断言
· common.safety.check_output         [read]  检查生成的回答是否含禁止断言；命中则整条拒答（不改写）
```

**这是静态镜像**，与工具注册表的一致性由
`tests/test_api.py::test_tools_mirror_matches_registry` 保证 ——
注册表加工具而镜像没同步，测试会红。

**为什么不运行时跨仓读**：注册表在另一个仓库，生产部署时两个服务可能在不同容器，
文件系统不共享。跨仓读只发生在测试，不发生在部署。

---

## ③ `POST /query` → HTTP 503

请求：

```json
{"question": "高血压和糖尿病该注意什么"}
```

响应：

```json
{"detail": "依赖不可用：MilvusException"}
```

**这条最关键** —— 它验证了两个判断：

**（1）503 而不是 500**

`MilvusException` **不继承** `ConnectionError`，所以按异常类型白名单判断会漏。
实现改用**探针归因**：失败时问一句「依赖现在活着吗？」

```
依赖 down ⇒ 503（这个失败是它造成的）
依赖 up   ⇒ 500（那是我自己的 bug）
```

调用方据此决定「等会儿重试」还是「报 issue」。

**（2）一个实现细节的代价**

失败路径会多一次探针调用（毫秒级）。**这个代价是刻意接受的** ——
失败路径本来就不该追求快，而正确归因比快重要。

---

## 验收对照

| # | 验收项 | 结果 |
|:--:|---|---|
| 1 | `uvicorn` 起得来 | ✅ 4 条路由注册（`/health` `/tools` `/query` `/query_multi`） |
| 2 | 3 端点各 1 次真实调用留痕 | ✅ 本文件 |
| 3 | ≥1 条 pytest 覆盖 `/health` 的 degraded 分支 | ✅ `test_health_degraded_when_pg_down` |
| 4 | 依赖不可用时 `/query` 返回 **503** 而非 500 | ✅ 见 ③ |

**测试读数**：`102 passed / 7 skipped / 0 failed`
（7 skipped 全部是记忆层实存测试 —— PostgreSQL 不可达时主动跳过，设计如此）

---

## 已知边界（不在本版范围内）

- **无鉴权/限流/多租户** —— 在反向代理或 MCP 层解决
- **`/health` 的 PG 探针只探端口**，不执行 `SELECT 1` —— 端口开着但 SQL 跑不起来会漏判。
  更严的探针需要连接密码与库名，健康检查不该引入这些依赖
- **`num_entities` 在 Milvus 不可用时为 null**，不是 0 —— 区分「查不到」和「确实是空的」
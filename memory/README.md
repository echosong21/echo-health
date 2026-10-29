# 记忆层

给问答加上"连续性"：同一个用户连问两次，第二次要知道第一次聊了什么。

---

## 三行边界（可以用自己的话复述）

1. **存什么** —— 只存**对话本身**（用户问了什么、答了什么）和**用户的稳定事实**（过敏、忌口、慢性病、目标）。**不存知识** —— 知识在向量库里，记忆不重复它。

2. **存多久** —— 情景记忆是**滑动窗口**（单会话最近 20 轮），语义事实**长期有效**直到被显式更新。两者的失效策略必须分开：裁剪对话时**不能**把用户的过敏史一起删掉。

3. **怎么取** —— 按 `session_id` 取最近 N 轮 + 按 `user_id` 取全部事实，渲染成一段文字插到 prompt 里。**记忆只做上下文注入，不参与检索召回**。

---

## 四类限制值

**只在一处定义**：`src/memory/base.py` 顶部常量块。任何实现都必须引用它们，不许散落到各处。

| 限制 | 值 | 理由 |
|---|--:|---|
| 单会话保留轮次 | **20**（40 条） | 超过 20 轮的上下文对 prompt 是噪音 |
| 单会话硬上限 | **200** 条 | 防止长会话把单会话数据撑爆 |
| 语义事实上限 | **30** 条 | 超过说明在收集不该收集的信息 |
| 单条 text 长度 | **500** 字符 | 防止把整篇文档塞进记忆 |

---

## 存储

**PostgreSQL**，两张表（`scripts/schema_memory.sql`）：

```
episodes  (id, session_id, user_id, turn, role, text, meta, ts)
          └─ 索引 (session_id, turn DESC) + UNIQUE(session_id, turn, role)

facts     (user_id, key, value, evidence, updated_at)
          └─ PK (user_id, key)，扁平 KV
```

**为什么 `facts` 用扁平 KV 而不是固定列**：事实类型随领域变化（健康领域有过敏，法律领域没有）。固定列每加一种就要 `ALTER TABLE`；扁平 KV 加一个 key 就完事。这是"未来不大改"的关键。

**为什么 `meta` 用 JSONB 而不是 TEXT**：只放可枚举元信息（命中形态、上下文条数），但需要能按条件查。**`meta` 里不允许放用户原文** —— 与 `src/observability/audit.py` 的 `args_summary` 同一条纪律。

---

## 接口

`src/memory/base.py` 的 `MemoryStore` 是 **Protocol**，只有两个方法：

```python
read(*, session_id, user_id, limit) -> {"episodes": [...], "facts": {...}}
write(*, session_id, role, text, user_id, meta) -> None
```

**刻意不做**：`search` / `summarize` / `forget`。
- 有 `search` 就把接口锁死成"必须具备语义检索能力"
- `forget` 要处理"删了但索引还在"，等真的需要（合规要求）再设计

**返回值是可扩展字典，不是 dataclass** —— 将来加一类记忆只需多一个 key，
老调用方用 `.get()` 读不到也不会崩。

---

## 故障语义

**记忆是增强项，不是关键路径。**

| 情况 | 行为 |
|---|---|
| PG 不可达（`read`） | 返回 `{"episodes": [], "facts": {}}`，主链路按无记忆继续 |
| 写入失败 | 静默 + 记 warning，不影响本次回答 |
| 事实达上限 | `upsert_fact` 返回 `False`，不静默丢弃 |

**理由**：记忆丢了只是少一点上下文，不值得让整次问答失败。
这与 MCP 侧 `resilience/` 的降级设计是同一条原则。

---

## ⚠️ 部署前必须改的

**当前连接配置用 `--auth=trust`（本地免密）。**

这意味着**本机任何进程都能无密码连 PG** —— 开发时方便，但**放到有公网入口的机器上就是零认证裸奔**。

部署前必须：
1. 改 `pg_hba.conf` 为 `scram-sha-256`
2. 设强密码，通过环境变量 `MEMORY_DSN` 注入（不写进代码）
3. PG 只监听 `127.0.0.1`，不对外暴露端口

---

## 本地启动

> ⚠️ **本仓不假设 PostgreSQL 的安装位置** —— 下面用 `$PG_HOME` 指代，
> 按你本机的实际安装目录设置（便携版 zip 解压后的顶层目录）。

```powershell
# 设置一次（当前会话有效；写进 $PROFILE 可持久化）
$env:PG_HOME = "<你的 PostgreSQL 安装目录>"

# 起 PG（便携版不装服务，每次重启要手动起）
& "$env:PG_HOME\bin\pg_ctl.exe" -D "$env:PG_HOME\data" -l "$env:PG_HOME\data\pg.log" start

# 建表（首次）
& "$env:PG_HOME\bin\psql.exe" -U postgres -d echo_health -P pager=off -f scripts/schema_memory.sql
```

> ⚠️ `-P pager=off` 必须加 —— 否则 Windows 下 psql 会调 `more` 并报
> `'more' 不是内部或外部命令`，输出被吞掉。

---

## 验证

```bash
python -c "from src.memory.base import MemoryStore; print(MemoryStore)"
python -m pytest tests/test_memory.py -q
```
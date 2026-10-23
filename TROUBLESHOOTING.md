# 排查记录

> 本文件记录**真实发生过的故障与排查过程**，包括走错的弯路。
> 设计取舍与方案对比在 `VALIDATION.md` / `DESIGN` 一类文档里，不混在此处。

---

## 1. CI 上 `--help` 崩溃：模块级导入配置

**现象**

新增的配置测试里有一项跑 `python scripts/build_index.py --help`，本地 39 个测试全过，
CI 上却有 3 个失败：

```
AssertionError: --help 失败（退出码 1）：Traceback (most recent call last):
  File ".../scripts/build_index.py", line 32, in <module>
```

**根因**

`build_index.py` 在**模块级**写着 `from config import settings`。
`config` 构造配置对象时会校验环境变量，缺少 `.env` 的环境（CI、全新 clone）在
**import 阶段**就抛异常 —— 于是连 `--help` 都跑不出来。

**修复**

改为 `get_settings()` 惰性加载：只有真正要走 embedding / Milvus 的代码路径才读取配置；
`--help` 在 argparse 阶段就退出，完全不碰配置。

**教训**

> **模块级副作用会让「只是想看一眼帮助」变成「必须准备完整运行环境」。**
>
> 一个 CLI 脚本的 `--help`、`--version` 这类纯元信息命令，不应该依赖业务配置。
> 判断标准很简单：**把模块 import 进来，会不会产生外部依赖？**（连接、读文件、读环境）
> 会，就应该推迟到真正使用时。

---

## 2. 改完懒加载 CI 仍然失败：真正原因是缺依赖

**现象**

第 1 条修完后重新 push，CI 还是红。此时错误已经变了，但因为没有第一时间去读日志，
我先入为主地继续沿第 1 条的思路排查（怀疑是环境变量），**猜了两轮都错**。

拿到完整日志后，真相一行就看清：

```
ModuleNotFoundError: No module named 'langchain_text_splitters'
  File ".../scripts/build_index.py", line 32, in <module>
    from langchain_text_splitters import RecursiveCharacterTextSplitter
```

**根因**

`requirements.txt` 里**从来没有** `langchain-text-splitters`。

为什么这个缺失能潜伏这么久？因为：

| 环节 | 是否 import `build_index.py` |
|---|---|
| `pytest`（原有 33 个测试）| ❌ 只测 `src/` 与 `tests/` |
| `compileall`（lint job）| ❌ 只做语法编译，不执行 import |
| 新加的配置测试 | ✅ **第一次真正启动这个脚本** |

**⇒ 一份缺失的依赖，在仓库里躺了很久，直到有人第一次真正运行那个入口才暴露。**

**修复**

`requirements.txt` 补上 `langchain-text-splitters==1.1.2`（与本机已装版本一致）。

**教训**

> **「本地能跑」和「CI 能跑」之间，差的是「首次 import 的时机」。**
>
> 一个模块如果长期没有被任何测试 import，它依赖什么就是完全未知的。
> ⇒ **每个可执行入口（脚本 / CLI / 服务）至少要被一个测试真正启动一次**，
> 哪怕只是 `--help`。这次加配置测试，价值恰好就在于此。

---

## 3. 一次误判：把「缺依赖」当成「环境变量被裁掉了」

**现象**

第 2 条那个阶段，我为了「模拟干净的 CI 环境」，在测试里写了：

```python
_MIN_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    "PYTHONIOENCODING": "utf-8",
}
subprocess.run([...], env=_MIN_ENV)
```

然后推断：Linux（CI）上没有 `SYSTEMROOT`、又没有 `HOME`，所以第三方库 import 失败。

**这个推断是错的。** 日志里没有任何与环境变量相关的报错，报的是缺依赖。
`_MIN_ENV` 根本不是原因（`PATH` 已经够 subprocess 启动 Python）。

**后来怎么发现的**

**去读了完整日志**。在此之前的两轮都是「看截图 + 猜测」。

**修复**

删掉 `_MIN_ENV`，让子进程继承父环境。
「测试不依赖 `.env`」这个需求改由 `test_settings_is_lazily_imported` 的**静态检查**保证
（读源码断言 `config` 不在模块级导入），不靠篡改环境实现。

**教训**

> **人为裁剪环境会制造假象**：把「缺依赖」伪装成「环境问题」，
> 把排查引到错误方向。而且裁剪过度（去掉 `HOME` / `TMPDIR`）本身就可能让
> 第三方库 import 失败，产出与真实 CI 无关的假失败。
>
> **更根本的一条**：
> **不看日志就推断，就是在猜。这一轮我连猜两次都错了，一读日志 5 分钟定位。**
> 截图上只有「红的」，日志里才有「为什么红」。

---

## 附：这三条的共同点

三条记录里，真正浪费时间的都不是技术难度，而是**顺序错了**：

| 错误顺序 | 正确顺序 |
|---|---|
| 看现象 → 提出假设 → 改代码 → 等 CI | 看现象 → **读完整日志** → 定位 → 改代码 → 验证 |

**⇒ 下次 CI 红的第一个动作，应该是「下载完整日志」，而不是「想一想可能是哪里」。**
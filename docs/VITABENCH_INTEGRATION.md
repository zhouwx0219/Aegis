# VitaBench 接入可行性评估与对象映射方案

> 调研日期 2026-06-16。目标：用**真实 VitaBench 环境**（任务/工具/状态）驱动 CAST，
> 但用**模拟 agent**（规则 winner 选择 + 对数正态延迟）代替真实 LLM，测真实负载上的并发收益。

## 0. 结论：可行性高，可在当前环境接入

已**实测**：
- `git clone` 成功（本 VM 能连 GitHub、能拉 VitaBench）。
- 环境层**独立于 LLM**：`Environment` 类提供 `get_tools()` / `use_tool(name, **kwargs)`（直接执行工具改状态）/ `check_db()` / `get_db_hash()`。
- 任务是**离线 JSON**：`data/vita/domains/{delivery,instore,ota,cross_domain}/tasks_en.json`，每域 100 个任务（英文版齐全），含初始状态。
- `DB.load(path)` 从 JSON 载入状态；`Task.environment` 自带初始 DB 状态。
- 依赖全为**纯 Python**（pydantic/deepdiff/loguru/pandas/litellm/redis 等，无 torch、无需编译）。

**只需绕过 `agent/` `user/` `evaluator/` 三个 LLM 模块**，用 `environment + db + tasks + 我们的模拟 agent` 即可。

## 1. VitaBench 架构（关键：环境层与 LLM 解耦）

```
src/vita/
├── environment/          # ★环境层（无需 LLM）
│   ├── environment.py    #   Environment: get_tools / use_tool / check_db / get_db_hash
│   ├── db.py             #   DB: load/dump(JSON) / get_hash / assign_order_id ; MergedDB 跨域
│   ├── toolkit.py        #   27 写工具 + 33 读工具的实现（改 db）
│   ├── tool.py / toolkit_schema.py
├── data_model/
│   ├── tasks.py          #   Task{id, domain, environment(初始DB状态), user_scenario, instructions, ...}
│   │                     #   状态实体 StoreBaseModel / ProductBaseModel(price, quantity) / Location（均 ThreadSafeBase）
│   └── simulation.py / thread_safe_base.py
├── agent/ user/ evaluator/ orchestrator/   # ← 仅这些用 LLM，可绕过
└── domains/{delivery,instore,ota}/         # 三域的工具与数据装配
data/vita/domains/*/tasks_en.json           # ← 离线任务（各100），含初始状态
```
血统：改编自 tau2-bench（成熟的"环境=DB+工具、可独立 step"架构）。

## 2. 接入架构（绕过 LLM）

```
tasks_en.json → Task → Environment(db = Task.environment)
                              │  get_tools / use_tool（执行工具→改 db）
        ┌─────────────────────▼──────────────────────┐
        │  我们的模拟 agent：枚举异构候选 + 规则 winner  │  ← 代替 agent-llm
        │  + 对数正态思考延迟（复用 timed/explore 模型） │
        └─────────────────────┬──────────────────────┘
        │  适配器：use_tool 前后 deepdiff(db) → 读写集 + 写意图
        ┌─────────────────────▼──────────────────────┐
        │  CAST / OCC / 2PL / MVCC 对象事务并发控制层    │  ← 我们的核心
        └──────────────────────────────────────────────┘
```

## 3. 对象映射方案（工具调用 → 统一对象事务）

核心思路：`use_tool` 执行前后对 db 做 **deepdiff**（VitaBench 自带该依赖），提取"读了哪些对象、写了哪些对象成什么"，并推断写意图：

| db 变化 | 写意图 | 并发类 |
|---|---|---|
| `product.quantity` 减少（下单扣库存） | DELTA | 可合并（commutative） |
| 列表/文本追加（购物车、备注、行程） | APPEND | 可合并 |
| 状态字段 pending→confirmed（确认订单/预订） | CAS | 条件 |
| 覆盖字段 / 新建订单（assign_order_id） | OVERWRITE | strict |

- read set = 工具读取对象的版本（包装 db 读取，或用执行前快照）。
- 实现两条路：(a) **黑盒 diff** 自动推断（通用、快）；(b) **白盒** 读 `toolkit.py` 逐写工具标注（准）。建议混合：diff 自动 + 对核心写工具校正。

## 4. 模拟 agent（代替真实 LLM）

- **异构多候选**：对一个用户请求枚举若干可行方案（不同商家/商品/时段）作为候选 → 直接喂给已验证的 `explore` 路径。
- **winner 选择**：规则打分（距离/价格/匹配度）。
- **思考延迟**：对数正态（复用 `timed_experiment` / `explore_experiment` 的模型）。

## 5. 并发构造

VitaBench 单任务本是顺序的；我们在**同一个 db 实例**上让多个任务（或同任务多用户）交错 `use_tool`，制造对热点实体（同商家 `product.quantity`、同时段名额）的真实争用。VitaBench 的 `ThreadSafeBase` 只是底层结构锁，**没有事务级并发控制**——正是 CAST 对象事务层的落点，不冲突。

## 6. 与路 C 的衔接

真实工具天然给出"不同读写操作"的分类（扣库存=可交换、加购=可追加、确认=条件、覆盖=strict），正好支撑**路 C**：
- 验证层：对可交换工具放宽（不互判冲突 → 真正减少冲突、提并发吞吐，回应最初设想）；
- 冲突解决层：成本不对称（多候选 reselect / merge 处理 strict 真冲突）。

## 7. 分步接入计划

- **P1 Spike（验证端到端无 LLM）**：`pip install -e .`（纯 Python，约 2–3 分钟）→ `import Environment` → 载入 `tasks_en.json` 一个任务 → `Environment(db=task.environment)` → 调一个写工具 `use_tool(...)` → `deepdiff` 打印 db 变化。坐实"无 LLM 也能执行 VitaBench 工具改状态"。
- **P2 适配器**：tool-call → 对象事务（diff→读写集+意图），先覆盖 delivery 域核心写工具（下单/扣库存/加购/确认）。
- **P3 并发驱动 + 指标**：模拟 agent 驱动多并发任务，跑 OCC/CAST/2PL/MVCC，复用现有 成本/延迟/吞吐 指标，并报告**真实可合并占比与冲突分布**。
- **P4（可选）**：cross_domain 更高争用 / 读密集场景区分 MVCC。

## 8. 风险与缓解

- 依赖装得慢（纯 Python 无编译）：耐心装，或精简到 import 环境所需子集。已实测 clone 成功、包可达。
- 工具意图推断需校正：黑盒 diff 先行，白盒读 `toolkit.py`（27 写工具）校正关键工具。
- 中文数据：用 `tasks_en.json`（英文版齐全）。
- 我们**不使用** VitaBench 的 evaluator-llm（不评 rubric 任务成功率）；只关心并发控制指标，必要时用 `check_db` 验证状态正确性。

## 9. 工程量

P1 spike 小（约 0.5 天）；P2 适配器中（1–2 天，取决于覆盖工具数）；P3 中。整体可控，且全部可在本环境内完成。

## 10. 下一步建议

先做 **P1 spike**：跑通"无 LLM 执行一个 VitaBench 写工具改 db + diff 出对象变化"，端到端坐实后再写适配器与并发驱动。

## 11. P1 spike 已跑通（2026-06-16，实测）

无 LLM 端到端验证成功：`agent/integrations/vitabench_spike.py`（配 `setup_vitabench.sh` 一次性 clone+install）。
- 用真实 delivery 任务（`tasks_en.json`，100 任务）的初始状态构造 `Environment`；
- 调真实写工具 `create_delivery_order` → 成功创建订单（**全程零 LLM 调用**）；
- `deepdiff` 捕获 db 对象级写入：`orders[新id]` 新增 → 映射为 CAST 写意图 **CREATE/OVERWRITE**。

结论：坐实"模拟 agent + 真实 VitaBench 环境"可行，对象映射机制成立。
下一步 **P2**：扩展 diff→意图映射覆盖 DELTA（扣 `product.quantity`）/APPEND/CAS，并接入并发驱动跑 OCC/CAST/2PL。
（注：VitaBench 装在 `/tmp/vb`，VM 会话间会重置，重跑前先 `bash agent/integrations/setup_vitabench.sh`。）


## 12. P2 已跑通（2026-06-16，真实 OTA 负载三方对比）

- 写模式实测：delivery=私有订单(CREATE/CAS/OVERWRITE,无共享扣减)；**OTA `create_*_order` 扣共享座位库存(DELTA)**(use_tool 实测 quantity 35→33)+ 抢热门航班=真实争用。
- OTA 座位争用三方对比(`agent/integrations/vitabench_ota_concurrency.py` → `results/vitabench_ota.png`,batch=16)：浪费/任务 OCC 0.31 / CAST 0.003；延迟 1.31 / 1.00；吞吐 2.67 / 15.2；OCC 200 次重跑 ↔ CAST 200 次 DELTA 合并。
- 结论：真实负载上 CAST 的语义合并收益成立（共享有限资源场景）。下一步 P3：路C 验证层语义放宽 + 读密集/写偏斜区分 MVCC + 跨域更高争用。

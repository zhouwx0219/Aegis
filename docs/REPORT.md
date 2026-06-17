# CAST 阶段性汇报
## 面向 LLM Agent 探索式任务的成本不对称投机事务（Cost-Asymmetric Speculative Transactions）

> 目标会场：SIGMOD / VLDB / ICDE（论文 A）。本汇报覆盖：研究背景 → 动机与挑战 → 创新点 → 系统设计 → 实验结果。
> 配套：设计文档 `../docs/cost_asymmetric_speculative_transactions_design.md`、定位分析 `../docs/advisor_feedback_data_agent_system_analysis.md`；实现与实验在本目录 `cast-das/`。

---

## 1. 研究背景

近一年，数据管理社区涌现出一批 **data agent system / 声明式 AI 分析**系统（MIT 的 **Palimpzest**(CIDR'25)、**Nirvana/Beyond Relational**(PACMMOD'25)、LOTUS、DocETL 等）。它们的共同思路是把数据库的查询处理与优化器范式搬到 LLM 驱动的数据处理上：用户声明高层语义查询，系统编排"执行算子 + LLM 算子"并优化执行成本。

但这一整脉系统有一个共同特征：**它们全是只读分析（AI-OLAP）**——做数据提取、过滤、聚合、洞察，**不涉及事务、并发写入与一致性**。与此同时，并发控制那一脉（Percolator、Doppel、DBx1000、ATCC 等）又全部面向传统结构化数据。

> **空白**：没有人把"LLM 语义算子 + 多模态对象 + 事务性写入/并发控制"统一在一个 data agent system 里。
> **定位**：如果 Palimpzest 是"AI 时代的 OLAP"，本工作要做"**AI 时代的 OLTP**"——在 data agent system 内部做统一对象事务与并发控制。

---

## 2. 动机与挑战

### 2.1 Agent 任务与传统事务的本质差异：探索成本被反转

LLM agent 任务是**探索式**的：一个任务常生成多个候选方案（分支），最终选一个 winner 提交。而它与传统事务在"探索/重试的成本"上相差四到五个数量级：

| 维度 | 传统事务 | LLM agent 任务 |
|---|---|---|
| 单候选执行成本 | 微秒级 CPU | 秒级 LLM 推理 + 工具 + token |
| abort/重试成本 | 重放读写集（廉价） | **重新生成候选 = 再跑一次昂贵 LLM** |
| 冗余多候选 | 经典 SCC 敢开多个 shadow | 多开一个候选 = 多烧一份钱 |
| 系统目标 | 最大化吞吐 / 满足截止时间 | **最小化浪费的 LLM 算力** |

### 2.2 经典并发控制的假设在 agent 场景失效

- **strict OCC** 的乐观假设是"abort 便宜，所以冲突了大不了重来"。在 agent 场景，abort = 重跑 LLM，乐观假设崩塌。
- **SCC（Speculative Concurrency Control, Bestavros 等）** 的核心假设是"冗余探索廉价，所以多开 shadow 赌序列化顺序"。在 agent 场景，每个 shadow 都是一次昂贵生成，假设同样崩塌。

### 2.3 由此导出的新问题

> 在"候选生成昂贵 + 候选写入语义可合并 + 读写冲突仅在提交时暴露"三重约束下，如何决定候选的生成与提交，使**为提交一个合格 winner 所浪费的 LLM 算力**最小？

这不是又一个并发控制机制，而是一个**被反转的成本结构所定义的新问题**。

---

## 3. 创新点

### 3.1 主张：Cost-Asymmetric Speculative Transactions（CAST）

把 SCC 的成本模型反转，给出面向 agent 探索式任务的投机事务模型与提交协议。形式化优化目标（n 个任务，单候选生成成本 c_gen，语义合并成本 c_merge ≪ c_gen）：

```
min  E[ (生成候选数 - n)·c_gen  +  n_regen·c_gen  +  n_merge·c_merge ]
     └ loser 候选 ┘            └ 昂贵重跑 ┘        └ 廉价合并 ┘
s.t. 每个任务提交一个达标 winner
```

### 3.2 事务语义扩大化：统一对象事务

事务边界不再是"调数据库的 BEGIN/COMMIT"，而是 agent 内部对**任意类型对象**操作的统一事务。第一版聚焦 **row + text + counter** 三类对象，统一为"对对象的写意图"管理；底层存储只提供版本化 KV 原语，不理解对象类型/事务/并发——语义全部上移。

### 3.3 统一洞察：语义可合并性在 agent 场景被重新定价

传统语义并发（Escrow/CRDT/Coordination-Avoidance）用可交换性避免 abort，省下的是**廉价的 CPU 重试**；在 agent 场景，同一机制省下的是**昂贵的 LLM 重跑**——价值放大四到五个数量级。提交协议据此把昂贵的 `regenerate` 压到最后，优先 `merge`（语义 rebase）与 `reselect`（改提交已有候选）。

### 3.4 与已有工作的区分（related work 命门）

| 簇 | 代表 | 与 CAST 的区别 |
|---|---|---|
| 投机并发 | SCC / PSCC | 假设候选廉价；CAST 反转成本假设，目标改为省 LLM 算力 |
| agentic 并发 | ATCC | 单事务 lock 调度，无多候选探索与语义合并 |
| 投机 agent（serving） | Speculative Actions / Sherlock / SPAgent | control-flow 延迟隐藏，不管事务一致性 |
| LLM agent 事务 | SagaLLM / STORM / Atomix | 多 agent 协调；CAST 是单任务多候选投机 |
| 语义/可交换并发 | Escrow / Walter / CRDT | 在 agent 成本下重新定价：省的是昂贵 LLM 重跑 |
| client 事务 over KV | Percolator / Warp | 复用存储边界，上层是投机多候选事务而非固定读写集 2PC |

---

## 4. 系统设计

### 4.1 混合栈架构

```
Python 层（agent/）   算子、调度、workload、实验
  • MockOperator / CandidateScheduler / 受控负载 / VitaBench-derived 负载
        │ pybind11（进程内）
C++ 核（core/）       事务/并发/提交（性能与正确性）
  • 统一对象 + 版本化存储（纯 KV 边界）
  • 写意图与并发类（PolicyDispatcher：strict / commutative_rebase / conditional_rebase）
  • 成本模型 + 成本不对称提交协议（CAST 核心）
        │ 仅 Get/PutIfVersion/BatchPutIfVersion/DeleteIfVersion
Versioned KV（底层）   只认 key→(value, version)，不理解类型/事务/并发
```

### 4.2 成本不对称提交协议（核心）

```
CommitTask(candidates, strategy):
  winner = 质量最高候选
  TryCommit(winner):                       # 版本未变直接写；版本变了——
     若 strategy=CAST 且写可合并: 语义 rebase 合并 (merge, 花 c_merge)
     否则(strict 写 / OCC):       记为冲突
  若提交成功: return
  若 CAST: 依次 reselect 其他已生成候选 (零额外生成成本)
  最后手段: regenerate (重读最新基线重跑, 花 c_gen) —— 唯一昂贵路径
```

- **OCC = baseline**：任何版本冲突 → 直接 regenerate。
- **CAST = ours**：merge → reselect → regenerate，把昂贵重跑压到最后。

### 4.3 底层存储边界（论文卖点，已钉死）

`VersionedObjectStore` 只提供 `Get / GetVersion / PutIfVersion / DeleteIfVersion / BatchPutIfVersion`，**不记录对象类型、不理解事务/分支/并发**。对象类型随写意图在上层流转，符合"语义全部上移、存储只提供版本化 KV 原语"的定位。

### 4.4 复用与新增

复用上层 Data-Agent-System 的：版本化 KV 边界、五类写意图、`PolicyDispatcher` 的并发类与语义 rebase。新增：成本模型、成本不对称提交协议、CandidateScheduler、Python 算子层与 pybind11 桥接。

---

## 5. 实验结果

### 5.1 实现状态

`cast-das/` 阶段 0/1 骨架完整，全链路可复现（`bash build.sh` → demo / sweep / vitabench）。OCC、CAST 真跑 C++ 核；SCC-kS 为解析成本模型（已与真跑 OCC 对齐验证，见 5.3）。

### 5.2 端到端 demo（机制验证）

- **可合并冲突**（DELTA 库存 + APPEND 备注，并发抢先改）：OCC 必须 regenerate（浪费 1.0），CAST 用语义合并（浪费 0.02）——**省 98%**。
- **strict 冲突**（OVERWRITE，并发覆盖）：两者均 regenerate，CAST 无优势——**诚实边界**：优势只来自语义可合并的写。

### 5.3 受控负载三方对比（OCC vs SCC-kS vs CAST）

见 `agent/experiments/results/sweeps3.png`。人均浪费（单位 c_gen/任务）：

- **并发度**：OCC 随并发上升（batch=16 时 0.90），CAST 贴地（0.02）；**SCC-2S 最差**（1.0→1.78，每任务先付 1 份冗余 shadow）。
- **可合并写占比**：p=0（全 strict）三者相近（≈0.81）；p≥0.75 时 CAST 降到 0.02。
- **成本不对称**：c_merge≪c_gen 时 CAST 省 >97%；**交叉点 c_merge≈0.4**，之后 CAST 反而更差——精确划出适用边界。
- **强结论**：SCC-best 在 k=1..8 的最优永远是 **k\*=1（即退化成 OCC）**——在 agent 成本下投机 shadow 永不划算。
- **对齐验证（alignment）**：OCC 实测 regenerate 数 == 结构性冲突任务数，且 OCC 实测浪费 == SCC-1S 解析值，**全部 PASS**，证明解析 SCC 与真跑核心一致。

### 5.4 VitaBench-derived 负载（贴近真实领域）

见 `agent/experiments/results/vitabench.png`。基于 VitaBench（外卖/到店/旅游）的领域对象（库存→DELTA、购物车/行程→APPEND、订单→CAS）构造并发负载，**可合并写占比由领域语义自然产生**：

- 自然可合并写占比 = **79.8%**（CAS 占 20.2%），冲突率 44.1%（batch=8，320 任务）。
- 人均浪费：OCC=0.441，SCC-2S=1.137，SCC-best=0.441(k\*=1)，**CAST=0.011**。
- **CAST 比 OCC 省 97.5%**（154 次语义合并，仅 2 次重跑）。

> 这证明 CAST 的收益不是 synthetic 调参的产物：在贴近真实 agent 任务领域的负载上，自然产生约 80% 的可合并写，CAST 把绝大多数昂贵重跑替换为廉价合并。

---

### 5.5 多维对比：成本 × 延迟 × 吞吐（含 2PL / MVCC 基线）

引入执行/时间模型（t_gen=候选生成、t_merge=语义合并）后，在 VitaBench-derived 负载上对比四种策略（见 `agent/experiments/results/timed.png`）：

| 指标（batch=16） | OCC | MVCC-SI | 2PL | CAST |
|---|---|---|---|---|
| 浪费算力 / 任务 | 0.66 | 0.66 | 0.00 | 0.02 |
| 平均延迟（t_gen 单位） | 1.66 | 1.66 | 2.38 | 1.02 |
| 吞吐（提交 / 墙钟） | 1.38 | 1.38 | 3.02 | 12.0 |

- **CAST 三维全赢**：成本≈0（像 2PL 不重跑）、延迟最低（1.02）、吞吐最高（12）——它用语义合并同时避免了"重算"与"串行化"。
- **2PL**：不重跑（浪费=0），但悲观锁把争用串行化 → 延迟最高、吞吐随并发仅升到 3。它用并发换零浪费。
- **OCC / MVCC**：重跑导致成本高、吞吐随并发不升反降；MVCC-SI 对写密集负载等价 OCC（读不阻塞的优势需读密集/写偏斜负载，留作后续）。
- 执行/时间模型与 2PL、MVCC 的解析语义在 `agent/experiments/timed_experiment.py` 顶部文档化；OCC/CAST 的成本与动作来自真跑 C++ 核。

### 5.6 探索式多候选的独立收益（纯 strict 场景，与语义合并正交）

为把"探索式多候选"卖点与语义合并彻底分离，构造**纯 strict 独占资源负载**（CAS 占用，语义合并完全用不上，全程 `n_merge=0`），并引入对数正态 LLM 思考延迟。每任务生成 k=4 个**异构**候选（占不同资源），winner=质量最高；winner 资源被并发占走时，CAST 复用本轮已生成的其他候选**免费 reselect**，OCC 只能**重新探索**（重跑 LLM）。见 `agent/experiments/results/explore.png`（batch=16）：

| 指标 | OCC | CAST |
|---|---|---|
| 浪费 / 任务 | 1.34 | 0.03 |
| 平均延迟（t_gen） | 2.34 | 1.77 |
| 吞吐 | 2.55 | 4.83 |
| reselect 次数 | 0 | 136 |

- 全程 `n_merge=0` → CAST 的收益**纯粹**来自异构多候选 reselect，与语义合并正交。
- CAST 的 reselect 次数随并发增长（10→32→76→136），**首次实证"探索式多候选 winner 选择"机制**（此前 k=1 时 reselect 恒为 0）。
- 这与 §5.3–5.5 的语义合并收益是**两条独立的收益来源**：可合并写 → 语义合并；strict 写 → 异构多候选 reselect。
- 实现中修正了一个 CAS 正确性 bug（regenerate 强行对齐版本会绕过 CAS 条件），保证独占资源语义正确。

### 5.7 真实 VitaBench 负载验证（P2：模拟 agent + 真实环境）

接入真实 VitaBench 环境（绕过 agent/user/evaluator 三个 LLM 模块，用模拟 agent 驱动）。两点关键实证：

**(1) 真实写模式（决定 CAST 在哪有用）**：
- delivery 域：写 = 私有订单 CREATE / 状态 CAS / 改单 OVERWRITE，**无共享资源扣减、天然争用低**（订单私有）→ 语义合并无从发挥。
- OTA 域：`create_*_order` **扣减共享座位/房间库存（DELTA）**——`use_tool` 实测 `flight.products[seat].quantity 35→33` + 多人抢热门航班 = 真实争用。

**(2) OTA 座位争用三方对比**（真实座位库存负载，`results/vitabench_ota.png`，batch=16）：

| 指标 | OCC | CAST |
|---|---|---|
| 浪费/任务 | 0.31 | 0.003 |
| 平均延迟（t_gen） | 1.31 | 1.00 |
| 吞吐 | 2.67 | 15.2 |
| 冲突处理 | 200 次重跑 | 200 次合并 |

CAST 把订票重跑全部替换为 DELTA 合并：浪费算力降 ~99%、吞吐 ~5.7×。负载来自真实 VitaBench 环境，并发控制在 cast_core（`agent/integrations/vitabench_ota_concurrency.py`）。

**诚实边界**：可合并收益集中在**共享有限资源**（OTA 座位/房间）；私有订单为主的域（delivery），CAST 收益主要来自多候选 reselect（§5.6）而非合并。这本身是有价值的实证结论：agent 负载里 CAST 的收益面取决于"共享可变资源 + 可交换写"的比例。

## 6. 当前限制与下一步

诚实声明：(a) SCC-kS 为解析成本模型（order-shadow 与 agent 候选语义不完全匹配，强行真跑反而引入争议）；(b) VitaBench 为领域结构负载模型，**未接入真实 LLM-in-the-loop**；(c) CandidateScheduler 为 k=1 规则版。

下一步优先级：
1. **真实 LLM-in-the-loop VitaBench**：用真实 agent 产生候选与冲突，验证冲突分布与可合并占比。
2. **CandidateScheduler 自适应 k**：strict 写多时按"边际收益 < c_gen 即停"增开候选，用 reselect 兜底。
3. **SCC order-shadow 真实现** 与 **2PL 对照**（引入吞吐/延迟维度）。
4. **隔离级别形式化**：统一对象事务在可交换写放宽下的正确性保证。

---

## 7. 一句话总结

CAST 把一个看似"agent 工程问题"重述为**数据库并发控制问题**：当探索的成本被反转，投机事务的最优策略也随之反转——从"多算几个赌得快"变成"用便宜的语义合并省下昂贵的重算"。受控负载与 VitaBench-derived 负载上的实验一致显示：CAST 比 strict OCC 省 ~97% 浪费算力，并对最强对手 SCC-kS 形成正面碾压，同时给出诚实的适用边界。

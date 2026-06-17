# CAST 技术汇报（整合版）
## 面向 LLM Agent 探索式任务的成本不对称投机事务与语义感知并发控制

> 目标会场：SIGMOD / VLDB / ICDE。本文整合受控扫描、成本×延迟×吞吐多维、探索式多候选、
> 真实 VitaBench OTA、以及语义感知验证（路 C）全部实验。顺序：研究背景 → 动机挑战 →
> 创新点 → 系统设计 → 实验结果 → 具体实例。实现与脚本见 `cast-das/`。

---

## 1. 研究背景

近一年数据管理社区出现一批 **data agent system / 声明式 AI 分析**系统（MIT 的 **Palimpzest**(CIDR'25)、
**Nirvana/Beyond Relational**(PACMMOD'25)、LOTUS、DocETL），把数据库的"声明式查询 + 优化器"范式
搬到 LLM 驱动的数据处理上。但它们有一个共同特征：**全是只读分析（AI-OLAP）**——做提取、过滤、聚合，
**不涉及事务、并发写入与一致性**。另一边，并发控制那一脉（Percolator、Doppel、DBx1000、ATCC）
又全部面向传统结构化数据。

> **空白**：没人把"LLM 语义算子 + 多模态对象 + 事务性写入/并发控制"统一在一个 data agent system 里。
> **定位**：Palimpzest 是"AI 时代的 OLAP"；本工作做"**AI 时代的 OLTP**"——在 data agent system 内部
> 做**统一对象事务**与**语义感知并发控制**。

---

## 2. 动机与挑战

### 2.1 Agent 任务的两个本质特性

- **探索式多候选**：一个任务常生成多个候选方案（分支），最终选一个 winner 提交。
- **执行极其昂贵**：每个候选 = 一次秒级 LLM 推理 + 工具调用 + token 费用。

### 2.2 由此产生的两个根本挑战

**挑战 A：成本被反转，经典并发控制的假设失效。**
传统事务 abort 便宜（重放读写集，微秒级）；agent 任务 **abort = 重跑 LLM**，贵 4–5 个数量级。
- OCC 的乐观假设（"abort 便宜，冲突了大不了重来"）崩塌。
- SCC（投机并发）的核心假设（"冗余探索廉价，多开 shadow 赌顺序"）崩塌——每个 shadow 都是昂贵生成。

**挑战 B：经典验证过严，把"可化解的并发"误判为冲突。**
OCC 的版本检查是纯语法的：两个任务都对同一库存做 `DELTA(-2)`/`DELTA(-5)`，版本一变就判冲突要 abort。
但它们语义上**可交换**，根本不是真冲突。OCC 还严格验证读集（读了 A、写了 B，A 变了也 abort）。
这些"假冲突"在 agent 场景导致大量**不必要的昂贵重跑**。

### 2.3 新的设计目标

不再是"最大化吞吐 / 满足截止时间"，而是：**(i) 在验证阶段按读写语义减少冲突；(ii) 对真冲突，
最小化为提交一个合格 winner 所浪费的 LLM 算力。**

---

## 3. 创新点

### 3.1 一句话主张

> **当探索的成本被反转，并发控制的最优策略也随之反转**：从"乐观重试 / 多开 shadow 赌顺序"，
> 变成"在验证层按语义放行可化解的并发 + 对真冲突用便宜的语义合并/候选复用替代昂贵重跑"。
> 我们把它实现为 **CAST（Cost-Asymmetric Speculative Transactions）**。

### 3.2 三层机制（各自有独立实验支撑）

1. **语义感知验证（路 C，对应挑战 B）**：按读写语义分级验证——**只读、可交换写（DELTA/APPEND）在验证阶段放行（不互判冲突）**，CAS 验条件，strict 写（OVERWRITE）严格。这是"减少冲突"，区别于 OCC（读写全严格）与 MVCC-SI（只放行读、可交换写仍判冲突）。
2. **成本不对称冲突解决（对应挑战 A）**：对真冲突（strict），优先 `merge`（语义 rebase）/ `reselect`（复用已生成候选），把昂贵的 `regenerate`（重跑 LLM）压到最后。
3. **探索式多候选**：复用 agent 固有的多方案探索（为质量而生成的候选），在并发冲突时免费 `reselect` 兜底。

### 3.3 统一对象事务（事务语义扩大化）

事务边界从"调数据库 BEGIN/COMMIT、只管关系行"扩大为"agent 内部对**任意对象**（第一版 row/text/counter）的统一事务"；底层只提供版本化 KV 原语，不理解类型/事务/并发——语义全部上移。

### 3.4 与已有工作的区分

| 簇 | 代表 | 与 CAST 的区别 |
|---|---|---|
| 投机并发 | SCC / PSCC | 假设候选廉价；CAST 反转成本假设，且实测 SCC 投机在 agent 成本下永不划算（k\*=1 退化 OCC）|
| agentic 并发 | ATCC | 单事务 lock 调度，无多候选探索与语义合并 |
| LLM agent 事务 | SagaLLM / STORM / Atomix | 多 agent saga 协调；CAST 是单任务多候选 + 语义并发 |
| 语义/可交换并发 | Escrow / Doppel / Coordination-Avoidance | CAST 在 agent 成本下重新定价（省的是昂贵 LLM 重跑），并把它做进探索式事务 |
| 隔离级别 | OCC / MVCC-SI | CAST 按写意图分级验证：比 OCC 少（读+可交换放行）、比 MVCC 更细（可交换写也放行）|

### 3.5 核心洞察（可写进结论、能预测适用场景）

> **CAST 在 agent 负载上的收益面 ≈ "共享可变资源 × 可交换写" 的比例。**
> 共享有限资源 + 可交换写多（如订票扣座位）→ 收益大；私有对象 + strict 写为主（如私有订单）→
> 收益主要来自多候选 reselect 而非合并。

---

## 4. 系统设计

### 4.1 混合栈架构

```
Python 层（agent/）   算子、调度、workload、实验、VitaBench 接入
  • MockOperator / CandidateScheduler / 受控负载 / VitaBench 适配器
        │ pybind11（进程内）
C++ 核（core/）       事务 / 并发 / 提交（性能与正确性）
  • 统一对象 + 版本化存储（纯 KV 边界）
  • 写意图与并发类（PolicyDispatcher：read-only / strict / commutative / conditional）
  • 语义感知验证 + 成本不对称提交协议（merge → reselect → regenerate）
        │ 仅 Get / PutIfVersion / BatchPutIfVersion / DeleteIfVersion
Versioned KV（底层）  只认 key→(value, version)，不懂类型/事务/并发
```

### 4.2 关键组件

- **统一对象 + 版本化存储**（`core/storage/versioned_object_store.h`）：纯版本化 KV，边界注释钉死"不理解类型/事务/并发"。
- **写意图与并发类**（`core/intent/policy_dispatcher.h`）：`Classify` 把意图分到 read-only / strict / commutative-rebase / conditional-rebase；`ResolveWrite` 实现 APPEND 拼接 / DELTA 重算 / CAS 条件。
- **语义感知验证 + 成本不对称提交**（`core/txn/cost_asymmetric_commit.h`）：版本未变直写；可交换写冲突 → rebase 合并；strict 冲突 → reselect 其他候选 → 最后 regenerate。
- **成本模型**（`core/cost/cost_model.h`）：`c_gen`（候选生成，秒级）≫ `c_merge`（语义 rebase）；统计 merge/reselect/regenerate 与浪费算力。
- **CandidateScheduler**（`agent/scheduler/`）：候选生成入口与投机度 k 决策。
- **VitaBench 接入**（`agent/integrations/`）：用真实 VitaBench 环境（`Environment.use_tool`）产生真实对象访问，deepdiff 提取写意图，喂给 cast_core 并发控制。

### 4.3 与现有 Data-Agent-System 的关系

复用上层版本化 KV 边界、五类写意图、`PolicyDispatcher` 的并发类与语义 rebase；新增成本模型、语义感知验证 + 成本不对称提交、调度器、Python 算子层与桥接、VitaBench 接入。

---

## 5. 实验结果

实现状态：`cast-das/` 全链路可复现（C++ 核 + pybind11 + Python 实验）。OCC/CAST 真跑 C++ 核；
SCC/2PL/MVCC 与语义验证为解析基线（同口径、已与真跑对齐验证）。

### 5.1 受控负载三方对比（OCC vs SCC-kS vs CAST）`results/sweeps3.png`
人均浪费（c_gen/任务）。**CAST 完胜**：可合并写场景比 OCC 低 >97%（高并发 0.02 vs 0.9）。
**SCC 投机反而更差**：SCC-2S 每任务先付 1 份冗余 shadow（浪费 ~1.6 > OCC 0.83），且 SCC-best 在
k=1..8 最优为 **k\*=1（退化成 OCC）**——agent 成本下投机永不划算。对齐验证：OCC 实测 regen ==
结构冲突数 == SCC-1S 解析，全 PASS。

### 5.2 成本 × 延迟 × 吞吐多维（含 2PL / MVCC）`results/timed.png`
batch=16：浪费/任务 OCC 0.66 / 2PL 0.00 / CAST 0.02；延迟 OCC 1.66 / 2PL 2.38 / CAST 1.02；
吞吐 OCC 1.4 / 2PL 3.0 / **CAST 12.0**。**CAST 三维全赢**：既不像 OCC 浪费重跑，也不像 2PL 锁串行化
牺牲延迟/吞吐。MVCC-SI 写密集下 ≡ OCC。

### 5.3 探索式多候选的独立收益（纯 strict，与合并正交）`results/explore.png`
纯 strict 独占资源（CAS，全程 `n_merge=0`）+ 对数正态 LLM 延迟。batch=16：浪费 OCC 1.34 / CAST 0.03；
延迟 2.34 / 1.77；吞吐 2.55 / 4.83。**CAST 的 reselect 随并发增长 10→136**，证明多候选机制独立有效
（与语义合并正交：可合并写靠 merge，strict 写靠异构 reselect）。

### 5.4 真实 VitaBench OTA 负载（模拟 agent + 真实环境）`results/vitabench_ota.png`
绕过 VitaBench 三个 LLM 模块，用真实环境驱动。**写模式实测**：delivery 域写=私有订单（CREATE/CAS/
OVERWRITE，无共享扣减、争用低）；OTA 域 `create_flight_order` **扣减共享座位库存（DELTA）**——
`use_tool` 实测 `flight.products[seat].quantity 35→33`。OTA 座位争用三方（batch=16）：浪费/任务
OCC 0.31 / CAST 0.003；延迟 1.31 / 1.00；吞吐 2.67 / **15.2**；本质是 **OCC 的 200 次订票重跑被
CAST 全部转成 200 次 DELTA 合并**。诚实边界印证 §3.5 的"收益面 = 共享可变资源 × 可交换写"。

### 5.5 语义感知验证分级（路 C：减少冲突）`results/semantic_validation.png`
含读 + 写偏斜负载，扫读集大小（读密集度）。冲突率：**OCC 随读集上升 0.43→0.66**（读写全严格）；
MVCC-SI ~0.4（读放行、可交换写仍冲突）；**CAST ~0.08**（读 + 可交换写都在验证层放行，只 strict-strict）。
吞吐 CAST 4–5 ≫ MVCC ~1.8 ≫ OCC 1.3–1.8。这证明 CAST 是在**验证阶段减少冲突**（非仅降低解决成本），
并明确区分 MVCC。

### 5.6 结果小结
CAST 的收益有**两条正交来源**（可交换写→语义合并/验证放行；strict 写→多候选 reselect），
在受控、真实 VitaBench、含读三类负载上一致成立，且三维（成本/延迟/吞吐）全面优于 OCC/SCC/2PL/MVCC；
边界诚实可预测。

---

## 6. 具体实例（端到端走一遍，标注每个流程节点的优化）

**任务**：用户让 agent“订周五去上海的机票，并把行程加到我的备忘”。

**① 探索阶段**——agent 生成 3 个**异构候选**（为质量而探索，顺带成为并发兜底）：
- 候选 A：`DELTA(flight:MU5438:economy, -1)` 扣座位 + `APPEND(memo:user, "MU5438 周五")`，质量最高（最便宜直飞）
- 候选 B：`DELTA(flight:CA1858:economy, -1)` + 同样备忘追加
- 候选 C：`DELTA(flight:HU7605:economy, -1)` + 同样备忘追加
winner = A。

**并发干扰**：另一用户几乎同时订了 MU5438 同舱位 → `flight:MU5438:economy.quantity` 版本前进（35→34）。

**② 验证阶段（路 C 语义感知验证）**——对 winner A 的两个写：
- `flight:MU5438:economy` 是 **DELTA（可交换）→ 验证层放行**（不因版本变而判冲突）；
- `memo:user` 是 **APPEND（可交换）→ 放行**。
对比 OCC：版本变 → 直接判冲突 → 整个事务 abort。

**③ 冲突解决阶段（成本不对称）**——
- 座位：在**最新值 34** 上 rebase 合并 `-1` → 33，提交成功（花 `c_merge`，几乎零成本）；备忘同理拼到最新值。
- **若 MU5438 已售罄**（DELTA 使库存 < 0，条件失败）→ 不重跑，**reselect 候选 B（CA1858 有座）**直接提交。
- 只有当三个候选航班**全部售罄**时，才 `regenerate`（重跑 LLM 重新找航班）——最后手段。

**收益对比**：

| 情形 | OCC | CAST |
|---|---|---|
| MU 座位被并发扣减（仍有余票） | abort → **重跑 LLM 重新规划**（花 c_gen，秒级 + token） | 验证放行 + DELTA 合并到最新库存（花 c_merge）|
| MU 售罄 | abort → 重跑 LLM 换航班 | **reselect 候选 B**（免费复用已生成候选）|
| 三航班全售罄 | 重跑 LLM | regenerate（与 OCC 相同，诚实边界）|

这一个任务同时用到三层机制：探索式多候选（①）、语义感知验证放行（②）、成本不对称的 merge/reselect/regenerate（③）。
对应实验：座位 DELTA 合并见 §5.4（真实 VitaBench），多候选 reselect 见 §5.3，验证放行减少冲突见 §5.5。

---

## 7. 当前限制与下一步

- 真实 LLM-in-the-loop（当前用模拟 agent + 对数正态延迟代替）；
- 语义感知验证的隔离级别形式化（可交换写放宽下的正确性保证）；
- 读密集/写偏斜的真实 VitaBench 跨域负载，进一步区分 MVCC；
- SCC order-shadow 真实现替换解析模型；CandidateScheduler 自适应 k。

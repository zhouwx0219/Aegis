# CAST: Cost-Asymmetric Speculative Transactions for LLM-Agent Data Management
## 论文初稿（供组会/导师讨论）

> 目标会场：SIGMOD / VLDB / ICDE。本稿整合本项目全部理论、系统与实验。所有性能数字以**真多线程并发实测**为准（`concurrent_harness.py` / `cc_comparison.py` / `ablation_experiment.py`），解析模型与早期实验作为佐证或敏感性分析；结果可复现（`cast-das/`）。

---

## 摘要（Abstract）

LLM agent 正成为新的数据访问主体：它们探索式地生成多个候选方案、调用工具读写共享状态，最终提交一个。这类负载与传统 OLTP 有一个根本差异——**执行成本被反转**：传统事务 abort 代价微秒级，而 agent 的 abort 意味着**重跑一次秒级、付费的 LLM**，二者相差 4–6 个数量级。我们指出，这一成本反转使经典并发控制的两个隐含假设失效：OCC 的"abort 廉价"与投机并发（SCC）的"冗余探索廉价"。

我们提出 **CAST（Cost-Asymmetric Speculative Transactions）**：一个面向 agent 探索式任务、运行在任意"版本化 KV"之上的统一对象事务层。CAST 的核心是**按写语义分级的并发控制**——在验证阶段放行只读与可交换写（从源头减少冲突），对真冲突用**廉价的语义合并 / 候选复用**替代昂贵重跑，并以 **escrow** 表达带约束的扣减。我们给出 CAST 的混合隔离级别 **CSI-SS** 及其正确性证明。在真多线程实测下，CAST 相对一整族先进并发控制（OCC/Silo/TicToc/MVCC/2PL）在吞吐与延迟上一致占优，且优势随冲突强度与候选数单调增大；消融实验分离出读放行、语义合并、多候选复用三者各自的贡献。

---

## 1. 引言（Introduction）

### 1.1 背景
数据管理社区正把"声明式查询 + 优化器"范式搬到 LLM 驱动的数据处理（Palimpzest、Nirvana、LOTUS 等），但这一脉**全是只读分析（AI-OLAP）**。与此同时，agent 越来越多地**写**共享状态（下单、改单、占用资源），却没有面向 agent 的事务/并发控制层。**本文填补"AI 时代的 OLTP"这一空白。**

### 1.2 关键观察：探索成本被反转
LLM agent 任务有两个本质特性：(i) **探索式多候选**——为质量生成多个候选方案、择优提交；(ii) **执行极贵**——每个候选 = 秒级 LLM 推理 + 工具调用 + token 费用。于是单位"重试/冗余"成本相对传统事务被放大 4–6 个数量级（`c_gen ≈ 10⁴–10⁶ · c_merge`）。

### 1.3 经典并发控制为何失效
- **OCC**：乐观假设"abort 便宜，冲突重来"——在 agent 场景 abort=重跑 LLM，假设崩塌。
- **SCC（投机并发）**：假设"多开 shadow 赌序列化顺序廉价"——每个 shadow 都是一次昂贵生成；我们实测其最优投机度退化为 1（即 OCC）。
- 此外，经典验证是**纯语法**的：把语义上可交换的并发写（如两次库存扣减）误判为冲突，触发不必要的昂贵重跑。

### 1.4 贡献
1. **问题设定**：形式化"成本不对称的 agent 事务"，把并发控制目标从"最大化吞吐"重述为"最小化为提交合格 winner 所浪费的 LLM 算力"。
2. **CAST 机制**：按写语义分级的并发控制——验证层放行只读/可交换写（减少冲突），冲突解决层用语义合并 → 候选复用 → 重跑的成本不对称顺序；并以 escrow 把"带约束扣减"从超卖边界转为可并发收益。
3. **正确性**：给出混合隔离级别 **CSI-SS** 与四条定理（strict 可串行化 / commutative 收敛 / CAS 条件安全 / escrow 安全）及证明。
4. **系统**：统一对象事务层，运行在仅需 5 个原语的"版本化 KV"抽象之上（后端可替换）。
5. **评估**：真多线程实测，对比 OCC/Silo/TicToc/MVCC/2PL/SCC，跨冲突强度 × 候选数 × 可合并占比，并接入真实 VitaBench 环境验证负载特征；消融分离各机制贡献。

---

## 2. 问题定义（Problem Statement）

**成本模型**：候选生成成本 `c_gen`（秒级 LLM+工具），语义合并成本 `c_merge`（微秒级 KV 操作），候选复用 `c_reselect ≈ 0`（复用已生成候选）。核心假设 `c_gen ≫ c_merge`。

**优化目标**：对一批并发 agent 任务（各需提交一个达标 winner），最小化总浪费算力
`E[ waste ] = (冗余候选数)·c_gen + n_regen·c_gen + n_merge·c_merge`，
等价地在真并发下最大化吞吐、最小化延迟。

**与传统 CC 的区别**：传统 CC 目标是"满足隔离级别下最大化吞吐"，且 abort 视为廉价；CAST 在成本反转下，把"避免昂贵重跑"提升为一等目标，并允许按写语义放宽验证以减少冲突。

---

## 3. 相关工作（Related Work）

- **声明式 AI 分析 / data agent system**（Palimpzest, Nirvana/Beyond Relational, LOTUS, DocETL）：把数据库优化器范式用于 LLM 数据处理，但**仅只读分析**；CAST 做事务/写入侧。
- **并发控制族**（OCC, Silo, TicToc, MVCC-SI, 2PL, SCC）：均为**纯语法**机制——对可交换写写冲突一律 abort/重跑（差异在读集处理与多核扩展性，与"可交换性"正交）。我们实测它们聚成一簇，CAST 的语义合并是其上的独立收益。SCC 的投机 shadow 在 agent 成本下最优退化为 OCC。
- **语义 / 可交换并发**（Escrow[O'Neil'86], Doppel, Coordination-Avoidance, CRDT）：用可交换性避免冲突，省的是廉价 CPU 重试；CAST 在 agent 成本下**重新定价**（省昂贵 LLM 重跑），并把它嵌入探索式多候选事务与 escrow 约束表达。
- **LLM agent 事务**（SagaLLM, Atomix, STORM, ATCC）：面向**多 agent 协调**（saga/补偿/锁调度）；CAST 面向**单任务多候选**的成本不对称投机事务。
- **投机执行**（Speculative Actions, Sherlock, SPAgent）：serving 层的 control-flow 延迟隐藏，**不处理事务一致性**；CAST 是 data-write 并发控制。

---

## 4. 系统设计（Design）

### 4.1 架构
混合栈：上层算子/调度（Python）—— pybind11 —— **C++ 事务内核** —— **版本化 KV**（仅需 5 原语 `Get/GetVersion/PutIfVersion/BatchPutIfVersion/DeleteIfVersion`，后端可替换）。事务、分支、并发语义全部在内核，存储不理解类型/事务/并发。

### 4.2 统一对象与写意图
对象统一抽象（第一版 row/text/counter）；写按意图分类：`READ`（只读）、`OVERWRITE`（strict）、`APPEND/DELTA`（commutative）、`CAS`（conditional）、`ESCROW`（带约束扣减）。意图由算子声明（未来可由 LLM 推断）。

### 4.3 提交协议（两阶段）
**阶段一·语义感知验证**（按意图分级判冲突）：只读与可交换写**放行**（不因版本变判冲突），CAS 验条件，strict 严格版本校验。
**阶段二·成本不对称解决**（状态机）：`direct → merge（可交换 rebase，花 c_merge）→ reselect（复用其他候选，≈0）→ regenerate（重跑，花 c_gen，最后手段）`。

### 4.4 escrow 约束表达
带下界/容量约束的扣减用**额度预留**：并发事务各预留 `q`（只要剩余 `≥ q`），提交确认。预留可交换、互不阻塞（总额 `≤` 容量）⇒ 保留并发合并；任何越界预留被拒 ⇒ 不超卖。把"超卖边界"转成"可并发收益"。

### 4.5 候选调度
`CandidateScheduler` 决定候选数 `k` 与生成时序（当前规则版；自适应版按 strict 占比/争用增开候选用 reselect 兜底）。

---

## 5. 正确性（Correctness）：CSI-SS

完整形式化与证明见 `PROOFS.md`，此处给结论。

**系统模型**：对象状态 `S(o)=(val,ver)`；事务 `T=(RS,WS)`，写按 `cls∈{S,C,K,Escrow}`；提交序 `H` 由提交点互斥定全序。

**定理**（证明见附录）：
- **T1（strict 可串行化）**：仅 strict 写时，`H` 冲突可串行化（= OCC first-committer-wins）。
- **T2（commutative 收敛）**：仅可交换写时，终态 `= val₀ ⊕ (⊕ Δ_T)`，与提交序无关、不丢更新 ⇒ 状态可串行化（前提：`⊕` 真可交换；顺序敏感 APPEND 须降级 strict）。
- **T3（CAS 条件安全）**：CAS 写仅在提交点条件成立时落库。
- **T4（escrow 安全+可交换）**：不超卖（不变量归纳）且预留可交换（并发合并）。

**主定理（CSI-SS）**：读 = 快照隔离；strict 可串行化、commutative 收敛、CAS 条件安全、escrow 约束保持的收敛。即 **读 SI + strict 可串行 + commutative 收敛(CRDT-SEC) + CAS 条件** 的混合点；锚 Adya（读 PL-SI、strict PL-3）。

**明确不保证**（反例）：write-skew（读放行，同 SI）、无 escrow 的带约束可交换写（超卖，实测 5−8=−3）、跨对象不变量、顺序敏感 APPEND。需要时把对应写升一档（conditional/escrow/strict）。

---

## 6. 评估（Evaluation）

**设置**：统一真多线程并发框架，真实墙钟计时（`sleep` 代表 `c_gen` 秒级、`c_merge≈0`）；对象池含 counter(可合并)/row(strict)；多 seed 报均值±std。Baseline：OCC、Silo、TicToc、MVCC-SI、2PL、SCC-kS。所有图见 `agent/experiments/results/`。
> 口径说明：**性能数字以真并发实测为准**（§6.1–6.3）；早期解析时间模型（`timed.png`、`vitabench_ota.png`）作为趋势佐证，其绝对倍数偏乐观，正文不据其下结论。

### 6.1 多 CC 对比（`cc_comparison.png`，真并发）
扫**冲突等级 × 候选数 k**。结论：**一整族 syntactic CC 聚成一簇**（OCC≈Silo；TicToc≈MVCC 读放行略高），因其都对可交换写写冲突 abort/重跑；**CAST 因语义合并单独领先，且冲突/候选越多优势越大**。高冲突（池=12）throughput：CAST≈2043 vs 最优 syntactic≈1434 vs 2PL≈483。

### 6.2 真并发吞吐 / 延迟（`concurrent.png`，3 seeds）
8 线程实测：吞吐 **CAST 1289 > OCC 1035 > 2PL 530**；平均延迟 **CAST 6.0ms < OCC 7.5ms < 2PL 14.6ms**（2PL 锁等待最高）。CAST 用 88 次合并把 OCC 的 128 次重跑降到 71。**误差棒小，结论稳定。**

### 6.3 消融（`ablation.png`，中等冲突）
从 OCC 逐步叠加：OCC 1472 → +读放行 1591(+119) → +语义合并 1861(+270) → +多候选 reselect（=CAST）2351(+490)，**总 +60%**。三机制均有正贡献，**语义合并与多候选 reselect 是主力**。

### 6.4 成本不对称与投机失效（`sweeps3.png`）
CAST 优势随 `c_gen/c_merge` 增大而增大，**交叉点 c_merge≈0.4·c_gen**；**SCC 投机 shadow 在 agent 成本下最优退化为 k\*=1（即 OCC）**——印证"投机永不划算"。对齐验证（OCC 实测 regen == 结构冲突数 == SCC-1S 解析）全通过。

### 6.5 探索式多候选的独立收益（`explore.png`，纯 strict）
纯 strict 独占资源（CAS，全程 `n_merge=0`）：CAST 的 reselect 随并发 10→136，证明**多候选机制独立于语义合并**有效（可合并写靠 merge、strict 写靠异构 reselect，两条正交收益线）。

### 6.6 真实负载特征（`vitabench_ota.png`，VitaBench OTA）
接入**真实 VitaBench 环境**（绕过其 3 个 LLM 模块，模拟 agent 驱动）：`use_tool` 实测 `create_flight_order` 扣减共享座位库存（`quantity 35→33`=DELTA）。证明**真实 agent 负载中存在可合并写 + 共享资源争用**（delivery 域则以私有订单 CREATE 为主、争用低）。**收益面 = "共享可变资源 × 可交换写" 的比例。**

### 6.7 正确性边界与 escrow（`escrow.png`）
带下界约束扣减：纯 DELTA 放行合并在需求>容量时**超卖**（5−8=−3）；strict-CAS 正确但重跑代价线性升；**escrow 唯一同时"正确（封顶容量、不超卖）+ 零重跑"**——把超卖边界转成可并发收益。

---

## 7. 讨论与限制（Discussion & Limitations）

- **诚实的收益幅度**：真并发实测下，CAST 相对最优 syntactic CC 领先约 **40–60%（高冲突）**，**温和但稳定、可解释、单调随冲突/可合并占比增大**——并非碾压。我们认为"对比一整族先进 CC 仍有正交收益 + 边界诚实"比夸大倍数更可信。
- **收益边界**：可合并写少、私有对象为主的负载（如 delivery 私有订单），CAST 收益主要来自多候选 reselect 而非合并。
- **限制**：(i) Silo/TicToc 的多核可扩展性创新在单提交点模型下未体现，本文只对比其 abort 语义；(ii) escrow 当前为算法演示，尚未下沉 C++ 内核；(iii) **无真实 LLM-in-the-loop**——`c_gen` 用受控 sleep 表示，"agent 探索式多候选"的真实频率有待真实负载实证（motivation 的最后一公里）；(iv) 无 crash recovery / 持久化；(v) 存储为内存 reference 实现（抽象可替换，接真实后端为可移植性验证）；(vi) CSI-SS 定理为 appendix 级，完整机检（TLA+/Coq）列为 future。

---

## 8. 结论（Conclusion）

agent 数据管理把"探索的成本"反转，从而反转了并发控制的最优策略：从"乐观重试/多开 shadow"转向"按语义放行可化解的并发 + 对真冲突用便宜的合并/复用替代昂贵重跑"。CAST 将其实现为运行在版本化 KV 之上的统一对象事务层，给出混合隔离级别 CSI-SS 与证明，并在真并发实测中对一整族先进并发控制取得一致、可解释的收益。

**Future work**：真实 LLM-in-the-loop 与跨域真实负载、escrow 下沉与真实后端、CSI-SS 机检、自适应验证策略选择。

---

## 附：实验复现
```bash
cd cast-das && bash build.sh
for e in sweep_contention timed_experiment explore_experiment semantic_validation_experiment \
         correctness_boundary escrow_experiment concurrent_harness cc_comparison ablation_experiment plot_sweeps; do
  python3 agent/experiments/$e.py
done
bash agent/integrations/setup_vitabench.sh && cd /tmp/vb && \
  python3 .../agent/integrations/vitabench_ota_concurrency.py
```
配套文档：`CAST_REPORT.md`（详版）、`ISOLATION_LEVELS.md` + `PROOFS.md`（理论）、`GAP_ANALYSIS.md`（差距）、`GROUP_MEETING.md`（汇报）。

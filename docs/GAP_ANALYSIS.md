# CAST 距离顶会论文的 Gap 分析

> 前提：暂不考虑"接入真实 LLM API"与"更多真实负载"。对照 SIGMOD / VLDB / ICDE 的投稿标准。

## 0. 一句话判断

当前 = **强 idea + 完整原型 + 自洽的实验框架 + 理论草图**。距离可投稿，主要差四块：
**实验可信度（真实并发实测 + baseline 真实现 + 统计严谨）、理论严格化（定理证明）、系统深度（escrow 下沉/后端/恢复）、论文成稿**。
> **最大单点风险**：当前所有吞吐/延迟/浪费数字，要么是单线程 cast_core 真跑 + **解析时间模型**，要么纯解析——**没有真正的并发执行 + 真实墙钟计时**。审稿人第一刀就会问"12× 吞吐是算出来的还是跑出来的"。

---

## 1. 已具备（参照系）

- **Problem / 定位**：成本不对称 + agent 探索式事务（AI-OLTP）；与 Palimpzest 系（AI-OLAP）的空白区分。
- **机制**：CAST 四件套——语义感知验证分级、成本不对称提交（merge/reselect/regenerate）、探索式多候选、escrow。
- **系统骨架**：C++ 核（header-only 内存）+ pybind11 + Python 算子；统一对象事务、纯版本化 KV 边界。
- **实验框架**：受控三方（OCC/SCC）、多维（成本×延迟×吞吐 + 2PL/MVCC）、explore（多候选）、VitaBench-OTA（真实环境对象访问）、P3 语义验证、escrow、正确性边界。
- **理论**：隔离级别形式化草图（CSI-SS + 三定理草图 + 4 条不保证）。
- **文档**：CAST_REPORT / ISOLATION_LEVELS / GROUP_MEETING / VITABENCH_INTEGRATION。

---

## 2. Gap 分类（对照 SIGMOD/VLDB）

### A. 实验可信度（最关键）
- **A1 真实并发执行 + 真实计时**（最高优先）：把"makespan 解析公式 + t_gen 常数/对数正态"换成**真多线程并发 + 真实墙钟**。不接 LLM 也能做——用受控 `sleep`/CPU 负载代表 `c_gen`（秒级）与 `c_merge`（微秒级），真并发跑出吞吐/延迟。这是把"模拟"变"实测"。
- **A2 baseline 真实现** **[已完成]**：OCC/Silo/TicToc/MVCC-SI/2PL/CAST 已在**同一真并发框架**真跑（`cc_comparison.py` + `concurrent_harness.py`）——syntactic CC 聚成一簇、CAST 语义合并正交领先（冲突/候选越多优势越大）。
- **A3 统计严谨**：当前基本单 seed。需多 seed + 置信区间 + 必要的显著性。
- **A4 消融（ablation）**：系统地分离"验证放行 / 成本不对称解决 / 多候选 reselect / escrow"各自的贡献（现在分散在不同实验，未统一消融）。
- **A5 敏感性完备**：对象分布、争用模式、`k`、`c_gen/c_merge` 比值的系统化扫描（已有部分：c_merge 扫描、并发扫描）。

### B. 理论严格化
- **B1 三定理完整证明**：CSI-SS 现为草图。需严格的系统模型 + 操作语义 + 历史/等价性定义 + 完整证明（或 TLA+/Coq 机检、反例库佐证）。
- **B2 escrow 正确性证明**：预留不超卖 + 与可串行化/收敛的关系。
- **B3 恢复 / 持久性**：事务系统论文的标配。CAST 当前**无 crash recovery / durability**（原 `data_agent_system` 有 commit-log/fallback，尚未接入 cast-das）。

### C. 系统深度
- **C1 escrow（kEscrow）下沉 C++ 核**：现为 Python 算法演示，需作为一等并发类进核。
- **C2 真实存储后端** **[降级为非阻塞]**：版本化 KV 是可替换抽象（5 原语 Get/PutIfVersion/BatchPutIfVersion/...），内存 reference 实现足以验证 CC 正确性与收益；接 RocksDB/TiKV = 可移植性工程验证，不影响核心贡献。
- **C3 真正的多线程 / 并行执行**：现"批内依次提交"是顺序模拟，需真并发（与 A1 同根）。

### D. 写作与成稿
- **D1 论文成稿**：尚无 LaTeX 双栏稿（abstract/intro/problem/related/design/eval/conclusion，~12 页）。
- **D2 related work 章节 + 与最强对手实测对比**：ATCC / SagaLLM / Doppel 目前只有文字区分，缺实测。
- **D3 motivation 叙事**：把"为什么 agent 需要这套"写成有冲击力的引言。

---

## 3. 诚实的张力（排除 LLM/真实负载的代价）

- 顶会（尤其系统/DB）对"真问题"的说服力，很依赖**真实负载 + 真实 agent**。排除后，**motivation 的真实性是隐患**——审稿人会问"真实 agent 真会产生这么高比例的可合并并发写、这么多多候选探索吗？"。
- VitaBench-OTA 已是真实环境的对象访问（可继续做深、做严谨，不算"更多真实负载"），但**不接 LLM，"探索式多候选"的真实频率无法实证**——这会让 reselect 那条收益线的强度存疑。
- 结论：不接 LLM/真实负载，可以把**系统 + 理论 + 受控实验**做扎实、足以撑起论文骨架；但 **motivation 的"最后一公里"（真实性）迟早要补**，否则核心主张易被质疑为"假设驱动"。

---

## 4. 优先级与工作量（约束内能做的）

**P0（投稿门槛，约 1–2 月）**
1. **A1 真实并发执行 + 真实计时**——把吞吐/延迟从模型变实测（最高优先，直接消除最大风险）。
2. **A2 baseline 真实现**——至少 OCC / 2PL / MVCC / SCC-kS 在同一真并发框架真跑。
3. **B1 三定理证明严格化**（+ B2 escrow 正确性）。

**P1（约 1 月）**
4. A3 多 seed + 置信区间；A4 统一消融。
5. C1 escrow 下沉 C++。
6. D1 论文成稿（intro/design/eval 先行）。

**P2（其后）**
7. C2 真实存储后端；B3 恢复/持久性；C3 真并行；D2 与最强对手实测对比。

---

## 5. 一句话路线

在不接 LLM/真实负载的前提下，**先把"实验从模拟变实测 + baseline 真实现 + 定理证严 + 论文成稿"做完**——这能让论文骨架站住、消除最大可信度风险；但要清醒：**motivation 的真实性（LLM / 真实负载）是迟早要补的最后一公里**，越早补，novelty 越稳。

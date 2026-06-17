# cast-das

**论文 A 的实现骨架**：Data Agent System 中的统一对象事务与成本不对称投机提交（CAST）。

> 设计依据：`../docs/paperA_unified_object_txn_system_design.md`
> 这是阶段 0（端到端骨架）：C++ 事务内核 + pybind11 桥接 + Python 模拟算子，
> 已能跑出"CAST 用语义合并替代昂贵重跑、浪费算力远低于 strict OCC"的核心结果。

## 架构（混合栈）

```text
Python 层（agent/）          —— 算子、调度、workload、实验
   • MockOperator：可配置成本/冲突的模拟算子（确定性、可复现）
        │  pybind11（进程内）
C++ 核（core/）              —— 事务/并发/提交，性能与正确性
   • 统一对象 + 版本化存储 + 写意图与并发类 + 成本模型
   • 成本不对称提交协议（merge / reselect / regenerate）
```

所有事务/并发/提交都在 C++ 核完成；Python 只负责产生候选、声明写意图、提供成本数字。

## 目录

```text
cast-das/
├── core/                                  # C++ 事务内核（header-only）
│   ├── object/unified_object.h            # 统一对象类型 + 版本化值
│   ├── intent/intent.h                    # 五类写意图
│   ├── intent/policy_dispatcher.h         # 并发类分类 + 语义 rebase（核心复用）
│   ├── storage/versioned_object_store.h   # 版本化对象存储（最小 KV 边界）
│   ├── cost/cost_model.h                  # 成本模型与统计（浪费算力）
│   ├── branch/speculative_branch.h        # 成本标注的投机分支
│   ├── txn/cost_asymmetric_commit.h       # ★成本不对称提交协议（CAST 核心）
│   └── bindings/cast_bindings.cpp         # pybind11 桥接
├── agent/                                 # Python 层
│   ├── operators/mock_operator.py         # 可配置模拟算子
│   └── experiments/demo_e2e.py            # 端到端 demo：CAST vs strict OCC
├── build.sh                               # 编译脚本（生成 cast_core 扩展）
└── README.md
```

## 构建与运行

依赖：g++ (C++17)、python3、pybind11（`pip install pybind11`）。

```bash
bash build.sh                          # 编译 cast_core 扩展模块
python3 agent/experiments/demo_e2e.py  # 运行端到端 demo
```

## 当前 demo 展示什么

- **场景1（可合并冲突）**：候选对 counter 做 `DELTA`、对 text 做 `APPEND`，并发任务抢先改了两者。
  strict OCC 必须 `regenerate`（浪费一次昂贵生成）；CAST 用语义 rebase `merge`（仅 KV 操作）。
  结果：CAST 浪费算力比 OCC 低约 98%。
- **场景2（strict 冲突，诚实边界）**：候选对 row 做 `OVERWRITE`，并发抢先覆盖。
  OVERWRITE 不可合并，CAST 退化到与 OCC 相同——说明优势来自语义可合并的写。

## 与现有 Data-Agent-System 的关系

本骨架移植/复用了上层 `data_agent_system/` 的核心思想：版本化 KV 边界、五类写意图、
`PolicyDispatcher` 的并发类与语义 rebase、统一对象缓存（ObjectType）。
新增的是成本模型、成本不对称提交协议，以及 Python 算子层与桥接。

## 阶段 1 进展

### 受控负载扫描与三方对比（OCC vs SCC-kS vs CAST）

组件：
- `agent/workloads/synthetic_contention.py`：确定性 task-plan + 批内并发争用模型。
- `agent/scheduler/candidate_scheduler.py`：候选生成统一入口，决定投机度 k（当前 k=1 规则版）。
- `agent/experiments/sweep_contention.py`：真跑 OCC/CAST + SCC-kS 解析成本模型，扫并发度/可合并占比/成本不对称度。
- `agent/experiments/plot_sweeps.py`：画三方对比图 `results/sweeps3.png`（早期两方图 `results/sweeps.png` 保留）。

SCC-kS 解析模型：`waste = (k-1)*c_gen*n_tasks + #(冲突深度 d>=k)*c_gen`；SCC-1S == OCC，用于对齐验证。
task-plan 让"真跑 OCC/CAST"与"SCC 解析/结构性冲突深度"读同一访问序列 —— alignment 全部 PASS
（OCC 实测 regen == 结构冲突任务数，OCC 浪费 == SCC-1S 解析）。

核心结论（人均浪费，单位 c_gen/任务，见 `results/sweeps3.png`）：
- **CAST 完胜**：可合并写场景浪费比 OCC 低 >97%（高并发下约 0.02 vs 0.9）。
- **SCC 投机反而更差**：SCC-2S 每任务先付 1 份冗余 shadow，浪费约 1.6 > OCC 的 0.83；
  且 SCC-best 在 k=1..8 的最优是 k\*=1（退化成 OCC）—— agent 成本下投机 shadow 永不划算。
- **CAST 的诚实边界**：优势来自"可合并写占比高 + 成本不对称"；p_mergeable=0（全 strict）时 CAST≈OCC；
  c_merge 接近 c_gen（交叉点≈0.4）后 CAST 反而更差。

运行：
```bash
python3 agent/experiments/sweep_contention.py   # 三方 CSV: results/sweep3_*.csv
python3 agent/experiments/plot_sweeps.py        # 三方图: results/sweeps3.png
```

### 多维对比：成本 × 延迟 × 吞吐（Step A，含 2PL / MVCC）

`agent/experiments/timed_experiment.py` → `results/timed.png`。引入时间模型后四方对比，batch=16：
- 浪费/任务：OCC 0.66 / 2PL 0.00 / CAST 0.02
- 平均延迟（t_gen）：OCC 1.66 / 2PL 2.38 / CAST 1.02
- 吞吐（提交/墙钟）：OCC 1.4 / 2PL 3.0 / CAST 12.0

=> **CAST 三维全赢**；2PL 不重跑但锁串行化→延迟/吞吐受压；MVCC-SI 写密集下 ≡ OCC。

### 探索式多候选的独立收益（纯 strict，与语义合并正交）

`agent/experiments/explore_experiment.py` → `results/explore.png`。纯 strict 独占资源负载（CAS，全程 n_merge=0）+ 对数正态 LLM 延迟。每任务 k=4 异构候选，winner 被占→CAST 免费 reselect、OCC 重探索。batch=16：
- 浪费/任务：OCC 1.34 / CAST 0.03；平均延迟 OCC 2.34 / CAST 1.77；吞吐 OCC 2.55 / CAST 4.83。
- CAST reselect 随并发增长（10→32→76→136），**首次实证多候选机制**（此前 k=1 时恒为 0）。
- 与语义合并正交：可合并写靠 merge，strict 写靠异构多候选 reselect，两条独立收益来源。

## 仍待做（阶段 2）

1. **真实 LLM-in-the-loop VitaBench**：用真实 t_gen 与真实冲突分布复现结论。
2. **CandidateScheduler 自适应 k**：strict 写多时增开候选用 reselect 兜底。
3. **读密集/写偏斜负载**：让 MVCC-SI 的"读不阻塞"优势显现，区分 MVCC vs OCC。
4. **SCC order-shadow 真实现** 与 **隔离级别形式化**。

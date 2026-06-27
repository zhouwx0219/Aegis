# Data Agent System / CAST-DAS

这是一个面向 **Data Agent 事务处理** 的研究原型。项目的核心思想是：

- 底层只负责 **版本化 KV 存储**；
- Agent runtime 负责 **事务语义、候选分支、并发控制选择、提交协议和重试/再生成边界**；
- ATCC、传统 CC、语义 CC 都作为可插拔模块接入，而不是写死进底层存储。

当前项目重点支持在类 Agent 的 YCSB / TPC-C 改造负载上，对比：

- OCC
- 2PL No-wait / Wait-die
- MVCC
- SILO
- TicToc
- ATCC 

## 1. 项目结构

```text
cast-das/
├── agent/
│   ├── runtime/        # Agent 事务运行时、CC registry、ATCC、提交协议
│   ├── workloads/      # Agent-YCSB / Agent-TPCC 改造负载
│   └── evaluation/     # 实验 runner、profile 训练、DBx1000 native runner
├── core/               # C++ 核心接口：版本化 KV、并发控制、事务提交内核
├── tests/              # 单元测试
├── third_party/dbx1000 # DBx1000 源码，用作参考和 native baseline
├── scripts/            # 交接用简化运行脚本
├── docs/               # 精简后的说明文档；历史过程已归档到 docs/archive/
├── results/            # 精简后的关键结果；历史结果已归档到 results/archive/
├── build.sh            # 构建 Linux/WSL Python 扩展
├── pyproject.toml
└── README.md
```

## 2. 系统架构

系统边界可以理解为：

```text
Agent Task
  -> K 个候选计划
  -> AgentTransactionManager
  -> Branch Semantics / CC Registry / Commit Protocol
  -> VersionedKVStore
```

关键模块：

- `agent/runtime/transaction.py`
  - 管理事务生命周期、snapshot、prelock lease、提交/abort 结果。
- `agent/runtime/cc_registry.py`
  - 注册和选择并发控制策略。
- `agent/runtime/adaptive.py`
  - Operation-level ATCC 策略表。
- `agent/runtime/atcc.py`
  - Phase-aware ATCC 模块、Q 表、reward 和 priority。
- `agent/runtime/traditional_cc.py`
  - Agent runtime 层的传统 CC baseline：2PL、MVCC、SILO、TicToc。
- `agent/workloads/ycsb.py`
  - 类 Agent 的 YCSB 改造负载。
- `agent/workloads/tpcc.py`
  - 类 Agent 的 TPC-C 改造负载。
- `agent/evaluation/atcc_retry_experiment.py`
  - 当前主要实验入口。

底层 DBx1000 在这个项目里主要作为 **进程内、非持久化、版本化 KV substrate**。Agent 事务语义不依赖 DBx1000 原生事务线程模型。

## 3. 环境与构建

推荐使用 WSL/Linux 运行，因为 `cast_core.cpython-312-x86_64-linux-gnu.so` 是 Linux Python 扩展。

```bash
python3 -m pip install -e .
bash build.sh
python3 -m unittest discover -s tests -v
```

如果在 Windows PowerShell 中操作，涉及运行实验时请通过 WSL：

```powershell
wsl -e bash -lc "cd /mnt/z/Data-Agent-System-master/cast-das && python3 -m unittest discover -s tests -v"
```

## 4. 快速验证

交接时建议先跑一组轻量测试：

```bash
python3 -m unittest \
  tests.test_atcc_ycsb_strict_tuned \
  tests.test_traditional_cc_full \
  tests.test_atcc_retry_experiment \
  -v
```

这能覆盖：

- 新的 YCSB tuned ATCC variant；
- 传统 CC baseline；
- retry experiment 的核心指标汇总。

## 5. 简化实验命令

新增了 PowerShell 包装脚本。

### 5.1 跑 YCSB

```powershell
.\scripts\run_ycsb_compare.ps1 -Profile high -StrategySet atcc -PolicyVariant ycsb-strict-tuned
```

常用参数：

- `-Profile low|medium|high|all`
- `-StrategySet atcc|full`
  - `atcc`：`occ,tictoc-full,adaptive-op-strict`
  - `full`：`occ,2pl-nowait,2pl-wait-die,mvcc-full,silo-full,tictoc-full,adaptive-op-strict`
- `-PolicyVariant ycsb-strict-tuned|default`
- `-TaskCount 60`
- `-OutputDir results/handoff_ycsb_compare`

输出：

```text
results/handoff_ycsb_compare/
├── ycsb-low.json
├── ycsb-medium.json
├── ycsb-high.json
└── summary.csv
```

### 5.2 跑 TPC-C

```powershell
.\scripts\run_tpcc_compare.ps1 -Profile high -StrategySet full
```

输出：

```text
results/handoff_tpcc_compare/
├── tpcc-low.json
├── tpcc-medium.json
├── tpcc-high.json
└── summary.csv
```

### 5.3 手动汇总已有 JSON

```powershell
python .\scripts\summarize_retry_results.py --input-dir .\results\ycsb_strict_tuned_atcc_20260625
```

## 6. 当前推荐实验入口

如果只想复现当前最重要的 YCSB 结果：

```powershell
.\scripts\run_ycsb_compare.ps1 `
  -Profile all `
  -StrategySet atcc `
  -PolicyVariant ycsb-strict-tuned `
  -OutputDir results/handoff_ycsb_tuned
```

如果要覆盖所有传统 CC：

```powershell
.\scripts\run_ycsb_compare.ps1 `
  -Profile all `
  -StrategySet full `
  -PolicyVariant ycsb-strict-tuned `
  -OutputDir results/handoff_ycsb_tuned_full
```

## 7. 当前关键结果

保留在 results 顶层的关键实验：

- `results/ycsb_strict_tuned_atcc_20260625/`
  - YCSB default ATCC vs tuned ATCC A/B。
- `results/ycsb_strict_tuned_full_cc_20260625/`
  - tuned ATCC 与完整传统 CC 矩阵。
- `results/full_traditional_cc_ycsb_20260625/`
  - 调优前 YCSB 完整传统 CC 对比。
- `results/full_traditional_cc_tpcc_20260625/`
  - TPCC 完整传统 CC 对比。
- `results/atcc_pressure_reactive_guard_obs4_profiles_20260624_23_confirm/`
  - 当前保留的 ATCC policy artifact。

历史 smoke、probe、cost sweep 和旧 profile 已归档到：

```text
results/archive/exploratory_20260624/
```

## 8. ATCC policy variant

当前最值得关注的 ATCC variant 是：

```text
ycsb-strict-tuned
```

它通过 `--policy-variant ycsb-strict-tuned` 启用，只影响 YCSB policy table，不改变默认 ATCC、底层 KV、事务边界或传统 CC。

设计意图：

- 低风险首轮保持 OCC；
- retry 后优先保护 hot writes；
- 更高 retry 才扩大锁范围；
- 避免 medium 下过早进入 full read-write locking；
- 在 high 下减少 retry 和 token waste。


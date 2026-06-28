# Data Agent System

这是一个面向 Data Agent System 的事务处理研究原型。当前代码的核心目标是：底层提供版本化 KV 存储，agent runtime 负责事务语义、候选计划、并发控制策略选择、提交协议和失败重试。

当前重点是 ATCC 在 agent-style TPCC/YCSB workload 上的实现和实验，包括操作级 ATCC、事务级 ATCC、传统 CC baseline 和 adaptive-hybrid family selector。

## 交接入口

建议先读：

1. [代码结构](docs/代码结构.md)
2. [事务级 ATCC](docs/ATCC事务级项目-2026.06.28.13/事务级ATCC汇报.md) 
3. [TPCC 订单事务类 Agent 负载流程](docs/ATCC事务级项目-2026.06.28.13/TPCC订单事务类Agent负载流程.md)
4. [两种 ATCC 差异说明](docs/ATCC事务级项目-2026.06.28.13/两种ATCC差异说明.md)

## 核心目录

```text
agent/runtime/       Agent 事务 runtime、CC registry、ATCC、commit protocol
agent/workloads/     Agent-style YCSB / TPCC workload
agent/evaluation/    实验 runner、policy training、profile/manifest runner
core/                C++/pybind 版本化 KV 和底层并发控制内核
docs/                论文、汇报、文档
results/             保留的正式实验结果
scripts/             实验复现和结果汇总脚本
tests/               单元测试和实验入口回归测试
```

## 当前关键策略

传统 CC：

```text
occ
2pl-nowait
2pl-wait-die
mvcc
silo
tictoc
```

ATCC / adaptive 策略：

```text
adaptive-op-strict        操作级 ATCC
transaction-atcc-strict   事务级 ATCC
adaptive-hybrid           策略族选择器
```

## 两种 ATCC

操作级 ATCC：

- 策略名：`adaptive-op-strict`
- 决策粒度：单个 read/write operation
- 输出：每个对象 optimistic 或 pessimistic
- 优点：锁粒度细，在 YCSB high 和 TPCC medium 上表现稳定

事务级 ATCC：

- 策略名：`transaction-atcc-strict`
- 决策粒度：整个 agent transaction
- 输入：事务阶段、read/write set、hot/cold set、retry、全局 abort/lock wait
- 输出：`occ`、`lock-hot-writes`、`lock-hot-read-write`、`lock-write-set`、`lock-read-write-set`
- 优点：更接近原论文 ATCC，TPCC-high下平均吞吐约为操作级 ATCC 的 `1.94x`

## Workload

YCSB key：

```text
ycsb:record:{record}:field:{field}
```

TPCC NewOrder key：

```text
tpcc:district:{w}:{d}:next_order_id
tpcc:district:{w}:{d}:orders
tpcc:stock:{w}:{item}:quantity
tpcc:stock:{w}:{item}:ytd
```

Agent workflow：

```text
explore -> refine -> commit
```

这和传统短事务不同：agent task 会有多候选计划、阶段化读取、planning delay、失败重试和 token/latency 成本。

## 保留结果

- `transaction_atcc_full_20260628_13`：事务级 ATCC 的 YCSB/TPCC low/medium/high 完整矩阵。

关键结果文件：

```text
results/transaction_atcc_full_20260628_13/transaction_atcc_metrics.csv
results/transaction_atcc_full_20260628_13/transaction_atcc_ratios.csv
```

## 关键实验结论

TPCC-high：


| 策略       | 平均吞吐   | 平均提交率  |
| -------- | ------ | ------ |
| OCC      | 0.000  | 0.0%   |
| MVCC     | 0.000  | 0.0%   |
| TicToc   | 0.021  | 0.3%   |
| 操作级 ATCC | 21.813 | 100.0% |
| 事务级 ATCC | 42.246 | 100.0% |


稳定结论：

- 传统 CC 在 TPCC-high agent workload 下基本失效。
- 事务级 ATCC 相对唯一非零传统 baseline TicToc 约 `1984x`。
- 事务级 ATCC 相对操作级 ATCC 平均吞吐约 `1.94x`。

## 实验命令

### 汇总已有结果

```powershell
python .\scripts\summarize_retry_results.py --input-dir .\results\transaction_atcc_full_20260628_13
python .\scripts\summarize_retry_results.py --input-dir .\results\transaction_atcc_tpcc_high_multiseed_20260628_13
```

### 跑 YCSB 对比

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ycsb_compare.ps1 `
  -Profile all `
  -StrategySet full `
  -TaskCount 60 `
  -Workers 24 `
  -OutputDir results/ycsb_reproduce
```

### 跑 TPCC 对比

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_tpcc_compare.ps1 `
  -Profile all `
  -StrategySet full `
  -TaskCount 60 `
  -Workers 24 `
  -OutputDir results/tpcc_reproduce
```




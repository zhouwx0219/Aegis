# Data Agent System / CAST-DAS

本项目是一个面向 **Data Agent 事务处理** 的研究原型。当前版本的核心目标是：底层提供版本化 KV 存储，agent runtime 负责事务语义、候选计划、并发控制策略选择、提交和重试。

当前交接重点是 ATCC 在 agent-style TPCC/YCSB workload 上的自适应并发控制实验。

## 核心交接文件

- 汇报文档：`docs/ATCC项目-2026.06.28.03/ATCC汇报.md`
- ATCC 论文：`docs/ATCC.pdf`
- 最终结果：`results/atcc_final_handoff_20260628/`

最终结果目录结构：

```text
results/atcc_final_handoff_20260628/
├── policies/
│   ├── tpcc-family-policy-window.json
│   ├── tpcc-family-search.json
│   └── ycsb-adaptive-readheavy-family-policy.json
├── tpcc/
│   ├── summary.csv
│   ├── tpcc-low.json
│   ├── tpcc-medium.json
│   └── tpcc-high.json
└── ycsb/
    ├── summary.csv
    ├── ycsb-low.json
    ├── ycsb-medium.json
    └── ycsb-high.json
```

## 项目结构

```text
agent/runtime/       Agent 事务运行时、并发控制注册、ATCC、提交协议
agent/workloads/     Agent-style YCSB / TPCC 工作负载
agent/evaluation/    实验 runner、策略训练、结果聚合
core/                版本化 KV 和底层绑定
scripts/             实验脚本和结果汇总脚本
tests/               单元测试和实验入口回归测试
docs/                汇报文档和论文
results/             精简后的核心实验结果
```

## Data Agent System 架构

系统分两层：

- 底层：版本化 KV 存储，只负责对象版本、读写、条件提交和原子更新。
- 上层：agent runtime，负责事务语义、候选计划、多阶段 workflow、并发控制策略选择和失败重试。

一个 agent task 会被表示成：

```text
AgentTask
  -> 多个 ranked candidates
  -> explore / refine / commit 阶段
  -> AgentTransactionManager
  -> CC registry / ATCC policy / commit protocol
  -> Versioned KV Store
```

## 自适应混合策略

自适应混合策略不是新的底层协议，而是策略族选择器。它会按 task 或 task window 判断应该使用哪类并发控制：

- OCC
- 2PL-nowait / 2PL-wait-die
- MVCC
- Silo 类协议
- TicToc 类协议
- 操作级 ATCC

操作级 ATCC 负责按每个操作决定乐观执行、提前加锁或提交前刷新。自适应混合策略负责决定整个任务走哪个策略族。

## 工作负载

### YCSB

YCSB 对象：

```text
ycsb:record:{record}:field:{field}
```

值类型是字符串字段值。操作包括 `read` 和 `overwrite`。每个 task 有多个候选计划，并按 `explore -> refine -> commit` 分阶段执行。

### TPCC

TPCC 对象被展开成版本化 KV：

```text
tpcc:warehouse:{w}:ytd
tpcc:district:{w}:{d}:next_order_id
tpcc:district:{w}:{d}:orders
tpcc:district:{w}:{d}:history
tpcc:customer:{w}:{d}:{c}:balance
tpcc:customer:{w}:{d}:{c}:payment_count
tpcc:customer:{w}:{d}:{c}:status
tpcc:stock:{w}:{item}:quantity
tpcc:stock:{w}:{item}:ytd
```

值类型包括 counter、append stream、status row、constrained counter。当前核心实验使用 NewOrder，workflow 是先读取库存，再提交订单号、订单流和库存更新。

## 核心实验结果

YCSB 自适应混合策略相对传统 CC 的吞吐收益：

| Profile | OCC | 2PL-nowait | 2PL-wait-die | MVCC | Silo | TicToc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 1.016x | 1.286x | 1.319x | 1.523x | 1.647x | 1.259x |
| medium | 1.514x | 2.333x | 2.559x | 0.901x | 2.251x | 0.826x |
| high | 5.819x | 18.809x | 17.444x | 1.766x | 11.504x | 1.358x |

TPCC 窗口感知自适应策略结果：

| Profile | Selected strategy | Throughput | Commit rate | vs OCC | vs MVCC | vs TicToc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| low | OCC | 5.415 | 100% | 1.116x | 1.205x | 1.290x |
| medium | operation-level ATCC | 20.106 | 100% | 2.154x | 2.377x | 1.840x |
| high | operation-level ATCC | 23.930 | 100% | 24.480x | 186.260x | 41.213x |

注意：TPCC high 的传统协议提交率很低，因此高倍数必须和 commit rate、attempts/task、p99 一起解释。TPCC low 的 `1.116x` 也不应解释成 OCC 算法本身更快，而是低冲突下自适应策略选择 OCC 快路径且本轮单 seed p99 略低。

## 环境和测试

推荐用 WSL/Linux 运行：

```powershell
wsl -e bash -lc "cd /mnt/z/Data-Agent-System-master/cast-das && python3 -m unittest tests.test_core tests.test_compare_scripts tests.test_atcc_family_policy tests.test_atcc_policy_training tests.test_atcc_profile_runner tests.test_atcc_manifest_runner tests.test_atcc_retry_experiment tests.test_atcc_ycsb_strict_tuned tests.test_traditional_cc_full -v"
```

最后一次相关回归结果：

```text
Ran 124 tests in 3.919s
OK
```

## 查看原始结果

```powershell
Import-Csv results\atcc_final_handoff_20260628\ycsb\summary.csv
Import-Csv results\atcc_final_handoff_20260628\tpcc\summary.csv
```

查看单个 JSON 聚合：

```powershell
wsl -e bash -lc "cd /mnt/z/Data-Agent-System-master/cast-das && python3 - <<'PY'
import json
from pathlib import Path
for path in [
    'results/atcc_final_handoff_20260628/ycsb/ycsb-high.json',
    'results/atcc_final_handoff_20260628/tpcc/tpcc-high.json',
]:
    data = json.loads(Path(path).read_text())
    print('\\n', path)
    for row in data['aggregates']:
        print(row['strategy'], row['committed_throughput'], row['commit_rate'], row.get('selected_strategy_counts'))
PY"
```

## 复现实验

### YCSB

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ycsb_compare.ps1 `
  -Profile all `
  -StrategySet full `
  -TaskCount 40 `
  -Workers 24 `
  -OutputDir results/atcc_matrix_ycsb_reproduce `
  -PolicyArtifact results/atcc_final_handoff_20260628\policies\ycsb-adaptive-readheavy-family-policy.json
```

### TPCC

TPCC 最终结果需要开启 `--hybrid-selected-fast-through`，因此直接调用 runner。下面是 high 档示例，low/medium 替换最后的规模参数即可。

```powershell
wsl -e bash -lc "cd /mnt/z/Data-Agent-System-master/cast-das && python3 -m agent.evaluation.atcc_retry_experiment \
  --workload tpcc \
  --strategies occ,2pl-nowait,2pl-wait-die,mvcc-full,silo-full,tictoc-full,adaptive-op-strict,adaptive-hybrid \
  --task-count 40 \
  --seed 920104 \
  --repeats 1 \
  --workers 24 \
  --agent-slots 4 \
  --agent-admission-mode before-begin \
  --planning-delay-ms 50 \
  --latency-distribution lognormal \
  --latency-cv 0.8 \
  --latency-max-ms 500 \
  --max-attempts 8 \
  --background-workers 4 \
  --background-interval-ms 2 \
  --background-strategy occ \
  --object-lock-scheduler bounded-priority \
  --object-lock-priority-burst 2 \
  --prelock-wait-budget-ms 70 \
  --prelock-wait-budget-mode object \
  --prelock-lease-mode yield-refresh-regenerate \
  --agent-execution-mode staged \
  --snapshot-timing before-planning \
  --policy-variant default \
  --policy-artifact results/atcc_final_handoff_20260628/policies/tpcc-family-policy-window.json \
  --hybrid-selected-fast-through \
  --transaction-mix new_order:1.0 \
  --warehouses 1 \
  --districts-per-warehouse 2 \
  --customers-per-district 40 \
  --items 100 \
  --order-lines 10 \
  --output results/atcc_matrix_tpcc_reproduce/tpcc-high.json"
```

规模参数：

```text
low:    --warehouses 8 --districts-per-warehouse 5 --customers-per-district 100 --items 500 --order-lines 5
medium: --warehouses 2 --districts-per-warehouse 3 --customers-per-district 60 --items 200 --order-lines 8
high:   --warehouses 1 --districts-per-warehouse 2 --customers-per-district 40 --items 100 --order-lines 10
```

生成 summary：

```powershell
python .\scripts\summarize_retry_results.py --input-dir .\results\atcc_matrix_tpcc_reproduce
```

## 下一步

- 对 TPCC 窗口感知策略做 multi-seed 验证。
- 优化 YCSB medium 边界，使其不输给 MVCC/TicToc。
- 将 TPCC 窗口阈值从手工规则推进到在线自适应策略。
- 继续把多分支事务语义、语义感知并发控制、成本感知提交协议和 ATCC 策略表做成可插拔模块。

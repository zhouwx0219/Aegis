# ATCC 汇报

## 1. 汇报主题

本次汇报的主题是：面向 Data Agent System 的自适应并发控制策略。

我们不是直接复用传统数据库的 TPCC/YCSB benchmark，而是把 TPCC/YCSB 改造成 agent-style workload：一个事务任务不再是固定的一条事务逻辑，而是包含多候选计划、分阶段执行、提交前选择、失败后重试和 agent 侧额外成本。

核心问题是：

在这种 data agent 场景下，传统并发控制协议是否仍然合适？ATCC 策略表能否帮助系统在不同冲突强度下自适应选择并发控制策略？

## 2. Data Agent System 架构

系统分成两层。

底层是版本化 KV 存储。它只负责对象版本、读写、条件提交、冲突检测和原子更新。底层不理解 TPCC/YCSB 的业务语义，也不决定哪个任务该用哪种并发控制。

上层是 agent runtime。它负责事务语义和并发控制：

- 生成 agent task；
- 生成多个候选执行计划；
- 按阶段执行 explore、refine、commit；
- 根据并发控制策略提交；
- 在失败时重试、刷新版本或重新生成计划；
- 统计冲突、中止、锁等待、尾延迟和 token 浪费。

这种结构的目标是把 data agent system 做成可扩展框架：agent 侧负责事务语义和策略选择，底层只提供版本化 KV 存储。

## 3. 自适应混合策略是什么

自适应混合策略不是一个新的底层并发控制协议，而是一个策略族选择器。

它做的事情是：每个 agent 事务任务来了以后，先判断它更适合哪类并发控制，然后在下面这些策略之间选择：

- OCC；
- 2PL-nowait；
- 2PL-wait-die；
- MVCC；
- Silo 类协议；
- TicToc 类协议；
- 操作级 ATCC。

这里要区分两个概念。

操作级 ATCC 是细粒度策略。它按每个操作判断是否乐观执行、是否提前加锁、是否提交前刷新。

自适应混合策略是粗粒度策略。它按一个 task 或一个 task window 判断整个任务应该走哪一类协议。

为什么需要这两层？因为 ATCC 在高冲突下能显著减少中止和重试，但在低冲突下提前加锁和刷新会有固定开销。因此系统不能无脑使用 ATCC，而应该：

- 低冲突：走轻量传统快路径；
- 中高冲突：走操作级 ATCC；
- 读多场景：优先选择 MVCC/TicToc 等读友好协议；
- 热点写场景：使用 ATCC 保护关键写对象。

## 4. 公共 Agent Workload 模型

我们的 TPCC/YCSB 都基于同一个 agent workload 抽象。

### 4.1 对象

对象是底层版本化 KV 里的 key-value item。每个对象包含：

- object id；
- 初始值；
- 对象类型；
- 元数据。

对象类型会影响策略判断。例如字段值、计数器、追加流和状态行的冲突语义不同。

### 4.2 操作

当前支持五类操作：

- read：读取对象；
- overwrite：覆盖写；
- append：向追加流写入新片段；
- delta：对计数器做增量或减量；
- cas：比较后写入。

### 4.3 候选计划

一个 agent task 有多个 ranked candidates。每个 candidate 是一个可能的执行计划，有质量分数。

这模拟 agent 的真实行为：agent 往往不是直接生成唯一事务，而是生成多个可能方案，再选择质量最高的方案提交。

### 4.4 工作流阶段

一个 task 被拆成多个阶段：

- explore：探索，读取一部分上下文；
- refine：细化，读取更多上下文或修正计划；
- commit：提交，执行真正写入。

这比传统事务更接近 agent：agent 会先读、再思考和生成计划，最后提交。并发冲突往往发生在这些阶段之间。

## 5. YCSB 工作负载构造

YCSB 的对象是字段级 KV：

```text
ycsb:record:{record}:field:{field}
```

值类型是字符串字段值，初始值是 `"0"`。

YCSB 操作包括：

- read：读取字段；
- overwrite：覆盖字段值。

每个 YCSB task 会生成多个候选计划。每个候选访问固定数量的 record/field。访问分布由读写比例、Zipf 偏斜和热点访问概率控制。

YCSB workflow：

1. explore：执行前一部分 read；
2. refine：执行剩余 read；
3. commit：执行 overwrite。

YCSB 三档冲突配置：

| 档位 | records | fields | requests/task | candidates | 读写比例 | Zipf | 热点 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| low | 512 | 10 | 10 | 3 | 读 95%，写 5% | 0.0 | 无热点 |
| medium | 128 | 10 | 10 | 3 | 读 90%，写 10% | 0.7 | 10% 记录承载 50% 访问 |
| high | 64 | 10 | 10 | 3 | 读 50%，写 50% | 0.99 | 10% 记录承载 75% 访问 |

传统 YCSB 是一次请求直接执行；我们的 YCSB 是 agent task，有多候选、分阶段、热点写和重试成本。

## 6. TPCC 工作负载构造

TPCC 保留业务对象，但映射成版本化 KV。

主要 key/value 设计：

| 对象 | key | 值类型 |
| --- | --- | --- |
| 仓库年度销售额 | `tpcc:warehouse:{w}:ytd` | counter |
| 地区年度销售额 | `tpcc:district:{w}:{d}:ytd` | counter |
| 地区下一个订单号 | `tpcc:district:{w}:{d}:next_order_id` | counter |
| 地区订单流 | `tpcc:district:{w}:{d}:orders` | append stream |
| 地区历史流 | `tpcc:district:{w}:{d}:history` | append stream |
| 客户余额 | `tpcc:customer:{w}:{d}:{c}:balance` | counter |
| 客户付款次数 | `tpcc:customer:{w}:{d}:{c}:payment_count` | counter |
| 客户状态 | `tpcc:customer:{w}:{d}:{c}:status` | status row |
| 库存数量 | `tpcc:stock:{w}:{item}:quantity` | constrained counter |
| 库存年度销售量 | `tpcc:stock:{w}:{item}:ytd` | counter |

TPCC 操作包括：

- delta：计数器增量；
- constrained delta：库存扣减，不能低于 0；
- append：订单流和历史流追加；
- cas：客户状态比较后写入；
- read：库存检查和订单状态读取。

本轮实验主要使用 NewOrder，因为它最能体现多对象写冲突和库存约束。

NewOrder workflow：

1. explore：读取前半库存数量；
2. refine：读取后半库存数量；
3. commit：增加地区订单号、追加订单、扣减库存、增加库存销售量。

TPCC 三档冲突配置：

| 档位 | 仓库数 | 每仓库地区数 | 每地区客户数 | 商品数 | 每订单行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| low | 8 | 5 | 100 | 500 | 5 |
| medium | 2 | 3 | 60 | 200 | 8 |
| high | 1 | 2 | 40 | 100 | 10 |

low 中订单分散到更多仓库和地区；high 中大量事务集中在少数地区和库存对象。

## 7. ATCC 怎么训练

ATCC 训练的目标不是训练一个黑盒模型，而是训练一张可解释的策略表。

训练流程：

1. 固定 workload profile，例如 YCSB high 或 TPCC NewOrder。
2. 用固定 seed 生成 agent tasks。
3. 运行多个 episode 或搜索候选策略。
4. 收集反馈：
   - commit rate；
   - conflict aborts；
   - attempts/task；
   - lock wait；
   - p99 latency；
   - wasted tokens；
   - selected strategy counts。
5. 将反馈归入策略状态。
6. 更新或选择策略表。
7. 输出策略 artifact。

策略状态包括：

- workload 类型；
- task 类型；
- object class；
- access kind；
- operation intent；
- agent phase；
- candidate 数量；
- write ratio；
- retry 信息；
- runtime lock/latency 信号。

## 8. ATCC 怎么应用

运行时分两层。

第一层是自适应混合策略。它先选择整个任务或窗口应该走哪个策略族。

第二层是操作级 ATCC。进入操作级 ATCC 后，它再对每个操作判断是否需要提前保护。

YCSB 策略：

- low：避免重型 ATCC，走传统快路径；
- medium：读多场景优先考虑 MVCC/TicToc；
- high：热点写进入操作级 ATCC。

TPCC 策略：

- low：统计窗口里不同 `next_order_id` 写目标数量。如果数量多，说明冲突分散，走 OCC；
- medium/high：订单号和库存对象竞争集中，走操作级 ATCC。

TPCC low 中，自适应策略选择 OCC 后比普通 OCC 快 `1.116x`。这个结果不能解释成 OCC 算法本身变快。两者 commit rate 都是 100%，attempts/task 都是 1.00，conflict aborts 都是 0。差异主要来自本轮自适应策略开启了已选策略快速直通，并且单 seed 下普通 OCC 的 p99 更高。因此更稳妥的表述是：low 下不退化，并在本次单 seed 下略优。

## 9. 实验结果

### 9.1 YCSB

YCSB 使用自适应混合策略对比传统 CC。

| YCSB 档位 | vs OCC | vs 2PL-nowait | vs 2PL-wait-die | vs MVCC | vs Silo | vs TicToc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 1.016x | 1.286x | 1.319x | 1.523x | 1.647x | 1.259x |
| medium | 1.514x | 2.333x | 2.559x | 0.901x | 2.251x | 0.826x |
| high | 5.819x | 18.809x | 17.444x | 1.766x | 11.504x | 1.358x |

结论：

- high 冲突下收益明显；
- low 没有退化；
- medium 对 OCC、2PL、Silo 有明显提升，但还输给 MVCC 和 TicToc，说明中冲突边界还需要优化。

### 9.2 TPCC

TPCC 使用窗口感知策略表。low 走 OCC，medium/high 走操作级 ATCC。

| TPCC 档位 | 实际选择 | 吞吐 | 提交率 | vs OCC | vs 2PL-nowait | vs 2PL-wait-die | vs MVCC | vs Silo | vs TicToc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | OCC | 5.415 | 100% | 1.116x | 1.395x | 1.429x | 1.205x | 1.365x | 1.290x |
| medium | 操作级 ATCC | 20.106 | 100% | 2.154x | 3.123x | 4.465x | 2.377x | 2.697x | 1.840x |
| high | 操作级 ATCC | 23.930 | 100% | 24.480x | 90.426x | 92.134x | 186.260x | 43.747x | 41.213x |

结论：

- TPCC high 之前失败是路由错误：系统把 high 路由到 TicToc，导致提交率接近 0。
- 训练 TPCC 专用策略表后，medium/high 回到操作级 ATCC。
- 加入窗口级低冲突规则后，low 不再被 ATCC 固定开销拖垮。

## 10. 原始数据在哪里看

YCSB 原始数据：

```text
results/atcc_matrix_ycsb_20260628_11/summary.csv
results/atcc_matrix_ycsb_20260628_11/ycsb-low.json
results/atcc_matrix_ycsb_20260628_11/ycsb-medium.json
results/atcc_matrix_ycsb_20260628_11/ycsb-high.json
```

TPCC 原始数据：

```text
results/atcc_matrix_tpcc_window_fast_20260628_11/summary.csv
results/atcc_matrix_tpcc_window_fast_20260628_11/tpcc-low.json
results/atcc_matrix_tpcc_window_fast_20260628_11/tpcc-medium.json
results/atcc_matrix_tpcc_window_fast_20260628_11/tpcc-high.json
```

TPCC 策略训练结果：

```text
results/atcc_tpcc_family_search_20260628_11/tpcc-family-search.json
results/atcc_tpcc_family_search_20260628_11/tpcc-family-policy.json
results/atcc_tpcc_family_search_20260628_11/tpcc-family-policy-window.json
```

查看 summary：

```powershell
Import-Csv results\atcc_matrix_ycsb_20260628_11\summary.csv
Import-Csv results\atcc_matrix_tpcc_window_fast_20260628_11\summary.csv
```

查看单个 JSON 的策略聚合：

```powershell
wsl bash -lc "cd /mnt/z/Data-Agent-System-master/cast-das && python3 - <<'PY'
import json
from pathlib import Path
for path in [
    'results/atcc_matrix_ycsb_20260628_11/ycsb-high.json',
    'results/atcc_matrix_tpcc_window_fast_20260628_11/tpcc-high.json',
]:
    data = json.loads(Path(path).read_text())
    print('\\n', path)
    for row in data['aggregates']:
        print(row['strategy'], row['committed_throughput'], row['commit_rate'], row.get('selected_strategy_counts'))
PY"
```

## 11. 如何复现

### 11.1 运行 YCSB full 矩阵

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ycsb_compare.ps1 `
  -Profile all `
  -StrategySet full `
  -TaskCount 40 `
  -Workers 24 `
  -OutputDir results/atcc_matrix_ycsb_reproduce `
  -PolicyArtifact results/atcc_interleave_all_adaptive_readheavy_20260628_08/adaptive-readheavy-family-policy.json
```

### 11.2 训练 TPCC 专用 family 策略

```powershell
wsl bash -lc "cd /mnt/z/Data-Agent-System-master/cast-das && mkdir -p results/atcc_tpcc_family_search_reproduce && python3 -m agent.evaluation.atcc_retry_experiment \
  --mode family-search \
  --family-search-profiles tpcc-medium,tpcc-high \
  --family-search-read-heavy-strategies tictoc-full \
  --family-search-cold-read-heavy-strategies tictoc-full \
  --family-search-hot-write-strategies adaptive-op-strict \
  --family-search-fallback-strategies tictoc-full,adaptive-op-strict,silo-full,mvcc-full \
  --family-search-prelock-wait-budget-ms-values 70 \
  --family-search-prelock-lease-modes hold,yield-refresh-regenerate \
  --family-search-agent-execution-modes staged \
  --family-search-snapshot-timings before-planning \
  --family-search-object-lock-schedulers bounded-priority \
  --family-search-baseline-strategies occ,2pl-nowait,2pl-wait-die,mvcc-full,silo-full,tictoc-full,adaptive-op-strict \
  --family-search-score-mode baseline-balanced \
  --workload tpcc \
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
  --prelock-lease-mode hold \
  --agent-execution-mode staged \
  --snapshot-timing before-planning \
  --family-policy-output results/atcc_tpcc_family_search_reproduce/tpcc-family-policy.json \
  --output results/atcc_tpcc_family_search_reproduce/tpcc-family-search.json"
```

窗口感知策略表需要包含下面两个字段：

```json
{
  "tpcc_low_contention_min_distinct_order_counters": 8,
  "tpcc_low_contention_strategy": "occ"
}
```

当前正式使用的策略表是：

```text
results/atcc_tpcc_family_search_20260628_11/tpcc-family-policy-window.json
```

### 11.3 运行 TPCC full 矩阵

TPCC 最终矩阵需要开启已选策略快速直通。下面给出 high 档示例；low/medium 只需要替换 profile 参数。

```powershell
wsl bash -lc "cd /mnt/z/Data-Agent-System-master/cast-das && python3 -m agent.evaluation.atcc_retry_experiment \
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
  --policy-artifact results/atcc_tpcc_family_search_20260628_11/tpcc-family-policy-window.json \
  --hybrid-selected-fast-through \
  --transaction-mix new_order:1.0 \
  --warehouses 1 \
  --districts-per-warehouse 2 \
  --customers-per-district 40 \
  --items 100 \
  --order-lines 10 \
  --output results/atcc_matrix_tpcc_reproduce/tpcc-high.json"
```

TPCC low 参数：

```text
--warehouses 8 --districts-per-warehouse 5 --customers-per-district 100 --items 500 --order-lines 5
```

TPCC medium 参数：

```text
--warehouses 2 --districts-per-warehouse 3 --customers-per-district 60 --items 200 --order-lines 8
```

TPCC high 参数：

```text
--warehouses 1 --districts-per-warehouse 2 --customers-per-district 40 --items 100 --order-lines 10
```

生成 summary：

```powershell
python .\scripts\summarize_retry_results.py --input-dir .\results\atcc_matrix_tpcc_reproduce
```

## 12. 局限和下一步

当前结果有几个限制。

第一，TPCC 最新恢复结果还是单 seed、单 repeat。它可以说明策略方向有效，但还不能作为最终统计主表。

第二，YCSB medium 还没有打过 MVCC/TicToc，说明中冲突边界策略还需要优化。

第三，TPCC high 的倍数很大，是因为传统协议提交率很低。汇报时必须同时报告 commit rate、attempts/task 和 p99。

下一步：

1. 对 TPCC 窗口感知策略做 multi-seed 验证。
2. 对 YCSB medium 做更细的边界策略训练。
3. 将窗口阈值从手工阈值推进到在线自适应选择。
4. 继续推进模块化：多分支事务语义、语义感知并发控制、成本感知提交、ATCC 策略表都做成可插拔模块。


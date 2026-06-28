# 事务级 ATCC 汇报

## 1. 概述

把 ATCC 从原来的“操作粒度策略表”推进到更接近论文的“事务级状态表 + 阶段级动作 + hot/cold read/write set 保护”，并验证它在 Data Agent System 上是否能带来稳定收益。

我们的创新点不是把 ATCC 直接复刻到传统数据库，而是把它应用到 Data Agent System。这个系统里，一个事务不是固定 SQL 模板，而是 agent task：它有多候选计划、explore/refine/commit 阶段、长思考延迟、失败后重试和 token/latency 成本。因此，ATCC 应该看到事务上下文和 agent workflow，而不是只对单条读写操作做局部判断。

## 2. Data Agent System 架构

系统分成两层：

- 底层：版本化 KV 存储。只负责对象版本、条件写入、原子提交和基础冲突检测。
- 上层：Agent Runtime。负责事务语义、候选计划、并发控制策略选择、预锁、刷新、重试和提交协议。

一次 agent task 的执行流程是：

```text
AgentTask
  -> ranked candidates
  -> explore/refine/commit stages
  -> AgentTransactionManager
  -> CC Registry
  -> ATCC policy table / traditional CC
  -> CostAwareCommitProtocol
  -> Versioned KV Store
```

这保证了底层 KV 不需要理解 TPCC/YCSB 语义，ATCC 和传统 CC 都作为上层可插拔策略存在。

## 3. 为什么要从操作级 ATCC 改到事务级 ATCC

之前的 ATCC 是 operation-level ATCC。它会为每个 `AgentOperation` 判断：

- 这个操作是否 optimistic；
- 这个操作是否 pessimistic；
- 是否需要 pre-snapshot lock；
- 是否需要在 commit 前刷新。

这个设计在 YCSB/TPCC high 上已经有明显收益，但和论文 ATCC 仍有差距。论文式 ATCC 更强调事务上下文：事务阶段、读写集合、热点集合、重试次数、全局 abort/lock wait 信号和优先级。

因此本轮新增 `transaction-atcc-strict`，它先汇总整个事务，再选一个事务级动作。

## 4. 事务级 ATCC 设计

### 4.1 事务级状态表

事务级状态键包含：

- workload；
- task type；
- agent phase；
- read count；
- write count；
- hot read count；
- hot write count；
- retry count；
- agent interval；
- priority bucket；
- global abort bucket；
- global lock wait bucket。

示例：

```text
scope=transaction
workload=agent-tpcc-semantic
task=new_order
phase=commit
reads=0
writes=17+
hotR=0
hotW=1
retry=0
globalAbort=0
globalLockWait=0ms
```

### 4.2 阶段级动作

当前动作集合和论文式 ATCC 的锁范围思想对齐：


| 动作                    | 含义                             |
| --------------------- | ------------------------------ |
| `occ`                 | 不提前加锁，走 OCC fast path          |
| `lock-hot-writes`     | 只锁 hot write set               |
| `lock-hot-read-write` | 锁 hot read set + hot write set |
| `lock-write-set`      | 锁完整 write set                  |
| `lock-read-write-set` | 锁完整 read set + write set       |


### 4.3 hot/cold read/write set

事务级 ATCC 会把事务访问集合分成：

- `hot_read_set`
- `hot_write_set`
- `cold_read_set`
- `cold_write_set`

当前 hot 判断规则：

- YCSB：`record_id < hot_record_count` 的字段是热点对象；
- TPCC：`tpcc:district:{w}:{d}:next_order_id` 是 NewOrder 的核心热点写对象；
- retry 次数较高时，事务内对象风险提高。

### 4.4 OCC fast path

低冲突场景不能强行走 ATCC，否则会被固定开销拖慢。因此事务级 ATCC 保留 OCC fast path。

触发条件：

- prior action 是 `occ`；
- `retry_count == 0`；
- 没有 hot read/write；
- 全局 abort rate 小于 0.05；
- 全局 lock wait 小于 10ms；
- agent interval 较短，或者不是 commit 阶段。

## 5. 工作负载构造

### 5.1 YCSB

YCSB 对象是字段级 KV：

```text
ycsb:record:{record}:field:{field}
```

操作：

- `read`
- `overwrite`

workflow：

```text
explore: 读取部分字段
refine: 读取剩余字段
commit: 执行 overwrite
```

三档冲突：


| profile | records | fields | requests/task | candidates | read/update | Zipf | hotspot         |
| ------- | ------- | ------ | ------------- | ---------- | ----------- | ---- | --------------- |
| low     | 512     | 10     | 10            | 3          | 95% / 5%    | 0.0  | 无热点             |
| medium  | 128     | 10     | 10            | 3          | 90% / 10%   | 0.7  | 10% 记录承载 50% 访问 |
| high    | 64      | 10     | 10            | 3          | 50% / 50%   | 0.99 | 10% 记录承载 75% 访问 |




### 5.2 TPCC

TPCC 被映射成版本化 KV：


| 对象       | key                                   | 类型                  |
| -------- | ------------------------------------- | ------------------- |
| 地区下一个订单号 | `tpcc:district:{w}:{d}:next_order_id` | counter             |
| 地区订单流    | `tpcc:district:{w}:{d}:orders`        | append stream       |
| 库存数量     | `tpcc:stock:{w}:{item}:quantity`      | constrained counter |
| 库存销售量    | `tpcc:stock:{w}:{item}:ytd`           | counter             |


本轮核心使用 NewOrder：

```text
explore: 读取前半库存
refine: 读取后半库存
commit: 增加订单号、追加订单、扣减库存、增加库存 ytd
```

三档冲突：


| profile | warehouses | districts/warehouse | customers/district | items | order lines |
| ------- | ---------- | ------------------- | ------------------ | ----- | ----------- |
| low     | 8          | 5                   | 100                | 500   | 5           |
| medium  | 2          | 3                   | 60                 | 200   | 8           |
| high    | 1          | 2                   | 40                 | 100   | 10          |


## 6. 实验设置

完整矩阵使用以下策略：

```text
occ
2pl-nowait
2pl-wait-die
mvcc-full
silo-full
tictoc-full
adaptive-op-strict
transaction-atcc-strict
```

通用运行参数：

- task count: 60
- workers: 24
- agent slots: 4
- planning delay: lognormal，均值 50ms，CV 0.8，最大 500ms
- max attempts: 8
- background workers: 4
- background strategy: OCC
- object lock scheduler: bounded-priority
- prelock lease mode: `yield-refresh-regenerate`
- execution mode: staged
- snapshot timing: before-planning

## 7. 完整矩阵结果

### 7.1 YCSB


| profile | transaction ATCC 吞吐 | 提交率    | attempts/task | vs OCC  | vs MVCC | vs TicToc | vs operation ATCC |
| ------- | ------------------- | ------ | ------------- | ------- | ------- | --------- | ----------------- |
| low     | 24.012              | 100.0% | 1.050         | 0.803x  | 0.850x  | 0.883x    | 0.918x            |
| medium  | 36.399              | 100.0% | 1.000         | 3.833x  | 1.022x  | 1.001x    | 1.032x            |
| high    | 50.500              | 100.0% | 1.000         | 45.740x | 2.204x  | 1.827x    | 0.980x            |


结论：

- YCSB low 不应该强行走事务级 ATCC，OCC/MVCC/TicToc 更好。
- YCSB medium 事务级 ATCC 略优于 MVCC、TicToc 和旧操作级 ATCC。
- YCSB high 事务级 ATCC 和旧操作级 ATCC 接近，远好于传统 CC。

### 7.2 TPCC


| profile | transaction ATCC 吞吐 | 提交率    | attempts/task | vs OCC           | vs MVCC          | vs TicToc        | vs operation ATCC |
| ------- | ------------------- | ------ | ------------- | ---------------- | ---------------- | ---------------- | ----------------- |
| low     | 4.134               | 100.0% | 1.017         | 0.580x           | 0.718x           | 0.751x           | 1.064x            |
| medium  | 23.242              | 100.0% | 1.100         | 3.979x           | 4.383x           | 2.972x           | 0.894x            |
| high    | 40.609              | 100.0% | 1.017         | 传统协议提交率极低，约有千倍提升 | 传统协议提交率极低，约有千倍提升 | 传统协议提交率极低，约有千倍提升 | 2.758x            |


结论：

- TPCC low 仍应走 OCC。事务级 ATCC 会提前保护 `next_order_id`，低冲突下锁等待成本超过收益。
- TPCC medium 事务级 ATCC 明显优于传统 CC，但略低于旧操作级 ATCC。
- TPCC high 是事务级 ATCC 最有价值的场景。



### 7.3 全量 CC 吞吐、延迟、提交率矩阵

下面三组矩阵给出 high/medium/low 冲突等级下所有 CC 的完整对比。吞吐使用 committed throughput；延迟使用 agent task p99 latency，单位是秒。

#### YCSB 吞吐


| CC           | low    | medium | high   |
| ------------ | ------ | ------ | ------ |
| OCC          | 29.913 | 9.495  | 1.104  |
| 2PL-nowait   | 26.154 | 5.779  | 0.996  |
| 2PL-wait-die | 24.255 | 5.617  | 0.563  |
| MVCC         | 28.264 | 35.627 | 22.911 |
| Silo         | 22.983 | 8.669  | 1.064  |
| TicToc       | 27.188 | 36.373 | 27.639 |
| 操作级 ATCC     | 26.147 | 35.261 | 51.506 |
| 事务级 ATCC     | 24.012 | 36.399 | 50.500 |




#### YCSB p99 延迟，单位：秒


| CC           | low   | medium | high  |
| ------------ | ----- | ------ | ----- |
| OCC          | 1.706 | 5.044  | 6.946 |
| 2PL-nowait   | 1.996 | 6.393  | 7.677 |
| 2PL-wait-die | 2.160 | 7.182  | 8.699 |
| MVCC         | 1.885 | 1.213  | 2.373 |
| Silo         | 2.146 | 5.385  | 8.176 |
| TicToc       | 1.991 | 1.446  | 1.931 |
| 操作级 ATCC     | 2.026 | 1.162  | 0.949 |
| 事务级 ATCC     | 2.158 | 1.247  | 0.948 |




#### YCSB 提交率


| CC           | low    | medium | high   |
| ------------ | ------ | ------ | ------ |
| OCC          | 100.0% | 83.3%  | 13.3%  |
| 2PL-nowait   | 100.0% | 65.0%  | 13.3%  |
| 2PL-wait-die | 100.0% | 73.3%  | 8.3%   |
| MVCC         | 100.0% | 100.0% | 98.3%  |
| Silo         | 100.0% | 86.7%  | 15.0%  |
| TicToc       | 100.0% | 100.0% | 98.3%  |
| 操作级 ATCC     | 100.0% | 100.0% | 100.0% |
| 事务级 ATCC     | 100.0% | 100.0% | 100.0% |


YCSB 的核心观察是：low 下传统轻量协议更合适；medium 下事务级 ATCC、TicToc、MVCC 非常接近；high 下两种 ATCC 明显优于传统协议，且 p99 延迟最低。

#### TPCC 吞吐


| CC           | low   | medium | high   |
| ------------ | ----- | ------ | ------ |
| OCC          | 7.133 | 5.842  | 0.000  |
| 2PL-nowait   | 6.067 | 4.772  | 0.000  |
| 2PL-wait-die | 5.714 | 5.883  | 0.000  |
| MVCC         | 5.755 | 5.303  | 0.000  |
| Silo         | 5.273 | 6.449  | 0.000  |
| TicToc       | 5.506 | 7.821  | 0.000  |
| 操作级 ATCC     | 3.885 | 26.011 | 14.726 |
| 事务级 ATCC     | 4.134 | 23.242 | 40.609 |




#### TPCC p99 延迟，单位：秒


| CC           | low    | medium | high  |
| ------------ | ------ | ------ | ----- |
| OCC          | 7.800  | 7.208  | 8.481 |
| 2PL-nowait   | 9.243  | 8.151  | 8.594 |
| 2PL-wait-die | 9.788  | 7.428  | 9.285 |
| MVCC         | 9.854  | 7.446  | 8.519 |
| Silo         | 10.745 | 7.092  | 8.947 |
| TicToc       | 10.204 | 6.177  | 9.056 |
| 操作级 ATCC     | 14.597 | 1.974  | 3.794 |
| 事务级 ATCC     | 13.709 | 2.203  | 1.246 |


#### TPCC 提交率


| CC           | low    | medium | high   |
| ------------ | ------ | ------ | ------ |
| OCC          | 100.0% | 73.3%  | 0.0%   |
| 2PL-nowait   | 100.0% | 68.3%  | 0.0%   |
| 2PL-wait-die | 100.0% | 76.7%  | 0.0%   |
| MVCC         | 100.0% | 70.0%  | 0.0%   |
| Silo         | 100.0% | 81.7%  | 0.0%   |
| TicToc       | 100.0% | 86.7%  | 0.0%   |
| 操作级 ATCC     | 100.0% | 100.0% | 100.0% |
| 事务级 ATCC     | 100.0% | 100.0% | 100.0% |


TPCC 的核心观察是：low 下 OCC 吞吐最好，ATCC 因提前保护 `next_order_id` 付出锁等待成本；medium 下两种 ATCC 都把提交率拉到 100%，操作级 ATCC 吞吐更高；high 下传统 CC 全部 0 提交，事务级 ATCC 的 p99 延迟和吞吐都明显优于操作级 ATCC。

## 8. TPCC-high 稳定性验证

为了验证 TPCC-high 的单 seed 结果是否稳定，追加跑了 5 个 seed：

```text
920104, 920105, 920106, 920107, 920108
```



### 8.1 multi-seed 均值


| 策略                     | 平均吞吐   | 标准差    | 平均提交率  | 0 提交 seed 数 | transaction ATCC 相对倍数 |
| ---------------------- | ------ | ------ | ------ | ----------- | --------------------- |
| OCC                    | 0.000  | 0.000  | 0.0%   | 5/5         | 无穷大/不可定义              |
| 2PL-nowait             | 0.000  | 0.000  | 0.0%   | 5/5         | 无穷大/不可定义              |
| 2PL-wait-die           | 0.000  | 0.000  | 0.0%   | 5/5         | 无穷大/不可定义              |
| MVCC                   | 0.000  | 0.000  | 0.0%   | 5/5         | 无穷大/不可定义              |
| Silo                   | 0.000  | 0.000  | 0.0%   | 5/5         | 无穷大/不可定义              |
| TicToc                 | 0.021  | 0.048  | 0.3%   | 4/5         | 1984.230x             |
| operation-level ATCC   | 21.813 | 12.289 | 100.0% | 0/5         | 1.937x                |
| transaction-level ATCC | 42.246 | 2.425  | 100.0% | 0/5         | 1.000x                |


### 8.2 per-seed 对比


| seed   | operation ATCC 吞吐 | transaction ATCC 吞吐 | transaction / operation |
| ------ | ----------------- | ------------------- | ----------------------- |
| 920104 | 15.542            | 41.090              | 2.644x                  |
| 920105 | 14.215            | 38.966              | 2.741x                  |
| 920106 | 42.946            | 42.179              | 0.982x                  |
| 920107 | 14.057            | 43.696              | 3.109x                  |
| 920108 | 22.305            | 45.300              | 2.031x                  |


### 8.3 稳定结论

TPCC-high 下，传统 CC 在这个 agent workload 上基本失效：

- OCC、2PL、MVCC、Silo 在 5 个 seed 中全部 0 提交；
- TicToc 只有 1 个 seed 有极少提交，平均提交率 0.3%，平均吞吐 0.021；
- transaction-level ATCC 在 5 个 seed 中全部 100% 提交，平均吞吐 42.246。

因此，TPCC-high 的数量级结论是：

- 相对传统 CC：事务级 ATCC 是“从基本 0 有效提交到稳定 100% 提交”，相对唯一非零的 TicToc 平均吞吐约 1984x，属于千倍级提升；对 OCC/2PL/MVCC/Silo 因 baseline 为 0，不能给有限倍数，只能表述为传统协议在该设置下失效。
- 相对旧 operation-level ATCC：事务级 ATCC 平均吞吐 1.937x，约 2 倍；更重要的是标准差更小，吞吐更稳定。

## 9. 为什么事务级 ATCC 在 TPCC-high 更稳

TPCC-high 的冲突核心是少量 district 的 `next_order_id` 和大量 NewOrder commit 阶段写入。事务级 ATCC 在 commit 阶段直接看到完整事务上下文：

- 当前是 NewOrder commit；
- write set 很宽；
- hot write set 包含 `next_order_id`；
- 全局传统协议 abort 压力高；
- agent task 有长延迟，失败重试成本很高。

因此它会选择 `lock-hot-writes`，只保护核心热写对象。这个动作比完整 2PL 轻，比 OCC/MVCC/TicToc 更能避免最后验证失败，也比操作级 ATCC 更贴近论文里的“事务上下文 + 阶段动作”。


# 操作级 ATCC 和事务级 ATCC 的区别

## 1. 两种 ATCC 分别是什么

当前项目里有两种 ATCC：

- 操作级 ATCC：`adaptive-op-strict`
- 事务级 ATCC：`transaction-atcc-strict`

它们都属于 agent runtime 上层的并发控制策略，都复用版本化 KV、ObjectLockTable、commit feedback 和 staged execution。但它们的决策粒度不同。

```text
操作级 ATCC：判断某个操作该 optimistic 还是 pessimistic。
事务级 ATCC：判断整个事务在当前阶段该采取什么 ATCC 动作。
```

## 2. 决策粒度不同

### 2.1 操作级 ATCC

操作级 ATCC 的输入是一组 `OperationPolicyProfile`，每个 profile 对应一个对象访问：

```text
object_id
access_kind = read/write
intent_name = read/overwrite/delta/append
task_type
workload
agent_phase
retry_count
object statistics
```

它输出每个对象的策略：

```text
object A -> optimistic
object B -> pessimistic
object C -> optimistic
```

例子：

```text
tpcc:district:0:1:next_order_id -> pessimistic
tpcc:district:0:1:orders -> optimistic
tpcc:stock:0:8:quantity -> optimistic
```

### 2.2 事务级 ATCC

事务级 ATCC 的输入是整个事务的汇总状态：

```text
workload
task_type
agent_phase
read_set
write_set
hot_read_set
hot_write_set
cold_read_set
cold_write_set
retry_count
global_abort_rate
global_lock_wait
```

它先选择一个事务级动作：

```text
occ
lock-hot-writes
lock-hot-read-write
lock-write-set
lock-read-write-set
```

再把这个动作翻译成具体锁对象。

例子：

```text
事务状态：NewOrder commit，hot_write_set = {next_order_id}
事务动作：lock-hot-writes
锁对象：next_order_id
```

## 3. 状态表不同

### 3.1 操作级 ATCC 状态

操作级状态更偏局部：

```text
workload
task_type
access_kind
intent_name
object_class
operation_count_for_object
total_writes
agent_phase
retry_count
object-level conflict telemetry
```

它适合回答：

```text
这个对象最近冲突多不多？
这个写操作是不是热点？
这个 read 是否需要保护？
```

### 3.2 事务级 ATCC 状态

事务级状态更偏全局：

```text
workload = agent-tpcc-semantic
task = new_order
phase = commit
reads = bucket
writes = bucket
hotR = bucket
hotW = bucket
retry = bucket
priority = bucket
globalAbort = bucket
globalLockWait = bucket
```

它适合回答：

```text
当前事务是不是处于高风险 commit 阶段？
这个事务是否有 hot write set？
是否应该锁完整 write set，还是只锁 hot writes？
低冲突时是否应该直接走 OCC fast path？
```

## 4. 动作空间不同

### 4.1 操作级 ATCC 动作

操作级 ATCC 的核心动作是二分类：


| 动作          | 含义          |
| ----------- | ----------- |
| optimistic  | 不提前加锁，走乐观验证 |
| pessimistic | 对该对象提前加锁    |


虽然它也会记录 `atcc_action`，例如 `lock-hot-writes`，但最终落到每个对象上仍然是 optimistic/pessimistic。

### 4.2 事务级 ATCC 动作

事务级 ATCC 的动作是锁范围选择：


| 动作                    | 含义                             |
| --------------------- | ------------------------------ |
| `occ`                 | 整个事务走 OCC fast path            |
| `lock-hot-writes`     | 只锁 hot write set               |
| `lock-hot-read-write` | 锁 hot read set + hot write set |
| `lock-write-set`      | 锁完整 write set                  |
| `lock-read-write-set` | 锁完整 read set + write set       |


这更接近论文 ATCC 的形式：根据事务上下文选择一个动作，而不是孤立地处理每个操作。

## 5. 对 OCC fast path 的处理不同

### 5.1 操作级 ATCC

操作级 ATCC 可以让每个操作都 optimistic，但这不等于普通 OCC。

原因是它仍然要做：

- operation profile 构造；
- object class 判断；
- telemetry 查询；
- policy table 查询；
- per-operation decision 记录；
- staged prelock/refresh 逻辑判断。

所以低冲突场景下，“ATCC 内部全部 optimistic”仍然会有固定开销。

### 5.2 事务级 ATCC

事务级 ATCC 显式保留 OCC fast path。

低风险条件满足时，它直接输出：

```text
action = occ
fast_path = true
prelock_targets = empty
```

这解决的问题是：

```text
低冲突事务不要支付复杂 ATCC 决策和预锁成本。
```

但当前实现仍然需要外层 family selector 配合。比如 TPCC-low 里，整体直接选 OCC 仍然比事务级 ATCC 更好。

## 6. 对 TPCC NewOrder 的表现差异

### 6.1 操作级 ATCC

操作级 ATCC 会逐个判断对象：

```text
next_order_id -> pessimistic
orders -> optimistic
stock quantity -> optimistic
stock ytd -> optimistic
```

优点：

- 锁粒度细；
- YCSB high 和 TPCC medium 表现稳定；
- 不容易锁过多对象。

缺点：

- 缺少完整事务上下文；
- 对阶段和事务风险的表达不够直接；
- 更像工程上的 per-object 策略表，不完全像论文 ATCC。

### 6.2 事务级 ATCC

事务级 ATCC 会先看完整 NewOrder：

```text
phase = commit
write_set = wide
hot_write_set = next_order_id
task_type = new_order
```

然后选择：

```text
lock-hot-writes
```

优点：

- 更接近论文 ATCC；
- 能解释为“事务在 commit 阶段保护 hot write set”；
- TPCC-high multi-seed 下吞吐均值约为操作级 ATCC 的 1.937x；
- 吞吐标准差更小，稳定性更好。

缺点：

- 动作更粗，可能比操作级 ATCC 锁更多；
- YCSB high 下略低于操作级 ATCC；
- TPCC-low 下如果强行使用，仍然不如 OCC。

## 7. 实验结果对比

### 7.1 YCSB


| profile | 操作级 ATCC 吞吐 | 事务级 ATCC 吞吐 | 事务级 / 操作级 | 结论               |
| ------- | ----------- | ----------- | --------- | ---------------- |
| low     | 26.147      | 24.012      | 0.918x    | 事务级 ATCC 固定开销更明显 |
| medium  | 35.261      | 36.399      | 1.032x    | 事务级略优            |
| high    | 51.506      | 50.500      | 0.980x    | 两者接近，操作级略优       |




### 7.2 TPCC


| profile | 操作级 ATCC 吞吐 | 事务级 ATCC 吞吐 | 事务级 / 操作级 | 结论                      |
| ------- | ----------- | ----------- | --------- | ----------------------- |
| low     | 3.885       | 4.134       | 1.064x    | 两者都不如 OCC               |
| medium  | 26.011      | 23.242      | 0.894x    | 操作级更细，吞吐更高              |
| high    | 14.726      | 40.609      | 2.758x    | 事务级明显更好，需 multi-seed 验证 |




### 7.3 TPCC-high multi-seed


| 指标        | 操作级 ATCC | 事务级 ATCC |
| --------- | -------- | -------- |
| 平均吞吐      | 21.813   | 42.246   |
| 吞吐标准差     | 12.289   | 2.425    |
| 平均提交率     | 100.0%   | 100.0%   |
| 事务级 / 操作级 | -        | 1.937x   |


multi-seed 后的稳定结论：

```text
TPCC-high 下，事务级 ATCC 相对操作级 ATCC 约 2 倍吞吐提升。
相对传统 CC，事务级 ATCC 是从基本 0 有效提交到稳定 100% 提交。
```

## 8. 和论文 ATCC 的关系

论文 ATCC 更强调事务上下文和事务阶段动作，例如：

- 保持 OCC；
- 锁 hot write set；
- 锁 hot read/write set；
- 锁完整 write set；
- 根据冲突和代价调整优先级。

我们的操作级 ATCC 是一个工程上更细粒度的 per-object 版本，适合局部热点保护。

我们的事务级 ATCC 更接近论文形式：

```text
transaction state -> ATCC action -> lock scope -> feedback update
```

因此，如果要从“对标论文机制”的角度讲，事务级 ATCC 是更合适的主线；如果要从“某些 workload 的最优工程性能”角度讲，操作级 ATCC 仍然是重要 baseline。

## 9. 应该怎么使用两种 ATCC

推荐策略：


| 场景          | 推荐                            |
| ----------- | ----------------------------- |
| YCSB low    | OCC / TicToc / MVCC           |
| YCSB medium | 事务级 ATCC / TicToc / MVCC 都可比较 |
| YCSB high   | 操作级 ATCC 或事务级 ATCC            |
| TPCC low    | OCC                           |
| TPCC medium | 操作级 ATCC                      |
| TPCC high   | 事务级 ATCC                      |


最终系统不应该固定只用一种 ATCC，而应该让外层 family selector 判断：

```text
低冲突 -> OCC fast path
读多中冲突 -> MVCC/TicToc
局部热点写 -> 操作级 ATCC
强事务阶段热点 -> 事务级 ATCC
```

## 10. 总结

两种 ATCC 的区别不是“哪个绝对更好”，而是适用粒度不同：

- 操作级 ATCC：更细，适合 per-object 热点保护。
- 事务级 ATCC：更接近论文，适合解释 agent transaction 的阶段风险和事务级动作。

TPCC-high 证明事务级 ATCC 的价值最大：传统 CC 基本失效，操作级 ATCC 能恢复提交，事务级 ATCC 在 multi-seed 下进一步把平均吞吐提高到约 1.94 倍，并且稳定性更好。
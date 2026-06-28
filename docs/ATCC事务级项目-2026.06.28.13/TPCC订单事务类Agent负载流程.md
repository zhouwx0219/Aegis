# TPCC 订单事务的类 Agent 负载流程

## 1. TPCC NewOrder

TPCC NewOrder 是本项目里最适合说明类 agent 负载的事务，因为它同时具备三个特点：

- 有明确业务语义：创建订单、分配订单号、扣减库存、记录库存销售量。
- 有多对象写集合：一个订单会同时写 district、orders、stock quantity、stock ytd。
- 有天然热点：高冲突配置下，大量事务集中竞争少数 `next_order_id` 和库存对象。

传统 TPCC 通常是短事务：读数据、计算、提交。我们的 Data Agent System 里的 TPCC NewOrder 被改造成类 agent task：它有阶段、有候选计划、有 agent planning delay，有失败后的刷新、重放和重试成本。

## 2. 数据对象和 key/value 设计

TPCC 被展开成版本化 KV 对象。NewOrder 主要涉及这些 key：


| 对象       | key                                   | 值类型                 | 作用              |
| -------- | ------------------------------------- | ------------------- | --------------- |
| 地区下一个订单号 | `tpcc:district:{w}:{d}:next_order_id` | counter             | 为新订单分配 order id |
| 地区订单流    | `tpcc:district:{w}:{d}:orders`        | append stream       | 追加订单记录          |
| 库存数量     | `tpcc:stock:{w}:{item}:quantity`      | constrained counter | 扣减库存，不能低于 0     |
| 库存销售量    | `tpcc:stock:{w}:{item}:ytd`           | counter             | 累加该商品销售量        |


在 high profile 中，配置是：

```text
warehouses = 1
districts_per_warehouse = 2
customers_per_district = 40
items = 100
order_lines = 10
```

这会让大量 NewOrder 集中到很少的 warehouse/district 上，`next_order_id` 成为强热点。

## 3. AgentTask 结构

一个 TPCC NewOrder task 不是一条固定事务，而是一个 agent task：

```text
AgentTask
  workload = agent-tpcc-semantic
  task_type = new_order
  candidates = 多个候选订单计划
  context = agent 阶段、冲突 profile、运行参数
```

每个 candidate 是一个可能提交的订单计划。它包含一组操作：

```text
read stock quantity
delta district next_order_id +1
append district orders
delta stock quantity -order_quantity
delta stock ytd +order_quantity
```

候选计划带有质量分数。提交时 runtime 会选择合适候选，或者在语义允许时重选、刷新、重放。

## 4. 类 agent workflow

NewOrder 被拆成三个 agent 阶段：

```text
explore -> refine -> commit
```

### 4.1 explore 阶段

explore 阶段模拟 agent 初步查看上下文。

在 NewOrder 中，它会读取一部分库存对象：

```text
read tpcc:stock:{w}:{item_a}:quantity
read tpcc:stock:{w}:{item_b}:quantity
...
```

这个阶段只读，不真正写入订单。它扩大了事务读集合，也让事务从 snapshot 到 commit 的时间变长。

### 4.2 refine 阶段

refine 阶段模拟 agent 补充检查和修正计划。

它会继续读取剩余库存对象，完善候选订单计划：

```text
read tpcc:stock:{w}:{item_c}:quantity
read tpcc:stock:{w}:{item_d}:quantity
...
```

在传统短事务中，这些读取和提交之间间隔很短；在类 agent 负载中，explore/refine 阶段之间会有 planning delay，导致 snapshot 更容易变旧。

### 4.3 commit 阶段

commit 阶段执行真正写入：

```text
delta tpcc:district:{w}:{d}:next_order_id +1
append tpcc:district:{w}:{d}:orders
delta tpcc:stock:{w}:{item}:quantity -qty
delta tpcc:stock:{w}:{item}:ytd +qty
```

这个阶段是冲突最集中的地方。尤其是：

```text
tpcc:district:{w}:{d}:next_order_id
```

它决定新订单号，在 TPCC-high 下大量事务会抢同一个或少数几个 counter。

## 5. 一次事务的运行流程

下面是类 agent NewOrder 在 runtime 中的大体流程：

```text
1. 生成 AgentTask 和多个 candidate
2. 进入 AgentTransactionManager
3. CC registry 根据策略选择并发控制
4. 如果是 ATCC，先根据任务上下文生成 pre-snapshot lock plan
5. begin transaction，读取版本化 KV snapshot
6. 执行 explore 阶段
7. 执行 refine 阶段
8. 进入 commit 阶段前，根据策略可能重新加锁、刷新或重放
9. 提交候选计划
10. 如果提交失败，按 max_attempts 重试
```

对应到 staged execution：

```text
prepare_task_transaction(populate=False)
  -> populate_task_stage(txn, task, "explore")
  -> populate_task_stage(txn, task, "refine")
  -> populate_task_stage(txn, task, "commit")
  -> txn.commit(strategy)
```

## 6. 为什么传统 CC 在 TPCC-high 下容易 0 提交

TPCC-high 中，传统 CC 会遇到三个叠加问题。

第一，热点集中。

```text
warehouses = 1
districts = 2
workers = 24
background_workers = 4
```

大量事务竞争少量 `next_order_id`。

第二，agent workflow 拉长了冲突窗口。

```text
snapshot -> explore -> planning delay -> refine -> planning delay -> commit
```

事务拿到 snapshot 后不会马上提交。等它提交时，热点对象版本大概率已经被其他事务推进。

第三，传统 OCC/MVCC/TicToc/Silo 主要在提交时验证。

它们常见失败模式是：

```text
begin snapshot
read old versions
agent delay
commit validation fails
retry
repeat until max_attempts
```

在本轮 TPCC-high multi-seed 中，OCC、2PL、MVCC、Silo 在 5 个 seed 里全部 0 提交；TicToc 只有极少提交，平均提交率 0.3%。

## 7. ATCC 在这个流程中做了什么

### 7.1 操作级 ATCC

操作级 ATCC 会看每个 operation，判断它是否需要 pessimistic protection。

例如：

```text
tpcc:district:0:1:next_order_id -> pessimistic
tpcc:stock:0:42:quantity -> optimistic
tpcc:stock:0:42:ytd -> optimistic
```

它能把核心热点写保护起来，因此 TPCC-high 下能恢复 100% 提交率。

### 7.2 事务级 ATCC

事务级 ATCC 会先看整个事务：

```text
task_type = new_order
phase = commit
write_set = 很宽
hot_write_set = next_order_id
retry_count = 当前尝试次数
global_abort_rate = 当前运行时冲突压力
```

然后选择事务级动作：

```text
lock-hot-writes
```

也就是只保护 hot write set，而不是锁完整事务写集合。

## 8. TPCC NewOrder 的完整例子

假设某个 task 生成一个订单，包含 10 条 order line。

### 8.1 访问集合

读集合：

```text
tpcc:stock:0:8:quantity
tpcc:stock:0:19:quantity
tpcc:stock:0:33:quantity
...
```

写集合：

```text
tpcc:district:0:1:next_order_id
tpcc:district:0:1:orders
tpcc:stock:0:8:quantity
tpcc:stock:0:8:ytd
tpcc:stock:0:19:quantity
tpcc:stock:0:19:ytd
...
```

hot write set：

```text
tpcc:district:0:1:next_order_id
```

cold write set：

```text
tpcc:district:0:1:orders
tpcc:stock:0:8:quantity
tpcc:stock:0:8:ytd
...
```

### 8.2 事务级 ATCC 决策

事务级状态：

```text
workload = agent-tpcc-semantic
task = new_order
phase = commit
writes = 17+
hotW = 1
retry = 0
```

动作：

```text
lock-hot-writes
```

pre-snapshot lock targets：

```text
tpcc:district:0:1:next_order_id
```

### 8.3 提交效果

传统 CC：

```text
提交时发现 next_order_id 或库存版本已经变化
-> validation fail
-> retry
-> 高冲突下 8 次内仍可能失败
```

事务级 ATCC：

```text
提前保护 next_order_id
-> commit 阶段的核心热点写有序化
-> 冲突从提交失败变成可控锁等待
-> TPCC-high 下稳定 100% 提交
```

## 9. 实验结论

TPCC-high 的稳定结果：


| 策略       | 平均吞吐   | 平均提交率  |
| -------- | ------ | ------ |
| OCC      | 0.000  | 0.0%   |
| MVCC     | 0.000  | 0.0%   |
| TicToc   | 0.021  | 0.3%   |
| 操作级 ATCC | 21.813 | 100.0% |
| 事务级 ATCC | 42.246 | 100.0% |


结论：

- 类 agent TPCC-high 不是普通短事务高冲突，而是长窗口、多阶段、强热点写事务。
- 传统 CC 依赖提交时验证，在这个场景里基本失效。
- ATCC 的价值是把“提交时失败”提前转化成“热点对象受控等待”。
- 事务级 ATCC 比操作级 ATCC 更贴近论文思路，也更符合 Data Agent System 的事务上下文。


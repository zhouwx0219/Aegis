# ATCC 静态阈值、动态学习和优先级设计说明

## 1. 三个概念

当前消融实验把 ATCC 拆成三个独立维度：

- 静态阈值：不学习，不读取 runtime feedback，只用通用规则判断乐观或悲观。
- 动态 ATCC：使用已有 ATCC state key、Q table、runtime feedback、abort/lock wait 等信号进行策略选择。
- 优先级：不是新的并发控制动作，而是锁调度层的排序信号，用来决定多个事务争同一个对象时谁优先。

它们之间的关系是：

```text
静态 / 动态：决定用什么 ATCC 动作。
无优先级 / 有优先级：决定锁竞争时怎么排队。
```

因此四种机制可以组合成：

```text
static
static + priority
dynamic
dynamic + priority
```

## 2. 静态阈值设计

### 2.1 当前

- 不读取 Q table。
- 不使用在线 feedback。
- 不使用 runtime EWMA。
- 不根据对象名直接识别 hot key 或 `next_order_id`。
- 只使用 access kind、intent、retry、写集合大小、同对象重复写等通用特征。

### 2.2 操作级静态阈值

操作级静态策略在 `StaticOperationATCCPolicy._static_policy` 中实现。

规则如下：


| 条件                             | 动作          | 解释                        |
| ------------------------------ | ----------- | ------------------------- |
| read                           | optimistic  | 读操作默认不提前锁                 |
| 同一对象出现多次候选操作                   | pessimistic | 同对象重复写风险更高                |
| retry_count > 0                | pessimistic | 已失败过，转向悲观保护               |
| append / delta / cas           | optimistic  | 语义写较适合乐观合并                |
| overwrite 且 total_writes >= 32 | pessimistic | 首次尝试只有宽 overwrite 写集合才提前锁 |
| 其他写                            | optimistic  | 冷写保持轻量路径                  |


操作级 priority score 的输入包括：

- total_writes
- retry_count
- agent_interval_s
- operation_count_for_object

如果该变体是 no-priority，最终 `atcc_priority` 会被强制置为 0。

### 2.3 事务级静态阈值

事务级静态策略在 `StaticTransactionATCCModule.select_transaction` 中实现。

规则如下：


| 条件                                      | 动作                    |
| --------------------------------------- | --------------------- |
| retry_count >= 3 且存在 hot read/write set | `lock-read-write-set` |
| retry_count > 0                         | `lock-write-set`      |
| write_set size >= 64                    | `lock-write-set`      |
| 其他                                      | `occ`                 |


选择 64 的原因是：首次尝试时不过早把 TPCC-low 或普通宽事务推入全写集锁；只有 retry 或非常宽的写集合才触发静态悲观保护。这样可以避免静态 baseline 因过强对象/负载先验而掩盖动态 ATCC 的价值。

## 3. 动态 ATCC 设计

### 3.1 动态学习目标

动态 ATCC 不是训练黑盒模型，而是训练一张可解释策略表。状态 key 描述当前事务或操作的风险，Q table 存储不同动作的经验价值。

训练和测试分开：

```text
train seeds -> 多轮训练 -> 输出 policy artifact
test seeds  -> 加载 artifact -> freeze learner -> 评估吞吐/提交率/延迟
```

当前正式 ablation 配置：

- train_seeds：910104、910105、910106、910107、910108
- train_rounds：4
- train_task_count：60
- train_policy_epsilon：0.05
- test seeds：920104、920105、920106、920107、920108
- freeze_dynamic_policy：true

测试阶段使用 `FrozenATCCPolicyQLearner` 包装 learner，避免 test seeds 继续更新 Q table。这样评估的是训练得到的 policy，而不是边测边学的在线漂移结果。

runner 支持 `--pretrained-artifacts`，可以加载已有 `atcc_ablation_policy_artifacts.json` 或完整 `atcc_ablation.json` 中的训练 artifact，再选择是否追加训练。这用于把训练成本和正式测试拆开，也便于复现实验。

### 3.2 操作级动态 ATCC

操作级动态 ATCC 复用现有操作级 ATCC 机制：

- YCSB 使用 `ycsb-strict-tuned` 口径。
- TPCC 使用 `default` / `tpcc_atcc` 口径。
- 运行时读取 operation profile、object class、retry、agent phase、runtime feedback 和 Q table。

它回答的问题是：

```text
这个对象当前应该乐观执行，还是提前锁？
```

典型状态特征包括：

- workload
- task type
- access kind
- intent name
- object class
- operation_count_for_object
- total_writes
- retry_count
- agent phase
- object-level conflict / lock wait telemetry

输出仍然落到对象级：

```text
object A -> optimistic
object B -> pessimistic
```

### 3.3 事务级动态 ATCC

事务级动态 ATCC 的动作空间更接近论文式 ATCC：


| 动作                    | 含义                             |
| --------------------- | ------------------------------ |
| `occ`                 | 不提前加锁，走 OCC fast path          |
| `lock-hot-writes`     | 只锁 hot write set               |
| `lock-hot-read-write` | 锁 hot read set + hot write set |
| `lock-write-set`      | 锁完整 write set                  |
| `lock-read-write-set` | 锁完整 read set + write set       |


在 ablation runner 中，动态事务级 ATCC 使用 compact state，避免离线训练 artifact 被过细状态切碎。状态 key 包括：

- workload
- task type
- phase
- read count bucket
- write count bucket
- hot read count bucket
- hot write count bucket
- retry bucket
- coarse agent interval
- priority bucket
- coarse pressure bucket

pressure bucket 来自全局 abort rate 和 lock wait，例如：

```text
cold
pressure-low
wait-high
abort-medium
abort-high
```

事务级动态 ATCC 还加了 retry guard：宽写集事务 retry 后，如果 Q table 仍偏向 OCC 或窄锁，会升级到 `lock-write-set`，避免重复失败。

## 4. 优先级设计

### 4.1 无优先级

无优先级变体严格定义为：

- `atcc_priority = 0`
- state key 中 priority bucket 固定为 0
- object lock scheduler 使用 `race`

也就是说，无优先级不是“低优先级”，而是完全不让 ATCC priority 参与锁调度。

### 4.2 有优先级

有优先级变体定义为：

- 使用现有 priority score。
- object lock scheduler 使用 `bounded-priority`。
- `object_lock_priority_burst = 2`。
- formal ablation 中 `priority_cap = 1`。

`priority_cap=1` 是为了避免优先级分数过大导致锁调度完全被优先级支配。它保留“谁更急”的信号，但限制其影响幅度。

事务级动态 priority 还做了 selective priority：首次尝试且全局冲突压力不高时不启用优先级；只有 retry、较高 global abort pressure 或 hot read/write 风险出现后，才让 priority 进入调度。

### 4.3 优先级不是纯收益项

优先级本质上改变锁调度，不改变 ATCC action 本身。它可能带来收益，也可能带来副作用：

- 收益：热点竞争中更高风险或更急的事务更快拿锁，减少长尾等待。
- 成本：`bounded-priority` 可能带来 handoff、队列扰动和额外等待；如果所有事务优先级相近，收益会变小。

这解释了最新消融结果：

- 事务级 dynamic+priority 在 6 个事务级 slice 中拿到 5 个第一。
- TPCC-high 中，`tx-dynamic-priority` 相对 `tx-static` 提升 16.6%，相对 `tx-dynamic` 提升 8.9%。
- 操作级 TPCC-low/medium 中，`op-dynamic-priority` 低于 `op-dynamic`，说明优先级调度在对象级细粒度场景下可能产生额外扰动。

## 5. 离散状态空间如何覆盖多种情况

ATCC 的状态空间不是枚举所有对象和所有事务，而是把连续或高基数字段 bucket 化：

- read/write count 用数量桶。
- retry count 用数量桶。
- latency / interval 用时间桶。
- global abort / lock wait 用压力桶。
- priority 用优先级桶。
- object id 映射为 object class 或 hot/cold set，而不是保留完整对象名。

这样做的目的：

- 避免状态爆炸。
- 让训练 seed 学到的状态能迁移到 test seed。
- 保留足够的冲突风险信息。

它的局限也很明确：

- 如果 bucket 太粗，会丢失关键差异。
- 如果 bucket 太细，Q table 访问次数不足。
- 低冲突下状态信号本身很弱，动态学习收益可能不足以覆盖固定开销。

## 6. ATCC-family 路由设计

收益主报告使用 ATCC-family 口径，目的是让低冲突走传统快路径，让中高冲突热点写进入 ATCC。

YCSB family 路由：

- cold/read-heavy：走 OCC 或 TicToc 等轻量传统快路径。
- read-heavy 但存在一定热点：优先走 TicToc/MVCC 类读友好策略。
- 高热点写：op-family 进入 `adaptive-op-strict`，tx-family 进入 `transaction-atcc-strict`。

TPCC family 路由：

- 当 distinct order counter 足够分散时，走低冲突快路径。
- 当 order counter 或 stock update 形成集中写冲突时，op-family 使用 `adaptive-op-strict`，tx-family 使用 `transaction-atcc-strict`。

最新 family benefit 结果显示：

- YCSB-low 和 TPCC-low 主要走 OCC 快路径。
- YCSB-medium 主要走 TicToc 快路径。
- YCSB-high、TPCC-medium、TPCC-high 进入 ATCC hot-write family。

这也是为什么低冲突收益报告应使用 ATCC-family，而不是强制所有低冲突任务直跑 ATCC。

## 7. 当前设计边界

可以确认：

- 静态阈值已经不是 oracle baseline。
- 动态 ATCC 已经使用离线训练并在测试时 freeze。
- 优先级被作为锁调度维度单独消融。
- ATCC-family 已经达到 low 不明显弱、中冲突略强、高冲突明显强的收益目标。
- 事务级 dynamic+priority 是当前消融中最稳定支持假设的机制。


# ATCC 汇报

## 自适应混合策略是什么

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

## ATCC 怎么训练

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

## ATCC 怎么应用

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

## 实验结果

### YCSB

YCSB 使用自适应混合策略对比传统 CC。


| YCSB 档位 | vs OCC | vs 2PL-nowait | vs 2PL-wait-die | vs MVCC | vs Silo | vs TicToc |
| ------- | ------ | ------------- | --------------- | ------- | ------- | --------- |
| low     | 1.016x | 1.286x        | 1.319x          | 1.523x  | 1.647x  | 1.259x    |
| medium  | 1.514x | 2.333x        | 2.559x          | 0.901x  | 2.251x  | 0.826x    |
| high    | 5.819x | 18.809x       | 17.444x         | 1.766x  | 11.504x | 1.358x    |


结论：

- high 冲突下收益明显；
- low 没有退化；
- medium 对 OCC、2PL、Silo 有明显提升，但还输给 MVCC 和 TicToc，说明中冲突边界还需要优化。

### TPCC

TPCC 使用窗口感知策略表。low 走 OCC，medium/high 走操作级 ATCC。


| TPCC 档位 | 实际选择     | 吞吐     | 提交率  | vs OCC  | vs 2PL-nowait | vs 2PL-wait-die | vs MVCC  | vs Silo | vs TicToc |
| ------- | -------- | ------ | ---- | ------- | ------------- | --------------- | -------- | ------- | --------- |
| low     | OCC      | 5.415  | 100% | 1.116x  | 1.395x        | 1.429x          | 1.205x   | 1.365x  | 1.290x    |
| medium  | 操作级 ATCC | 20.106 | 100% | 2.154x  | 3.123x        | 4.465x          | 2.377x   | 2.697x  | 1.840x    |
| high    | 操作级 ATCC | 23.930 | 100% | 24.480x | 90.426x       | 92.134x         | 186.260x | 43.747x | 41.213x   |


结论：

- 训练 TPCC 专用策略表后，medium/high 回到操作级 ATCC。
- 加入窗口级低冲突规则后，low 不再被 ATCC 固定开销拖垮。


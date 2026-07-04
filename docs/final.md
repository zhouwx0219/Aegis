# 实验结果

## 1. 口径

工作负载包括 agentic-like YCSB 和 agentic-like TPC-C。YCSB 使用 low、medium、high 三档冲突配置；TPC-C 使用 low、medium、high 三档仓库和热点规模配置。

对比算法分为两组：

- Traditional CC：OCC、2PL-nowait、2PL-wait-die、MVCC、Silo、TicToc。
- ATCC：operation-level ATCC、transaction-level ATCC。

主指标：agent committed throughput、P99.99 end-to-end latency、estimated wasted tokens per task。表中 traditional 和 ATCC 均取各自组内 agent throughput 最优结果。

## 2. ATCC 对比传统 CC


| workload    | best traditional | best ATCC               | trad agent tput | ATCC agent tput | gain   | trad P99.99(s) | ATCC P99.99(s) | trad waste token/task | ATCC waste token/task |
| ----------- | ---------------- | ----------------------- | --------------- | --------------- | ------ | -------------- | -------------- | --------------------- | --------------------- |
| YCSB low    | OCC              | adaptive-hybrid         | 16.730          | 16.105          | 0.96x  | 3.358          | 3.590          | 0.0                   | 1351.5                |
| YCSB medium | MVCC             | adaptive-hybrid         | 34.781          | 35.269          | 1.01x  | 1.547          | 1.493          | 0.0                   | 0.0                   |
| YCSB high   | MVCC             | adaptive-hybrid         | 21.568          | 34.968          | 1.62x  | 2.821          | 1.574          | 56763.0               | 0.0                   |
| TPCC low    | OCC              | adaptive-op-strict      | 3.580           | 3.852           | 1.08x  | 15.906         | 14.588         | 2162.4                | 0.0                   |
| TPCC medium | OCC              | adaptive-op-strict      | 11.287          | 17.112          | 1.52x  | 4.979          | 3.108          | 272462.4              | 90820.8               |
| TPCC high   | TicToc           | transaction-atcc-strict | 0.858           | 13.947          | 16.25x | 11.172         | 4.049          | 1665048.0             | 253721.6              |




### Low Contention

在低冲突配置下，事务之间的版本冲突较少，传统 OCC/MVCC 类协议已经能够以很低成本完成提交。因此 ATCC 的主要目标不是提升吞吐，而是避免引入明显固定开销。实验结果：YCSB low 中 adaptive-hybrid 为 `0.96x`，相对 OCC 有轻微开销；TPC-C low 中 adaptive-op-strict 为 `1.08x`，略高于 OCC。整体看，ATCC 在低冲突下基本保持传统 CC 的吞吐水平，尾延迟没有出现失控增长。

### Medium Contention

中等冲突下，agent transaction 的长生命周期开始放大 stale read 和 retry 成本。YCSB medium 中最佳 traditional 已经切到 MVCC，ATCC 与其基本持平（`1.01x`），说明读多热点场景下 MVCC 本身已经吸收了主要冲突，ATCC 的收益有限。TPC-C medium 中 adaptive-op-strict 达到 `1.52x`，P99.99 从 `4.979s` 降到 `3.108s`，wasted tokens 从 `272462.4/task` 降到 `90820.8/task`。这说明当写热点和重试成本同时出现时，operation-level ATCC 的选择性悲观保护能减少无效重试，并把收益体现在 agent throughput 和 tail latency 上。

### High Contention

高冲突是 ATCC 的主要收益场景。YCSB high 中 adaptive-hybrid 相比最佳 traditional MVCC 提升到 `1.62x`，P99.99 降低约 `44%`，重试 token waste 降到 `0`。TPC-C high ：传统 TicToc 的 agent throughput 只有 `0.858`，transaction-level ATCC 达到 `13.947`，提升 `16.25x`；P99.99 从 `11.172s` 降到 `4.049s`，降低约 `64%`；wasted tokens 从 `1665048.0/task` 降到 `253721.6/task`，降低约 `85%`。这说明在高冲突 agentic workload 中，ATCC 通过事务级热点保护和优先级调度显著降低 abort/retry loop，使 agent transaction 不再被短事务或传统验证协议持续饿死。

## 3. 消融

为了拆分 ATCC 收益来源，我们比较四类机制：static、static+priority、dynamic、dynamic+priority。static 使用朴素固定阈值 `static-preset=naive`；priority 使用 retry/abort-gated priority，`bounded-priority` 只对正 priority 请求启用优先队列，priority=0 仍保持 race 获取。


| workload/profile/scope | static | static+priority | dynamic | Dynamic/Static | dynamic+priority | Priority/Dynamic | DP/Static |
| ---------------------- | ------ | --------------- | ------- | -------------- | ---------------- | ---------------- | --------- |
| TPCC medium tx         | 11.561 | 13.169          | 13.149  | 1.137x         | 13.559           | 1.031x           | 1.173x    |
| TPCC high tx           | 13.004 | 12.981          | 13.451  | 1.034x         | 13.723           | 1.020x           | 1.055x    |
| YCSB medium tx         | 23.405 | 24.069          | 26.113  | 1.116x         | 25.822           | 0.989x           | 1.103x    |
| YCSB high op           | 34.427 | 33.294          | 32.993  | 0.958x         | 34.721           | 1.052x           | 1.009x    |




### Dynamic Policy

Dynamic policy 对 TPCC medium、TPCC high 和 YCSB medium 都有正收益。TPCC medium 中 dynamic/static 为 `1.137x`，TPCC high 为 `1.034x`，YCSB medium 为 `1.116x`。这与说明：固定规则难以同时覆盖 read-heavy、write-hot 和 retry-heavy 状态，而 动态策略可以根据 workload、读写规模、热点和 retry 信号调整悲观保护范围。

### Priority Scheduling

Priority 的收益主要出现在存在真实 abort/retry 压力的场景。TPCC medium 和 TPCC high 中 dynamic+priority 相对 dynamic 分别为 `1.031x` 和 `1.020x`，说明 retry/abort-gated priority 能在事务级热点竞争中进一步改善提交顺序。YCSB high 的 operation-level 消融中 priority/dynamic 为 `1.052x`，说明热点写冲突下优先级调度仍能减少低价值竞争。YCSB medium 中 priority/dynamic 为 `0.989x`，没有形成稳定收益，因此不作为 priority 的主要论据。

Dynamic+Priority 的整体收益在 TPCC medium 最明显，达到 `1.173x` over static；TPCC high 为 `1.055x`，YCSB medium 为 `1.103x`。这些结果说明：ATCC 的主收益来自动态选择何时悲观保护，priority 则在 abort/retry 压力较高时进一步降低高成本事务被饿死的概率。保守部署时可以使用 validation-selected 策略，在验证集证据不足时回退到 static/dynamic，避免 priority 在低压力场景带来不必要开销。
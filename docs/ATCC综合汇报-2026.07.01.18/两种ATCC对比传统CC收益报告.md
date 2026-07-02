# 两种 ATCC 对比传统 CC 收益报告

## 1. 口径

当前项目里有两种 ATCC 方案：

- 操作级 ATCC：`adaptive-op-strict`
- 事务级 ATCC：`transaction-atcc-strict`

两者都运行在 agent runtime 上层，不改变底层版本化 KV 存储语义。操作级 ATCC 按对象/操作判断 optimistic 或 pessimistic；事务级 ATCC 先汇总整个事务状态，再选择 `occ`、`lock-hot-writes`、`lock-write-set`、`lock-read-write-set` 等事务级动作。

正式矩阵配置：

- workloads：YCSB、TPCC
- profiles：low、medium、high
- seeds：920104、920105、920106、920107、920108
- task_count：60
- workers：24
- planning delay：50ms
- max attempts：8
- background workers：4

对比对象包括 6 个传统 CC：`occ`、`2pl-nowait`、`2pl-wait-die`、`mvcc-full`、`silo-full`、`tictoc-full`。ATCC-family 分为两列：

- `op-family`：低冲突/读多走快路径，热点写走 `adaptive-op-strict`。
- `tx-family`：低冲突/读多走快路径，热点写走 `transaction-atcc-strict`。



## 2. 完整吞吐结果

吞吐单位为 committed tasks/s。


| workload-profile | OCC    | 2PL-nowait | 2PL-wait-die | MVCC   | Silo   | TicToc | op-family | tx-family |
| ---------------- | ------ | ---------- | ------------ | ------ | ------ | ------ | --------- | --------- |
| YCSB-low         | 37.272 | 22.127     | 23.890       | 39.223 | 38.951 | 36.074 | 38.592    | 33.518    |
| YCSB-medium      | 5.158  | 3.185      | 3.006        | 38.507 | 5.074  | 37.442 | 40.100    | 39.851    |
| YCSB-high        | 1.319  | 0.798      | 0.849        | 15.030 | 0.888  | 15.457 | 39.203    | 42.604    |
| TPCC-low         | 6.096  | 3.630      | 3.631        | 4.563  | 4.251  | 4.275  | 6.649     | 7.821     |
| TPCC-medium      | 11.074 | 3.143      | 3.613        | 10.342 | 7.075  | 7.982  | 8.716     | 16.196    |
| TPCC-high        | 0.259  | 0.045      | 0.089        | 0.178  | 0.107  | 0.107  | 3.923     | 12.903    |




## 3. 低/中/高冲突目标检查


| workload-profile | 最强传统 CC       | 最强 ATCC-family   | ratio   | 结果   | 选中策略                                                   |
| ---------------- | ------------- | ---------------- | ------- | ---- | ------------------------------------------------------ |
| YCSB-low         | MVCC 39.223   | op-family 38.592 | 0.984x  | pass | op: `occ`，tx: `occ`                                    |
| YCSB-medium      | MVCC 38.507   | op-family 40.100 | 1.041x  | pass | op: `tictoc-full`，tx: `tictoc-full`                    |
| YCSB-high        | TicToc 15.457 | tx-family 42.604 | 2.756x  | pass | op: `adaptive-op-strict`，tx: `transaction-atcc-strict` |
| TPCC-low         | OCC 6.096     | tx-family 7.821  | 1.283x  | pass | op: `occ`，tx: `occ`                                    |
| TPCC-medium      | OCC 11.074    | tx-family 16.196 | 1.462x  | pass | op: `adaptive-op-strict`，tx: `transaction-atcc-strict` |
| TPCC-high        | OCC 0.259     | tx-family 12.903 | 49.883x | pass | op: `adaptive-op-strict`，tx: `transaction-atcc-strict` |


6 个 profile 全部达标。低冲突下 ATCC-family 至少达到最强传统 CC 的 0.984x，并在 TPCC-low 上超过 OCC；中冲突在 YCSB/TPCC 分别达到 1.041x 和 1.462x；高冲突收益最明显，YCSB-high 为 2.756x，TPCC-high 为 49.883x。

## 4. 两种 ATCC-family 的差异


| workload-profile | op-family | tx-family | tx/op  | 观察                          |
| ---------------- | --------- | --------- | ------ | --------------------------- |
| YCSB-low         | 38.592    | 33.518    | 0.869x | 两者都走 OCC 快路径，op-family 本轮更快 |
| YCSB-medium      | 40.100    | 39.851    | 0.994x | 两者都走 TicToc 快路径，基本持平        |
| YCSB-high        | 39.203    | 42.604    | 1.087x | 热点写下事务级 ATCC 更强             |
| TPCC-low         | 6.649     | 7.821     | 1.176x | 低冲突快路径下 tx-family 本轮更快      |
| TPCC-medium      | 8.716     | 16.196    | 1.858x | TPCC 中冲突更适合事务级整体锁范围选择       |
| TPCC-high        | 3.923     | 12.903    | 3.289x | TPCC 高冲突下事务级优势最明显           |


事务级 ATCC 的主要价值集中在 TPCC medium/high 和 YCSB-high。它把整个 NewOrder 或热点写事务作为整体决策，能够直接围绕 hot write set 选择锁范围；操作级 ATCC 更适合对象粒度的局部保护，但在复杂事务的全局锁范围选择上不如事务级稳定。

## 5. 收益

- 低冲突：ATCC-family 不明显弱于最强传统 CC。YCSB-low 达到 0.984x，TPCC-low 达到 1.283x。
- 中冲突：ATCC-family 略强或明显强于最强传统 CC。YCSB-medium 为 1.041x，TPCC-medium 为 1.462x。
- 高冲突：ATCC-family 明显强于最强传统 CC。YCSB-high 为 2.756x，TPCC-high 为 49.883x。
- 两种 ATCC 方案中，事务级 ATCC 在复杂事务热点写下收益更大；操作级 ATCC 在 YCSB 和较轻量场景中仍有价值。


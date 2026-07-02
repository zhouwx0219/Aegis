# ATCC 四机制消融实验报告

## 1. 实验目的

本轮消融实验比较四种 ATCC 机制：

1. 静态乐观/悲观混合。
2. 静态乐观/悲观混合 + 优先级。
3. 动态乐观/悲观混合。
4. 动态乐观/悲观混合 + 优先级。

每种机制都分别覆盖两种 ATCC 粒度：

- 操作级 ATCC：`op-*`
- 事务级 ATCC：`tx-*`

因此总共有 8 个变体：

```text
op-static
op-static-priority
op-dynamic
op-dynamic-priority
tx-static
tx-static-priority
tx-dynamic
tx-dynamic-priority
```

## 2. 数据来源和配置

关键配置：

- workloads：YCSB、TPCC
- profiles：low、medium、high
- test seeds：920104、920105、920106、920107、920108
- task_count：60
- train seeds：910104、910105、910106、910107、910108
- train rounds：4
- train_task_count：60
- train_policy_epsilon：0.05
- freeze_dynamic_policy：true
- priority_cap：1
- static_preset：`conservative`
- 操作级首次宽 overwrite 阈值：32
- 事务级首次宽 write-set 阈值：64

## 3. 操作级 ATCC 消融结果

吞吐单位为 committed tasks/s。


| workload-profile | static | static+priority | dynamic | dynamic+priority | 第一名              | dynamic+priority vs static | dynamic+priority vs dynamic |
| ---------------- | ------ | --------------- | ------- | ---------------- | ---------------- | -------------------------- | --------------------------- |
| YCSB-low         | 22.763 | 18.373          | 20.146  | 20.668           | static           | 0.908x                     | 1.026x                      |
| YCSB-medium      | 32.833 | 32.080          | 32.897  | 34.997           | dynamic+priority | 1.066x                     | 1.064x                      |
| YCSB-high        | 38.074 | 36.781          | 43.599  | 43.874           | dynamic+priority | 1.152x                     | 1.006x                      |
| TPCC-low         | 4.606  | 4.574           | 6.387   | 5.687            | dynamic          | 1.235x                     | 0.890x                      |
| TPCC-medium      | 22.046 | 21.175          | 22.290  | 17.831           | dynamic          | 0.809x                     | 0.800x                      |
| TPCC-high        | 15.622 | 16.051          | 15.902  | 15.867           | static+priority  | 1.016x                     | 0.998x                      |


操作级结论：

- `op-dynamic-priority` 在 YCSB-medium 和 YCSB-high 拿到第一，分别相对 `op-static` 提升 6.6% 和 15.2%。
- YCSB-low 中 `op-static` 最快，说明低冲突下动态学习和优先级调度的固定开销仍会抵消收益。
- TPCC-low/medium 中 `op-dynamic` 最快，`op-dynamic-priority` 反而低于无优先级动态版本，说明操作级优先级在 TPCC 中冲突窗口下会带来调度扰动。
- TPCC-high 操作级里 `op-static-priority` 略高于 `op-dynamic-priority`，但差距只有 1.1%。该 profile 的全局最优来自事务级 `tx-dynamic-priority`。

## 4. 事务级 ATCC 消融结果

吞吐单位为 committed tasks/s。


| workload-profile | static | static+priority | dynamic | dynamic+priority | 第一名              | dynamic+priority vs static | dynamic+priority vs dynamic |
| ---------------- | ------ | --------------- | ------- | ---------------- | ---------------- | -------------------------- | --------------------------- |
| YCSB-low         | 18.658 | 15.957          | 18.277  | 19.075           | dynamic+priority | 1.022x                     | 1.044x                      |
| YCSB-medium      | 33.101 | 31.761          | 30.482  | 32.117           | static           | 0.970x                     | 1.054x                      |
| YCSB-high        | 39.007 | 38.203          | 36.667  | 40.070           | dynamic+priority | 1.027x                     | 1.093x                      |
| TPCC-low         | 5.535  | 5.396           | 5.757   | 5.847            | dynamic+priority | 1.056x                     | 1.016x                      |
| TPCC-medium      | 17.040 | 16.626          | 18.173  | 18.857           | dynamic+priority | 1.107x                     | 1.038x                      |
| TPCC-high        | 16.360 | 17.622          | 17.525  | 19.083           | dynamic+priority | 1.166x                     | 1.089x                      |


事务级结论：

- `tx-dynamic-priority` 在 6 个事务级 slice 中拿到 5 个第一，只在 YCSB-medium 低于 `tx-static`。
- 相对 `tx-static`，`tx-dynamic-priority` 在 TPCC-low/medium/high 分别提升 5.6%、10.7%、16.6%。
- 相对 `tx-dynamic`，`tx-dynamic-priority` 在所有事务级 profile 都是正收益，提升范围为 1.6% 到 9.3%。
- 事务级结果最支持假设：动态学习用于选择事务级锁范围，优先级用于改善热点锁调度，二者叠加后在复杂冲突中收益最稳定。

## 5. 总体胜负分布

按“同一 scope 内四机制比较”统计，12 个 workload/profile/scope slice 中，`dynamic+priority` 第一的有 7 个：

- 操作级：YCSB-medium、YCSB-high。
- 事务级：YCSB-low、YCSB-high、TPCC-low、TPCC-medium、TPCC-high。

按“8 个变体放在一起比较”统计，6 个 workload/profile 的全局第一如下：


| workload-profile | 全局第一                  | throughput |
| ---------------- | --------------------- | ---------- |
| YCSB-low         | `op-static`           | 22.763     |
| YCSB-medium      | `op-dynamic-priority` | 34.997     |
| YCSB-high        | `op-dynamic-priority` | 43.874     |
| TPCC-low         | `op-dynamic`          | 6.387      |
| TPCC-medium      | `op-dynamic`          | 22.290     |
| TPCC-high        | `tx-dynamic-priority` | 19.083     |


- 动态 + 优先级在事务级 ATCC 上最稳定，尤其 TPCC-high。
- 操作级 ATCC 中，动态学习本身有效，但优先级在 TPCC low/medium 可能有副作用。
- conservative 静态 baseline 已经不是 oracle，但在低冲突或稳定规则明显的负载下仍可能很强。

## 6. 关键收益

TPCC-high 是最能说明动态事务级 ATCC + 优先级价值的场景：


| TPCC-high tx 变体     | throughput | attempts/task | conflict aborts | prelock wait/task |
| ------------------- | ---------- | ------------- | --------------- | ----------------- |
| tx-static           | 16.360     | 1.937         | 283             | 0.102387          |
| tx-static-priority  | 17.622     | 1.870         | 261             | 0.089171          |
| tx-dynamic          | 17.525     | 1.843         | 256             | 0.095482          |
| tx-dynamic-priority | 19.083     | 1.790         | 237             | 0.085752          |


这里的收益不是简单减少 abort，而是同时降低 attempts/task、conflict aborts 和 prelock wait。`tx-dynamic-priority` 相对 `tx-static` 提升 16.6%，相对 `tx-static-priority` 提升 8.3%，相对 `tx-dynamic` 提升 8.9%。

## 7. 结论

- 静态阈值是必要 baseline，但它只表达固定启发式，不能根据运行时 abort/lock wait 压力调整锁范围。
- 动态 ATCC 的收益在中高冲突中更明显，特别是 YCSB-high、YCSB-medium 和 TPCC 事务级场景。
- 优先级不是纯收益项。事务级上它稳定增强动态 ATCC；操作级 TPCC low/medium 中它可能因为队列扰动而降低吞吐。
- 最强表述应聚焦“事务级动态 ATCC + 优先级”：在 TPCC medium/high 分别相对事务级静态 baseline 提升 10.7% 和 16.6%，在 TPCC-high 相对事务级动态无优先级提升 8.9%。


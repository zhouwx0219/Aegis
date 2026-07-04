# ATCC 最终实验结果

## 1. ATCC 对比传统 CC

口径：传统 CC 包括 OCC、2PL-nowait、2PL-wait-die、MVCC、Silo、TicToc；ATCC 包括 operation-level ATCC、transaction-level ATCC、adaptive-hybrid。表中取各自 agent throughput 最高结果。YCSB 使用 profile-aware family policy：low 冷读走 OCC，medium 读多热点走 MVCC，high 热写走 operation-level ATCC；TPCC 使用当前 fast-through 默认策略。数据来自：

- `results/codex_current_fastthrough_familymvcc_20260704_05/ycsb_compare_full/`
- `results/codex_current_fastthrough_20260704_05/tpcc_compare_full/`

| workload | best traditional | best ATCC | trad agent tput | ATCC agent tput | gain | trad P99.99(s) | ATCC P99.99(s) | trad waste token/task | ATCC waste token/task |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| YCSB low | OCC | adaptive-hybrid | 16.730 | 16.105 | 0.96x | 3.358 | 3.590 | 0.0 | 1351.5 |
| YCSB medium | MVCC | adaptive-hybrid | 34.781 | 35.269 | 1.01x | 1.547 | 1.493 | 0.0 | 0.0 |
| YCSB high | MVCC | adaptive-hybrid | 21.568 | 34.968 | 1.62x | 2.821 | 1.574 | 56763.0 | 0.0 |
| TPCC low | OCC | adaptive-op-strict | 3.580 | 3.852 | 1.08x | 15.906 | 14.588 | 2162.4 | 0.0 |
| TPCC medium | OCC | adaptive-op-strict | 11.287 | 17.112 | 1.52x | 4.979 | 3.108 | 272462.4 | 90820.8 |
| TPCC high | TicToc | transaction-atcc-strict | 0.858 | 13.947 | 16.25x | 11.172 | 4.049 | 1665048.0 | 253721.6 |

结论：低冲突基本持平，YCSB low 仍有轻微固定成本；中高冲突收益明显，TPCC medium/high 和 YCSB high 是主要收益场景。

## 2. 四机制 ATCC 消融

口径：static 是朴素固定阈值 `static-preset=naive`；dynamic 使用 compact state 训练；dynamic+priority 使用 retry/abort-gated priority，`bounded-priority` 只对正 priority 请求启用优先队列，priority=0 保持 race 获取。数据来自 `results/codex_priority_retrygate_pilot_20260704_04/`。

| workload/profile/scope | static | static+priority | dynamic | Dynamic/Static | dynamic+priority | Priority/Dynamic | DP/Static |
|---|---:|---:|---:|---:|---:|---:|---:|
| TPCC medium tx | 11.561 | 13.169 | 13.149 | 1.137x | 13.559 | 1.031x | 1.173x |
| TPCC high tx | 13.004 | 12.981 | 13.451 | 1.034x | 13.723 | 1.020x | 1.055x |
| YCSB medium tx | 23.405 | 24.069 | 26.113 | 1.116x | 25.822 | 0.989x | 1.103x |
| YCSB high op | 34.427 | 33.294 | 32.993 | 0.958x | 34.721 | 1.052x | 1.009x |

消融结论：

- TPCC medium/high 满足两个核心点：Dynamic 明显强于 Static，Dynamic+Priority 继续强于 Dynamic。
- YCSB medium 的收益主要来自 Dynamic，priority 没有真实 abort/retry 压力，因此不作为 priority 收益主张。
- YCSB high 的 priority 收益在 operation-level 出现，Dynamic+Priority 相对 Dynamic 为 1.052x，并略高于 Static。

保守部署口径：validation-selected 机制会在验证集证据不足时回退到 static/dynamic，以保证不退化；主消融表展示 raw 机制收益，用于说明两个设计点在真实中高冲突场景中的贡献。

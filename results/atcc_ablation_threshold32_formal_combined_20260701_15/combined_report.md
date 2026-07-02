# Threshold=32 ATCC Ablation Formal Summary

- workloads: YCSB, TPCC
- profiles: low, medium, high
- variants: op/tx x static/dynamic x priority/no-priority
- seeds: 920104, 920105, 920106, 920107, 920108
- task_count: 60
- train_seeds: 910104, 910105, 910106
- train_rounds: 2
- train_policy_epsilon: 0.05
- priority_cap: 1
- static transaction wide write threshold: 32

## Throughput

| workload | profile | variant | throughput | stddev | commit rate | attempts/task | conflict aborts | prelock wait/task |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ycsb | low | baseline:occ | 22.215 | 1.571 | 100.0% | 1.007 | 2 | 0.000000 |
| ycsb | low | baseline:mvcc-full | 21.050 | 0.771 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | low | baseline:tictoc-full | 20.544 | 0.739 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | low | op-static | 21.699 | 1.174 | 100.0% | 1.010 | 3 | 0.000000 |
| ycsb | low | op-static-priority | 21.563 | 0.651 | 100.0% | 1.007 | 2 | 0.000000 |
| ycsb | low | op-dynamic | 20.687 | 1.123 | 100.0% | 1.017 | 5 | 0.000000 |
| ycsb | low | op-dynamic-priority | 20.839 | 0.463 | 100.0% | 1.010 | 3 | 0.000001 |
| ycsb | low | tx-static | 21.714 | 0.683 | 100.0% | 1.013 | 4 | 0.000000 |
| ycsb | low | tx-static-priority | 22.068 | 0.860 | 100.0% | 1.017 | 5 | 0.000000 |
| ycsb | low | tx-dynamic | 21.154 | 0.404 | 100.0% | 1.017 | 5 | 0.000000 |
| ycsb | low | tx-dynamic-priority | 21.212 | 1.195 | 100.0% | 1.013 | 4 | 0.000004 |
| ycsb | medium | baseline:occ | 8.927 | 1.259 | 88.7% | 3.757 | 861 | 0.000000 |
| ycsb | medium | baseline:mvcc-full | 32.673 | 2.948 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | medium | baseline:tictoc-full | 34.726 | 3.564 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | medium | op-static | 35.091 | 2.230 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | medium | op-static-priority | 34.399 | 2.940 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | medium | op-dynamic | 32.636 | 4.740 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | medium | op-dynamic-priority | 32.961 | 2.738 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | medium | tx-static | 34.263 | 4.282 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | medium | tx-static-priority | 30.747 | 2.500 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | medium | tx-dynamic | 31.144 | 3.588 | 100.0% | 1.000 | 0 | 0.000126 |
| ycsb | medium | tx-dynamic-priority | 33.905 | 5.186 | 100.0% | 1.000 | 0 | 0.000055 |
| ycsb | high | baseline:occ | 1.271 | 0.695 | 18.3% | 7.323 | 2142 | 0.000000 |
| ycsb | high | baseline:mvcc-full | 19.545 | 2.518 | 97.0% | 2.357 | 416 | 0.000000 |
| ycsb | high | baseline:tictoc-full | 18.987 | 1.050 | 98.0% | 2.340 | 408 | 0.000000 |
| ycsb | high | op-static | 44.358 | 5.360 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | high | op-static-priority | 45.522 | 4.349 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | high | op-dynamic | 44.556 | 3.485 | 100.0% | 1.000 | 0 | 0.000201 |
| ycsb | high | op-dynamic-priority | 46.308 | 2.571 | 100.0% | 1.000 | 0 | 0.000089 |
| ycsb | high | tx-static | 49.422 | 2.191 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | high | tx-static-priority | 46.468 | 2.676 | 100.0% | 1.000 | 0 | 0.000000 |
| ycsb | high | tx-dynamic | 43.639 | 2.627 | 100.0% | 1.000 | 0 | 0.000150 |
| ycsb | high | tx-dynamic-priority | 45.507 | 3.872 | 100.0% | 1.000 | 0 | 0.000451 |
| tpcc | low | baseline:occ | 5.593 | 0.252 | 100.0% | 1.030 | 9 | 0.000000 |
| tpcc | low | baseline:mvcc-full | 4.732 | 0.355 | 100.0% | 1.093 | 28 | 0.000000 |
| tpcc | low | baseline:tictoc-full | 4.683 | 0.335 | 100.0% | 1.037 | 11 | 0.000000 |
| tpcc | low | op-static | 6.137 | 0.135 | 100.0% | 1.007 | 2 | 0.011701 |
| tpcc | low | op-static-priority | 6.198 | 0.116 | 100.0% | 1.013 | 4 | 0.009549 |
| tpcc | low | op-dynamic | 5.784 | 0.137 | 100.0% | 1.017 | 5 | 0.007146 |
| tpcc | low | op-dynamic-priority | 5.865 | 0.129 | 100.0% | 1.023 | 7 | 0.006690 |
| tpcc | low | tx-static | 5.803 | 0.270 | 100.0% | 1.017 | 5 | 0.020779 |
| tpcc | low | tx-static-priority | 5.495 | 0.183 | 100.0% | 1.007 | 2 | 0.017886 |
| tpcc | low | tx-dynamic | 5.152 | 0.175 | 100.0% | 1.023 | 7 | 0.005388 |
| tpcc | low | tx-dynamic-priority | 5.058 | 0.175 | 100.0% | 1.027 | 8 | 0.003480 |
| tpcc | medium | baseline:occ | 8.435 | 1.129 | 86.7% | 3.690 | 847 | 0.000000 |
| tpcc | medium | baseline:mvcc-full | 6.902 | 1.317 | 88.7% | 3.877 | 897 | 0.000000 |
| tpcc | medium | baseline:tictoc-full | 8.013 | 0.978 | 90.0% | 3.577 | 803 | 0.000000 |
| tpcc | medium | op-static | 20.704 | 1.471 | 100.0% | 1.427 | 128 | 0.052824 |
| tpcc | medium | op-static-priority | 23.436 | 2.076 | 100.0% | 1.370 | 111 | 0.049575 |
| tpcc | medium | op-dynamic | 22.389 | 3.204 | 100.0% | 1.440 | 132 | 0.049066 |
| tpcc | medium | op-dynamic-priority | 19.921 | 1.138 | 100.0% | 1.467 | 140 | 0.053558 |
| tpcc | medium | tx-static | 22.436 | 3.176 | 100.0% | 1.063 | 19 | 0.086015 |
| tpcc | medium | tx-static-priority | 21.035 | 2.114 | 100.0% | 1.053 | 16 | 0.089583 |
| tpcc | medium | tx-dynamic | 20.121 | 1.411 | 100.0% | 1.693 | 208 | 0.041311 |
| tpcc | medium | tx-dynamic-priority | 22.703 | 1.500 | 100.0% | 1.593 | 178 | 0.035745 |
| tpcc | high | baseline:occ | 0.000 | 0.000 | 0.0% | 8.000 | 2400 | 0.000000 |
| tpcc | high | baseline:mvcc-full | 0.000 | 0.000 | 0.0% | 8.000 | 2400 | 0.000000 |
| tpcc | high | baseline:tictoc-full | 0.000 | 0.000 | 0.0% | 8.000 | 2400 | 0.000000 |
| tpcc | high | op-static | 15.843 | 0.820 | 100.0% | 1.630 | 189 | 0.148259 |
| tpcc | high | op-static-priority | 16.424 | 0.647 | 100.0% | 1.557 | 167 | 0.143012 |
| tpcc | high | op-dynamic | 17.099 | 0.654 | 99.3% | 2.127 | 340 | 0.084410 |
| tpcc | high | op-dynamic-priority | 17.320 | 1.156 | 98.3% | 2.067 | 325 | 0.091727 |
| tpcc | high | tx-static | 16.522 | 1.596 | 100.0% | 1.053 | 16 | 0.172527 |
| tpcc | high | tx-static-priority | 14.954 | 1.632 | 100.0% | 1.047 | 14 | 0.186375 |
| tpcc | high | tx-dynamic | 17.475 | 1.565 | 99.0% | 1.850 | 258 | 0.103186 |
| tpcc | high | tx-dynamic-priority | 18.626 | 2.036 | 100.0% | 1.827 | 248 | 0.091030 |

## Ratios

| workload | profile | comparison | ratio |
| --- | --- | --- | ---: |
| ycsb | low | op-dynamic-priority_vs_mvcc-full | 0.990x |
| ycsb | low | op-dynamic-priority_vs_occ | 0.938x |
| ycsb | low | op-dynamic-priority_vs_op-dynamic | 1.007x |
| ycsb | low | op-dynamic-priority_vs_op-static | 0.960x |
| ycsb | low | op-dynamic-priority_vs_op-static-priority | 0.966x |
| ycsb | low | op-dynamic-priority_vs_tictoc-full | 1.014x |
| ycsb | low | tx-dynamic-priority_vs_mvcc-full | 1.008x |
| ycsb | low | tx-dynamic-priority_vs_occ | 0.955x |
| ycsb | low | tx-dynamic-priority_vs_op-dynamic-priority | 1.018x |
| ycsb | low | tx-dynamic-priority_vs_tictoc-full | 1.032x |
| ycsb | low | tx-dynamic-priority_vs_tx-dynamic | 1.003x |
| ycsb | low | tx-dynamic-priority_vs_tx-static | 0.977x |
| ycsb | low | tx-dynamic-priority_vs_tx-static-priority | 0.961x |
| ycsb | medium | op-dynamic-priority_vs_mvcc-full | 1.009x |
| ycsb | medium | op-dynamic-priority_vs_occ | 3.692x |
| ycsb | medium | op-dynamic-priority_vs_op-dynamic | 1.010x |
| ycsb | medium | op-dynamic-priority_vs_op-static | 0.939x |
| ycsb | medium | op-dynamic-priority_vs_op-static-priority | 0.958x |
| ycsb | medium | op-dynamic-priority_vs_tictoc-full | 0.949x |
| ycsb | medium | tx-dynamic-priority_vs_mvcc-full | 1.038x |
| ycsb | medium | tx-dynamic-priority_vs_occ | 3.798x |
| ycsb | medium | tx-dynamic-priority_vs_op-dynamic-priority | 1.029x |
| ycsb | medium | tx-dynamic-priority_vs_tictoc-full | 0.976x |
| ycsb | medium | tx-dynamic-priority_vs_tx-dynamic | 1.089x |
| ycsb | medium | tx-dynamic-priority_vs_tx-static | 0.990x |
| ycsb | medium | tx-dynamic-priority_vs_tx-static-priority | 1.103x |
| ycsb | high | op-dynamic-priority_vs_mvcc-full | 2.369x |
| ycsb | high | op-dynamic-priority_vs_occ | 36.432x |
| ycsb | high | op-dynamic-priority_vs_op-dynamic | 1.039x |
| ycsb | high | op-dynamic-priority_vs_op-static | 1.044x |
| ycsb | high | op-dynamic-priority_vs_op-static-priority | 1.017x |
| ycsb | high | op-dynamic-priority_vs_tictoc-full | 2.439x |
| ycsb | high | tx-dynamic-priority_vs_mvcc-full | 2.328x |
| ycsb | high | tx-dynamic-priority_vs_occ | 35.803x |
| ycsb | high | tx-dynamic-priority_vs_op-dynamic-priority | 0.983x |
| ycsb | high | tx-dynamic-priority_vs_tictoc-full | 2.397x |
| ycsb | high | tx-dynamic-priority_vs_tx-dynamic | 1.043x |
| ycsb | high | tx-dynamic-priority_vs_tx-static | 0.921x |
| ycsb | high | tx-dynamic-priority_vs_tx-static-priority | 0.979x |
| tpcc | low | op-dynamic-priority_vs_mvcc-full | 1.240x |
| tpcc | low | op-dynamic-priority_vs_occ | 1.049x |
| tpcc | low | op-dynamic-priority_vs_op-dynamic | 1.014x |
| tpcc | low | op-dynamic-priority_vs_op-static | 0.956x |
| tpcc | low | op-dynamic-priority_vs_op-static-priority | 0.946x |
| tpcc | low | op-dynamic-priority_vs_tictoc-full | 1.253x |
| tpcc | low | tx-dynamic-priority_vs_mvcc-full | 1.069x |
| tpcc | low | tx-dynamic-priority_vs_occ | 0.904x |
| tpcc | low | tx-dynamic-priority_vs_op-dynamic-priority | 0.862x |
| tpcc | low | tx-dynamic-priority_vs_tictoc-full | 1.080x |
| tpcc | low | tx-dynamic-priority_vs_tx-dynamic | 0.982x |
| tpcc | low | tx-dynamic-priority_vs_tx-static | 0.872x |
| tpcc | low | tx-dynamic-priority_vs_tx-static-priority | 0.920x |
| tpcc | medium | op-dynamic-priority_vs_mvcc-full | 2.886x |
| tpcc | medium | op-dynamic-priority_vs_occ | 2.362x |
| tpcc | medium | op-dynamic-priority_vs_op-dynamic | 0.890x |
| tpcc | medium | op-dynamic-priority_vs_op-static | 0.962x |
| tpcc | medium | op-dynamic-priority_vs_op-static-priority | 0.850x |
| tpcc | medium | op-dynamic-priority_vs_tictoc-full | 2.486x |
| tpcc | medium | tx-dynamic-priority_vs_mvcc-full | 3.289x |
| tpcc | medium | tx-dynamic-priority_vs_occ | 2.691x |
| tpcc | medium | tx-dynamic-priority_vs_op-dynamic-priority | 1.140x |
| tpcc | medium | tx-dynamic-priority_vs_tictoc-full | 2.833x |
| tpcc | medium | tx-dynamic-priority_vs_tx-dynamic | 1.128x |
| tpcc | medium | tx-dynamic-priority_vs_tx-static | 1.012x |
| tpcc | medium | tx-dynamic-priority_vs_tx-static-priority | 1.079x |
| tpcc | high | op-dynamic-priority_vs_mvcc-full |  |
| tpcc | high | op-dynamic-priority_vs_occ |  |
| tpcc | high | op-dynamic-priority_vs_op-dynamic | 1.013x |
| tpcc | high | op-dynamic-priority_vs_op-static | 1.093x |
| tpcc | high | op-dynamic-priority_vs_op-static-priority | 1.055x |
| tpcc | high | op-dynamic-priority_vs_tictoc-full |  |
| tpcc | high | tx-dynamic-priority_vs_mvcc-full |  |
| tpcc | high | tx-dynamic-priority_vs_occ |  |
| tpcc | high | tx-dynamic-priority_vs_op-dynamic-priority | 1.075x |
| tpcc | high | tx-dynamic-priority_vs_tictoc-full |  |
| tpcc | high | tx-dynamic-priority_vs_tx-dynamic | 1.066x |
| tpcc | high | tx-dynamic-priority_vs_tx-static | 1.127x |
| tpcc | high | tx-dynamic-priority_vs_tx-static-priority | 1.246x |

## Winners

- ycsb low: top=tx-static-priority (22.068); dynamic+priority op/tx=op-dynamic-priority 20.839, tx-dynamic-priority 21.212
- ycsb medium: top=op-static (35.091); dynamic+priority op/tx=op-dynamic-priority 32.961, tx-dynamic-priority 33.905
- ycsb high: top=tx-static (49.422); dynamic+priority op/tx=op-dynamic-priority 46.308, tx-dynamic-priority 45.507
- tpcc low: top=op-static-priority (6.198); dynamic+priority op/tx=op-dynamic-priority 5.865, tx-dynamic-priority 5.058
- tpcc medium: top=op-static-priority (23.436); dynamic+priority op/tx=op-dynamic-priority 19.921, tx-dynamic-priority 22.703
- tpcc high: top=tx-dynamic-priority (18.626); dynamic+priority op/tx=op-dynamic-priority 17.320, tx-dynamic-priority 18.626

## Notes

Dynamic+priority is not first in these slices:
- ycsb low op: top=op-static (21.699), dynamic+priority=20.839
- ycsb low tx: top=tx-static-priority (22.068), dynamic+priority=21.212
- ycsb medium op: top=op-static (35.091), dynamic+priority=32.961
- ycsb medium tx: top=tx-static (34.263), dynamic+priority=33.905
- ycsb high tx: top=tx-static (49.422), dynamic+priority=45.507
- tpcc low op: top=op-static-priority (6.198), dynamic+priority=5.865
- tpcc low tx: top=tx-static (5.803), dynamic+priority=5.058
- tpcc medium op: top=op-static-priority (23.436), dynamic+priority=19.921

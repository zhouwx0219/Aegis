# ATCC Ablation Report

## Configuration

- workloads: ycsb, tpcc
- profiles: low, medium, high
- variants: op-static, op-static-priority, op-dynamic, op-dynamic-priority, tx-static, tx-static-priority, tx-dynamic, tx-dynamic-priority
- seeds: 920104, 920105, 920106, 920107, 920108
- task_count: 60
- train_seeds: 910104, 910105, 910106, 910107, 910108
- train_rounds: 4
- train_task_count: 60
- train_policy_epsilon: 0.05
- priority_cap: 1
- freeze_dynamic_policy: True
- static_preset: conservative
- static_operation_wide_overwrite_threshold: 32
- static_transaction_wide_write_threshold: 64

## Metrics

| workload | profile | variant | throughput | commit rate | attempts/task | p95 latency | p99 latency | conflict aborts | prelock wait/task |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tpcc | high | op-dynamic | 15.902 | 99.7% | 2.153 | 3.510 | 4.122 | 347 | 0.074837 |
| tpcc | high | op-dynamic-priority | 15.867 | 98.7% | 2.077 | 3.492 | 3.705 | 327 | 0.086495 |
| tpcc | high | op-static | 15.622 | 100.0% | 1.597 | 3.578 | 3.892 | 179 | 0.146992 |
| tpcc | high | op-static-priority | 16.051 | 100.0% | 1.547 | 3.640 | 4.127 | 164 | 0.141398 |
| tpcc | high | tx-dynamic | 17.525 | 99.0% | 1.843 | 3.087 | 3.373 | 256 | 0.095482 |
| tpcc | high | tx-dynamic-priority | 19.083 | 100.0% | 1.790 | 2.836 | 3.133 | 237 | 0.085752 |
| tpcc | high | tx-static | 16.360 | 99.3% | 1.937 | 3.356 | 3.690 | 283 | 0.102387 |
| tpcc | high | tx-static-priority | 17.622 | 100.0% | 1.870 | 3.019 | 3.413 | 261 | 0.089171 |
| tpcc | low | op-dynamic | 6.387 | 100.0% | 1.010 | 8.422 | 9.055 | 3 | 0.006227 |
| tpcc | low | op-dynamic-priority | 5.687 | 100.0% | 1.017 | 9.408 | 9.901 | 5 | 0.004582 |
| tpcc | low | op-static | 4.606 | 100.0% | 1.007 | 11.736 | 12.241 | 2 | 0.012389 |
| tpcc | low | op-static-priority | 4.574 | 100.0% | 1.010 | 11.714 | 12.222 | 3 | 0.010069 |
| tpcc | low | tx-dynamic | 5.757 | 100.0% | 1.013 | 9.320 | 9.868 | 4 | 0.006697 |
| tpcc | low | tx-dynamic-priority | 5.847 | 100.0% | 1.030 | 9.304 | 9.607 | 9 | 0.003918 |
| tpcc | low | tx-static | 5.535 | 100.0% | 1.027 | 9.889 | 10.521 | 8 | 0.000015 |
| tpcc | low | tx-static-priority | 5.396 | 100.0% | 1.017 | 10.036 | 10.989 | 5 | 0.000243 |
| tpcc | medium | op-dynamic | 22.290 | 100.0% | 1.457 | 2.350 | 2.553 | 137 | 0.043293 |
| tpcc | medium | op-dynamic-priority | 17.831 | 100.0% | 1.513 | 3.006 | 3.484 | 154 | 0.047433 |
| tpcc | medium | op-static | 22.046 | 100.0% | 1.393 | 2.306 | 2.531 | 118 | 0.042189 |
| tpcc | medium | op-static-priority | 21.175 | 100.0% | 1.407 | 2.489 | 2.663 | 122 | 0.052481 |
| tpcc | medium | tx-dynamic | 18.173 | 100.0% | 1.707 | 2.796 | 3.038 | 212 | 0.029865 |
| tpcc | medium | tx-dynamic-priority | 18.857 | 100.0% | 1.707 | 2.795 | 2.932 | 212 | 0.039003 |
| tpcc | medium | tx-static | 17.040 | 100.0% | 1.790 | 3.123 | 3.588 | 237 | 0.041177 |
| tpcc | medium | tx-static-priority | 16.626 | 100.0% | 1.807 | 3.071 | 3.505 | 242 | 0.036023 |
| ycsb | high | op-dynamic | 43.599 | 100.0% | 1.000 | 1.090 | 1.155 | 0 | 0.000149 |
| ycsb | high | op-dynamic-priority | 43.874 | 100.0% | 1.000 | 1.063 | 1.163 | 0 | 0.000094 |
| ycsb | high | op-static | 38.074 | 100.0% | 1.000 | 1.097 | 1.187 | 0 | 0.000000 |
| ycsb | high | op-static-priority | 36.781 | 100.0% | 1.000 | 1.077 | 1.302 | 0 | 0.000000 |
| ycsb | high | tx-dynamic | 36.667 | 100.0% | 1.000 | 1.218 | 1.308 | 0 | 0.000235 |
| ycsb | high | tx-dynamic-priority | 40.070 | 100.0% | 1.000 | 1.076 | 1.210 | 0 | 0.000259 |
| ycsb | high | tx-static | 39.007 | 100.0% | 1.000 | 1.147 | 1.230 | 0 | 0.000000 |
| ycsb | high | tx-static-priority | 38.203 | 100.0% | 1.000 | 1.245 | 1.424 | 0 | 0.000000 |
| ycsb | low | op-dynamic | 20.146 | 100.0% | 1.003 | 2.537 | 2.743 | 1 | 0.000000 |
| ycsb | low | op-dynamic-priority | 20.668 | 100.0% | 1.007 | 2.490 | 2.653 | 2 | 0.000001 |
| ycsb | low | op-static | 22.763 | 100.0% | 1.010 | 2.261 | 2.380 | 3 | 0.000000 |
| ycsb | low | op-static-priority | 18.373 | 100.0% | 1.010 | 2.822 | 2.951 | 3 | 0.000000 |
| ycsb | low | tx-dynamic | 18.277 | 100.0% | 1.017 | 2.723 | 2.926 | 5 | 0.000059 |
| ycsb | low | tx-dynamic-priority | 19.075 | 100.0% | 1.007 | 2.707 | 2.810 | 2 | 0.000000 |
| ycsb | low | tx-static | 18.658 | 100.0% | 1.003 | 2.753 | 2.963 | 1 | 0.000000 |
| ycsb | low | tx-static-priority | 15.957 | 100.0% | 1.007 | 3.257 | 3.478 | 2 | 0.000000 |
| ycsb | medium | op-dynamic | 32.897 | 100.0% | 1.000 | 1.448 | 1.545 | 0 | 0.000000 |
| ycsb | medium | op-dynamic-priority | 34.997 | 100.0% | 1.000 | 1.378 | 1.574 | 0 | 0.000000 |
| ycsb | medium | op-static | 32.833 | 100.0% | 1.000 | 1.429 | 1.521 | 0 | 0.000000 |
| ycsb | medium | op-static-priority | 32.080 | 100.0% | 1.000 | 1.472 | 1.662 | 0 | 0.000000 |
| ycsb | medium | tx-dynamic | 30.482 | 100.0% | 1.000 | 1.581 | 1.661 | 0 | 0.000037 |
| ycsb | medium | tx-dynamic-priority | 32.117 | 100.0% | 1.000 | 1.459 | 1.621 | 0 | 0.000039 |
| ycsb | medium | tx-static | 33.101 | 100.0% | 1.000 | 1.520 | 1.615 | 0 | 0.000000 |
| ycsb | medium | tx-static-priority | 31.761 | 100.0% | 1.000 | 1.610 | 1.823 | 0 | 0.000000 |

## Ratios

| workload | profile | comparison | ratio | note |
| --- | --- | --- | ---: | --- |
| tpcc | high | op-dynamic-priority_vs_op-static | 1.016x |  |
| tpcc | high | op-dynamic-priority_vs_op-static-priority | 0.989x |  |
| tpcc | high | op-dynamic-priority_vs_op-dynamic | 0.998x |  |
| tpcc | high | tx-dynamic-priority_vs_tx-static | 1.166x |  |
| tpcc | high | tx-dynamic-priority_vs_tx-static-priority | 1.083x |  |
| tpcc | high | tx-dynamic-priority_vs_tx-dynamic | 1.089x |  |
| tpcc | high | tx-dynamic-priority_vs_op-dynamic-priority | 1.203x |  |
| tpcc | low | op-dynamic-priority_vs_op-static | 1.235x |  |
| tpcc | low | op-dynamic-priority_vs_op-static-priority | 1.243x |  |
| tpcc | low | op-dynamic-priority_vs_op-dynamic | 0.890x |  |
| tpcc | low | tx-dynamic-priority_vs_tx-static | 1.056x |  |
| tpcc | low | tx-dynamic-priority_vs_tx-static-priority | 1.084x |  |
| tpcc | low | tx-dynamic-priority_vs_tx-dynamic | 1.016x |  |
| tpcc | low | tx-dynamic-priority_vs_op-dynamic-priority | 1.028x |  |
| tpcc | medium | op-dynamic-priority_vs_op-static | 0.809x |  |
| tpcc | medium | op-dynamic-priority_vs_op-static-priority | 0.842x |  |
| tpcc | medium | op-dynamic-priority_vs_op-dynamic | 0.800x |  |
| tpcc | medium | tx-dynamic-priority_vs_tx-static | 1.107x |  |
| tpcc | medium | tx-dynamic-priority_vs_tx-static-priority | 1.134x |  |
| tpcc | medium | tx-dynamic-priority_vs_tx-dynamic | 1.038x |  |
| tpcc | medium | tx-dynamic-priority_vs_op-dynamic-priority | 1.058x |  |
| ycsb | high | op-dynamic-priority_vs_op-static | 1.152x |  |
| ycsb | high | op-dynamic-priority_vs_op-static-priority | 1.193x |  |
| ycsb | high | op-dynamic-priority_vs_op-dynamic | 1.006x |  |
| ycsb | high | tx-dynamic-priority_vs_tx-static | 1.027x |  |
| ycsb | high | tx-dynamic-priority_vs_tx-static-priority | 1.049x |  |
| ycsb | high | tx-dynamic-priority_vs_tx-dynamic | 1.093x |  |
| ycsb | high | tx-dynamic-priority_vs_op-dynamic-priority | 0.913x |  |
| ycsb | low | op-dynamic-priority_vs_op-static | 0.908x |  |
| ycsb | low | op-dynamic-priority_vs_op-static-priority | 1.125x |  |
| ycsb | low | op-dynamic-priority_vs_op-dynamic | 1.026x |  |
| ycsb | low | tx-dynamic-priority_vs_tx-static | 1.022x |  |
| ycsb | low | tx-dynamic-priority_vs_tx-static-priority | 1.195x |  |
| ycsb | low | tx-dynamic-priority_vs_tx-dynamic | 1.044x |  |
| ycsb | low | tx-dynamic-priority_vs_op-dynamic-priority | 0.923x |  |
| ycsb | medium | op-dynamic-priority_vs_op-static | 1.066x |  |
| ycsb | medium | op-dynamic-priority_vs_op-static-priority | 1.091x |  |
| ycsb | medium | op-dynamic-priority_vs_op-dynamic | 1.064x |  |
| ycsb | medium | tx-dynamic-priority_vs_tx-static | 0.970x |  |
| ycsb | medium | tx-dynamic-priority_vs_tx-static-priority | 1.011x |  |
| ycsb | medium | tx-dynamic-priority_vs_tx-dynamic | 1.054x |  |
| ycsb | medium | tx-dynamic-priority_vs_op-dynamic-priority | 0.918x |  |

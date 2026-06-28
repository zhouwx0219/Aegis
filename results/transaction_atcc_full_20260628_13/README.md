# Transaction-Level ATCC Full Matrix

This directory contains the full low/medium/high matrix for the new
`transaction-atcc-strict` strategy on agent-style YCSB and TPCC workloads.

## Raw Files

- `ycsb-low.json`, `ycsb-medium.json`, `ycsb-high.json`
- `tpcc-low.json`, `tpcc-medium.json`, `tpcc-high.json`
- `summary.csv`: generic summarizer output.
- `transaction_atcc_metrics.csv`: per-strategy metrics used for analysis.
- `transaction_atcc_ratios.csv`: transaction-level ATCC throughput ratios.

## Strategy Set

```text
occ
2pl-nowait
2pl-wait-die
mvcc-full
silo-full
tictoc-full
adaptive-op-strict
transaction-atcc-strict
```

## YCSB Transaction-Level ATCC

| profile | throughput | commit rate | attempts/task | vs OCC | vs MVCC | vs TicToc | vs operation ATCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 24.012 | 100.0% | 1.050 | 0.803x | 0.850x | 0.883x | 0.918x |
| medium | 36.399 | 100.0% | 1.000 | 3.833x | 1.022x | 1.001x | 1.032x |
| high | 50.500 | 100.0% | 1.000 | 45.740x | 2.204x | 1.827x | 0.980x |

## TPCC Transaction-Level ATCC

| profile | throughput | commit rate | attempts/task | vs OCC | vs MVCC | vs TicToc | vs operation ATCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 4.134 | 100.0% | 1.017 | 0.580x | 0.718x | 0.751x | 1.064x |
| medium | 23.242 | 100.0% | 1.100 | 3.979x | 4.383x | 2.972x | 0.894x |
| high | 40.609 | 100.0% | 1.017 | baseline 0 commits | baseline 0 commits | baseline 0 commits | 2.758x |

## Interpretation

- YCSB low: transaction-level ATCC is slower than OCC/MVCC/TicToc because the
  workload has little conflict and ATCC still pays policy and prelock overhead.
- YCSB medium: transaction-level ATCC slightly beats MVCC/TicToc and operation
  ATCC in this run, while keeping 100% commit rate.
- YCSB high: transaction-level ATCC is close to operation-level ATCC and much
  faster than traditional CC. It locks more hot operations, so it is slightly
  below operation-level ATCC.
- TPCC low: OCC is best. Transaction-level ATCC protects `next_order_id` even
  when conflict is dispersed, so it avoids aborts but pays lock wait.
- TPCC medium: transaction-level ATCC is far better than traditional CC, but
  below operation-level ATCC because it locks a broader hot set and has more
  attempts per task.
- TPCC high: traditional CC made no successful commits in this run. Both ATCC
  variants restored 100% commit rate; transaction-level ATCC had a shorter
  latency distribution in this single-seed matrix.

## Caveat

This is a full profile matrix but still single seed / single repeat. Use it for
directional comparison and debugging. Before treating the TPCC high 2.758x over
operation-level ATCC as a stable result, run multi-seed validation.


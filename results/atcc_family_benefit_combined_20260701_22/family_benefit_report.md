# ATCC Family Benefit Summary

## Target Check

| workload | profile | best traditional | best family | ratio | target | status | selected |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| ycsb | low | mvcc-full 39.223 | op-family 38.592 | 0.984x | 0.95x | pass | `{"occ": 300}` |
| ycsb | medium | mvcc-full 38.507 | op-family 40.100 | 1.041x | 1.02x | pass | `{"tictoc-full": 300}` |
| ycsb | high | tictoc-full 15.457 | tx-family 42.604 | 2.756x | 1.20x | pass | `{"transaction-atcc-strict": 300}` |
| tpcc | low | occ 6.096 | tx-family 7.821 | 1.283x | 0.95x | pass | `{"occ": 300}` |
| tpcc | medium | occ 11.074 | tx-family 16.196 | 1.462x | 1.02x | pass | `{"transaction-atcc-strict": 300}` |
| tpcc | high | occ 0.259 | tx-family 12.903 | 49.883x | 1.20x | pass | `{"transaction-atcc-strict": 300}` |

## Notes

- low target: best ATCC-family throughput >= 0.95x best traditional CC.
- medium target: best ATCC-family throughput >= 1.02x best traditional CC.
- high target marker used here: >= 1.20x best traditional CC as a concrete clearly-stronger threshold.
- Results are factual; failed rows are not rewritten.

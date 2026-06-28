# ATCC Final Handoff Results

This directory keeps only the core ATCC handoff artifacts.

## Files

```text
policies/
  ycsb-adaptive-readheavy-family-policy.json
  tpcc-family-policy-window.json
  tpcc-family-search.json
ycsb/
  summary.csv
  ycsb-low.json
  ycsb-medium.json
  ycsb-high.json
tpcc/
  summary.csv
  tpcc-low.json
  tpcc-medium.json
  tpcc-high.json
```

## Result Summary

YCSB throughput ratio of adaptive selection vs traditional CC:

| Profile | OCC | 2PL-nowait | 2PL-wait-die | MVCC | Silo | TicToc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 1.016x | 1.286x | 1.319x | 1.523x | 1.647x | 1.259x |
| medium | 1.514x | 2.333x | 2.559x | 0.901x | 2.251x | 0.826x |
| high | 5.819x | 18.809x | 17.444x | 1.766x | 11.504x | 1.358x |

TPCC throughput ratio of window-aware adaptive selection vs traditional CC:

| Profile | Selected | Throughput | Commit rate | vs OCC | vs MVCC | vs TicToc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| low | OCC | 5.415 | 100% | 1.116x | 1.205x | 1.290x |
| medium | operation-level ATCC | 20.106 | 100% | 2.154x | 2.377x | 1.840x |
| high | operation-level ATCC | 23.930 | 100% | 24.480x | 186.260x | 41.213x |

For detailed explanation and reproduction commands, see:

```text
docs/ATCC项目-2026.06.28.03/ATCC汇报.md
README.md
```

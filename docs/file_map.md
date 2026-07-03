# File Map

## Root

| Path | Purpose |
|---|---|
| `README.md` | Delivery-oriented project overview and commands. |
| `pyproject.toml` | Package metadata and CLI entry points. |
| `requirements.txt` | Minimal Python dependency list. |
| `build.sh` | Builds the Linux/WSL `cast_core` pybind11 extension. |

## Agent Package

| Path | Purpose |
|---|---|
| `agent/native.py` | Loads `cast_core` and reports actionable build errors. |
| `agent/cli/smoke.py` | Delivery smoke check for native KV, runtime, ATCC, YCSB, and TPC-C. |
| `agent/cli/benchmark.py` | Delivery benchmark wrapper over the retry experiment runtime path. |
| `agent/runtime/types.py` | Transaction state, events, snapshot values, and result dataclasses. |
| `agent/runtime/transaction.py` | Agent transaction manager and transaction lifecycle. |
| `agent/runtime/branching.py` | Candidate draft and branch-selection semantics. |
| `agent/runtime/commit_protocol.py` | Object locks and cost-aware commit orchestration. |
| `agent/runtime/cc_registry.py` | Strategy registry, strategy resolution, and pre-snapshot lock plans. |
| `agent/runtime/traditional_cc.py` | Agent-side traditional CC protocols. |
| `agent/runtime/adaptive.py` | Operation-level adaptive/ATCC policy implementation. |
| `agent/runtime/atcc.py` | Phase-aware and transaction-aware ATCC implementation. |
| `agent/runtime/hybrid.py` | Family-level adaptive strategy selector. |
| `agent/policies/__init__.py` | Delivery-facing policy exports for ATCC/adaptive/hybrid classes. |
| `agent/workloads/base.py` | Agent workload data model and replay helpers. |
| `agent/workloads/ycsb.py` | Agent-style YCSB workload. |
| `agent/workloads/tpcc.py` | Agent-style TPC-C workload. |
| `agent/workloads/layers.py` | Semantic/faithful workload layer factory. |
| `agent/experiments/*.py` | Compatibility wrappers for research experiment modules. |
| `agent/evaluation/*.py` | Research runners, ablation, training, search, reporting, and native baselines. |

## Native Core

| Path | Purpose |
|---|---|
| `core/storage/versioned_kv.h` | Versioned KV store interface. |
| `core/storage/dbx1000_versioned_kv.*` | DBx1000-backed KV implementation. |
| `core/txn/*.h` | Native commit protocol interfaces and cost-aware commit kernel. |
| `core/concurrency/*.h` | Native concurrency-control resolve primitives. |
| `core/branch/speculative_branch.h` | Native speculative branch types. |
| `core/intent/*.h` | Intent and policy-dispatch definitions. |
| `core/object/unified_object.h` | Versioned object/value definitions. |
| `core/bindings/cast_bindings.cpp` | pybind11 bindings exposed as `cast_core`. |

## Scripts And Tests

| Path | Purpose |
|---|---|
| `scripts/smoke.ps1` | PowerShell smoke wrapper that runs the WSL Python CLI. |
| `scripts/run_benchmark.ps1` | PowerShell benchmark wrapper for YCSB/TPC-C. |
| `scripts/research/` | Historical research scripts retained outside the delivery path. |
| `tests/` | Delivery acceptance tests only. |
| `tests_dev/` | Detailed development/research regression tests retained for maintainers. |

## Data And Docs

| Path | Purpose |
|---|---|
| `docs/architecture.md` | System architecture and responsibility boundaries. |
| `docs/quickstart.md` | Build, smoke, benchmark, and test commands. |
| `docs/file_map.md` | File-by-file responsibility map. |
| `docs/代码结构.md` | Chinese handoff structure document. |
| `results/` | Formal and generated experiment outputs. |
| `third_party/dbx1000/` | Vendored DBx1000 reference code. |

# Architecture

## Boundary

CAST-DAS keeps transaction context on the agent side:

- `agent/runtime/` owns transaction lifecycle, snapshots, read sets, candidate
  plans, lock orchestration, commit protocol selection, retry, and tracing.
- `agent/policies/` exposes ATCC/adaptive/hybrid policy concepts for the
  delivered project.  The implementation currently re-exports the established
  runtime modules for compatibility.
- `agent/workloads/` turns YCSB and TPC-C into agent-style tasks with
  candidates and stages.
- `core/` provides native C++ primitives: versioned KV, conditional writes,
  branch/write intent structures, concurrency-control resolve logic, and the
  cost-aware commit kernel.

The native core is not the agent transaction manager.  It is the storage and
validation substrate used by the Python runtime.

## Runtime Flow

```text
YCSB/TPC-C workload
  -> AgentTask with candidate operations and stages
  -> AgentTransactionManager.begin()
  -> snapshot + read set + candidate branches
  -> ConcurrencyControlRegistry resolves strategy
  -> optional pre-snapshot object locks
  -> CostAwareCommitProtocol
  -> native CostAsymmetricCommit + VersionedKVStore
  -> commit, reject, reselect, merge, or retry
```

## Delivery vs Research

Delivery users should start with:

```text
agent/cli/smoke.py
agent/cli/benchmark.py
scripts/smoke.ps1
scripts/run_benchmark.ps1
tests/test_smoke_runtime.py
tests/test_benchmark_cli.py
```

Research code remains available under:

```text
agent/evaluation/
agent/experiments/
scripts/research/
tests_dev/
```

This keeps the project runnable and explainable without deleting the detailed
experiments that support the paper results.

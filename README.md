# Aegis

Aegis is a minimal research prototype for a Data Agent System transaction
runtime.  The storage backend exposes versioned KV primitives, while the
agent-side runtime owns transaction semantics, candidate plans, concurrency
control selection, commit orchestration, and retry behavior.

## Quick Start

Build the native extension in WSL or Linux:

```bash
python3 -m pip install -e .
bash build.sh
```

Run the delivery smoke check:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke.ps1
```

Run a small YCSB/TPC-C benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_benchmark.ps1 `
  -Workload all `
  -Profile low `
  -Strategies quick `
  -TaskCount 10
```

The scripts use WSL because the checked native extension is built for Linux.

## Delivered Structure

```text
agent/cli/          Delivery command entry points: smoke and benchmark.
agent/runtime/      Agent-side transaction lifecycle, locks, CC registry, commit protocol.
agent/policies/     Public policy-layer exports for ATCC/adaptive/hybrid policies.
agent/workloads/    Agent-style YCSB/TPC-C workload models.
agent/experiments/  Compatibility wrappers for research experiments.
agent/evaluation/   Research runners, training, ablation, and reporting internals.
core/               C++ native KV, commit kernel, concurrency-control primitives, pybind.
scripts/            Delivery scripts. Historical research scripts live in scripts/research/.
tests/              Delivery acceptance tests only.
tests_dev/          Development and research regression tests retained for maintainers.
docs/               Architecture, quickstart, file map, and work records.
third_party/        Vendored DBx1000 reference code.
```

## Core Capabilities

- Versioned KV backend with atomic conditional writes.
- Agent-side transaction manager with snapshots, read sets, candidate plans, traces, and retries.
- Pluggable concurrency control: OCC, 2PL, MVCC, Silo, TicToc, operation-level ATCC, transaction-level ATCC, and adaptive-hybrid.
- Agent-style YCSB/TPC-C tasks with candidate plans and explore/refine/commit stages.
- Delivery benchmark CLI that runs the same runtime path as the research experiments.

## Main Commands

```bash
python3 -m agent.cli.smoke --json
python3 -m agent.cli.benchmark --workload ycsb --profile low --strategies quick --task-count 10
python3 -m unittest tests.test_smoke_runtime tests.test_benchmark_cli -v
```

Research and paper reproduction commands are intentionally kept out of the main
path.  See `agent/experiments/`, `agent/evaluation/`, and `scripts/research/`
when you need ablation, training, native DBx1000 baselines, or historical
comparison scripts.

## Documentation

- `docs/quickstart.md`: build and run instructions.
- `docs/architecture.md`: responsibility boundaries and system flow.
- `docs/file_map.md`: current file responsibilities and post-delivery structure.
- `docs/代码结构.md`: Chinese handoff document for the project structure.

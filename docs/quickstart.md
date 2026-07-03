# Quickstart

## 1. Build

Run inside WSL or Linux:

```bash
python3 -m pip install -e .
bash build.sh
```

`build.sh` produces `cast_core<python-extension-suffix>`, the pybind11 native
extension used by the Python runtime.

## 2. Smoke Check

From Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke.ps1
```

Or directly inside WSL:

```bash
python3 -m agent.cli.smoke --json
```

The smoke check verifies native loading, versioned KV conditional writes,
agent transaction commit, transaction-level ATCC planning, and one tiny
YCSB/TPC-C task.

## 3. Benchmark

From Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_benchmark.ps1 `
  -Workload all `
  -Profile low `
  -Strategies quick `
  -TaskCount 10
```

Direct WSL equivalent:

```bash
python3 -m agent.cli.benchmark \
  --workload all \
  --profile low \
  --strategies quick \
  --task-count 10 \
  --output results/delivery_benchmark.json
```

`quick` runs OCC, operation-level ATCC, and transaction-level ATCC.  `full`
also includes 2PL, MVCC, Silo, TicToc, and adaptive-hybrid.

## 4. Acceptance Tests

Run in WSL:

```bash
python3 -m unittest tests.test_smoke_runtime tests.test_benchmark_cli -v
```

The old fine-grained regression tests are retained in `tests_dev/` for
maintainers, but they are not the delivery acceptance suite.

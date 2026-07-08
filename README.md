# CAST-DAS

CAST-DAS is a compact research prototype for agent-side transaction semantics
and concurrency control over a versioned KV backend. The storage layer is kept
small on purpose: it exposes object reads, version reads, and conditional batch
writes. Transaction boundaries, read/write sets, conflict validation, retry
logic, and concurrency-control selection stay in the Python agent runtime.

## Current Layout

```text
agent/runtime/       Transaction lifecycle, snapshots, validation, commit path.
agent/cc/            Concurrency-control registry and strategy implementations.
agent/cc/atcc/       ATCC actions, feature extraction, policy table, rewards.
agent/workloads/     YCSB and TPC-C style agent workloads.
agent/benchmarks/    Concurrent and mixed agent/background benchmark harnesses.
agent/cli/           Smoke, compare, mixed, matrix, training, ablation commands.
core/storage/        Versioned KV interface and DBx1000-backed implementation.
core/intent/         Read/write intent DTOs exposed to Python bindings.
core/bindings/       Pybind11 native extension.
scripts/             PowerShell wrappers that run the Python CLIs through WSL.
tests/               Runtime, benchmark, ATCC, and CLI regression tests.
third_party/         Vendored DBx1000 reference code.
results/             Local generated experiment outputs, ignored by git.
```

Removed or folded historical areas:

```text
agent/evaluation/    Replaced by agent/benchmarks/ and agent/cli/.
agent/experiments/   Replaced by explicit benchmark and matrix CLIs.
agent/policies/      Folded into agent/cc/atcc/.
scripts/research/    Replaced by scripts/*.ps1 wrappers.
docs/                Folded into this README until stable paper docs are needed.
```

## Build And Verify

The native extension is built for Linux, so use WSL or a Linux shell from the
repository root.

```bash
python3 -m pip install -e .
bash build.sh
python3 -m unittest tests.test_smoke_runtime tests.test_compare_cli -v
```

For a full local sweep of available tests:

```bash
python3 -m unittest discover -v
```

## Main Commands

Smoke check:

```bash
python3 -m agent.cli.smoke --json
```

Concurrent barrier benchmark:

```bash
python3 -m agent.cli.compare \
  --workload ycsb \
  --level high \
  --cc occ,dynamic-atcc \
  --tasks 64 \
  --workers 8 \
  --reasoning-profile agentic
```

Mixed long-agent/short-background benchmark:

```bash
python3 -m agent.cli.mixed \
  --workload tpcc \
  --level high \
  --workload-profile paper \
  --background-mode procedure \
  --cc occ,dynamic-atcc \
  --duration 5 \
  --clients 48 \
  --agent-ratio 0.8 \
  --retry-until-commit
```

Paper-style matrix:

```bash
python3 -m agent.cli.matrix \
  --paper-style \
  --cc occ,2pl-nowait,2pl-wait-die,mvcc,silo,tictoc,dynamic-atcc \
  --duration 5 \
  --output results/paper_matrix.json
```

All-agent client sweep:

```bash
python3 -m agent.cli.matrix \
  --workloads ycsb,tpcc \
  --levels medium,high \
  --workload-profile paper \
  --background-mode procedure \
  --client-counts 8,16,24,32,40,48 \
  --agent-ratio 1.0 \
  --retry-until-commit \
  --duration 5 \
  --output results/all_agent_matrix.json
```

YCSB Zipfian override, for example medium with theta 0.8:

```bash
python3 -m agent.cli.matrix \
  --workloads ycsb \
  --levels medium \
  --workload-profile paper \
  --zipfian 0.8 \
  --client-counts 8,16,24,32,40,48 \
  --agent-ratio 0.8 \
  --background-mode procedure \
  --retry-until-commit \
  --duration 5 \
  --output results/ycsb_medium_zipf08_matrix.json
```

Train ATCC policy:

```bash
python3 -m agent.cli.train_atcc \
  --benchmark mixed \
  --workloads ycsb,tpcc \
  --levels low,medium,high \
  --workload-profile paper \
  --background-mode procedure \
  --budget-seconds 600 \
  --duration 2 \
  --clients 48 \
  --agent-ratio 0.8 \
  --retry-until-commit \
  --output results/atcc_policy.json
```

ATCC ablation matrix:

```bash
python3 -m agent.cli.matrix \
  --workloads ycsb,tpcc \
  --levels low,medium,high \
  --seeds 920104,920105,920106 \
  --client-counts 8,16,24,32,40,48 \
  --workload-profile paper \
  --background-mode procedure \
  --cc static-atcc,static-atcc-priority,trained-atcc,trained-atcc-priority \
  --duration 5 \
  --reasoning-profile agentic \
  --retry-until-commit \
  --policy results/atcc_policy.json \
  --policy-mode eval \
  --output results/atcc_2x2_ablation.json
```

PowerShell wrappers mirror these commands from Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke.ps1 -Json
powershell -ExecutionPolicy Bypass -File .\scripts\compare_cc.ps1 -Workload ycsb -Level high -Cc occ,dynamic-atcc
powershell -ExecutionPolicy Bypass -File .\scripts\mixed_benchmark.ps1 -Workload tpcc -Level high -WorkloadProfile paper -BackgroundMode procedure -Clients 48 -RetryUntilCommit
powershell -ExecutionPolicy Bypass -File .\scripts\mixed_matrix.ps1 -PaperStyle -Cc occ,2pl-nowait,2pl-wait-die,mvcc,silo,tictoc,dynamic-atcc
powershell -ExecutionPolicy Bypass -File .\scripts\train_atcc.ps1 -Benchmark mixed -Workloads ycsb,tpcc -Levels low,medium,high -Output results\atcc_policy.json
```

## Transaction Model

Each workload task maps to one agent transaction. The runtime records a
snapshot, read set, write set, metadata, and trace events. At commit, the
selected CC strategy may acquire runtime locks or reservations, then the runtime
validates versions and installs writes with `BatchPutIfVersion`.

Workloads expose object-level `read` and `write` operations only. Business
semantics such as append, delta, CAS, escrow, and commutative updates are not
visible to the CC layer.

Default strategy expansion for `--cc all`:

```text
occ, 2pl-nowait, 2pl-wait-die, mvcc, silo, tictoc, dynamic-atcc
```

Additional explicit ATCC ablation strategies:

```text
static-atcc             static threshold, no priority
static-atcc-priority    static threshold with runtime priority
trained-atcc            trained policy table, no priority
trained-atcc-priority   trained policy table with runtime priority
```

## ATCC Mechanism

`dynamic-atcc` chooses one action per agent transaction from the policy/action
space below:

```text
occ, write-validate, reserve-hot, reserve-hot-rw,
reserve-read-write-set, lock-before-commit, retry-protect
```

The feature key is built from workload, task type, conflict level, contention
bucket, agent-cost bucket, write-set bucket, and retry stage. Training updates a
JSON policy table from observed commit/abort outcomes, elapsed time, lock wait,
lock hold time, wasted reasoning cost, skipped reasoning, and background
pressure. Evaluation loads the policy with `--policy` and freezes it with
`--policy-mode eval`.

Runtime execution has three paths:

```text
optimistic          execute first, validate at commit
deferred-protect    reason first, begin/protect near commit
early-protect       protect hot or retry transactions before object access
```

Priority is a runtime scheduling hint, not a separate CC protocol. It is derived
from learned row priority, retry count, estimated wasted reasoning cost, risk
score, and transaction start order. Current experiments show that priority can
help under high client pressure, but hard-coded priority can also reduce total
throughput, so the reliable claim should be based on the 2x2 ablation rather
than on a single hand-tuned priority rule.

## Benchmark Semantics

`agent.cli.compare` runs fixed task batches. It creates a fresh runtime per
strategy, uses the same seed-generated task sequence, and executes tasks in
barrier batches controlled by `--workers`.

`agent.cli.mixed` runs wall-clock mixed contention with long agent transactions
and short background transactions. `--background-mode procedure` runs short
YCSB/TPC-C tasks against the same store and is the preferred paper-style setup.
Set `--agent-ratio 1.0` with `--clients` or `--client-counts` to run an all-agent
sweep with zero background workers.

With `--retry-until-commit`, agent latency is measured from first submission to
eventual commit, including retries and injected reasoning delay. Token cost
uses:

```text
T_avg = (1 + R_abort) * N_ops * omega
omega = 2703 tokens per operation
```

The paper YCSB profile uses 10 operations per transaction and these contention
settings:

```text
low     95/5 read/write, uniform
medium  90/10 read/write, 10% hot tuples receive about 50% of accesses
high    50/50 read/write, 10% hot tuples receive about 75% of accesses
```

By default, these profiles keep the hotspot access model above. Passing
`--zipfian <theta>` or `--ycsb-zipf-theta <theta>` switches YCSB to global
Zipfian record sampling for that run and records the theta in the JSON output.
This is intended for sweeps such as YCSB medium theta 0.7/0.8 and YCSB high
theta 0.99.

The paper TPC-C profile uses `NewOrder` and `Payment` transactions.

## Results Policy

Generated JSON files in `results/` and root-level `paper_*_results.json` are
local experiment outputs and are ignored by git. Keep only artifacts that are
needed to reproduce a report or paper figure, and document the command that
created them. If an artifact must be versioned, add it explicitly with
`git add -f`.

Recommended retained artifacts after the current ATCC work:

```text
results/paper_like_atcc_small_policy_v5_write_validate.json
results/atcc_2x2_ablation_smoke_v2_priority_reservation.json
results/atcc_2x2_ablation_high_clients_smoke_v1.json
results/atcc_2x2_ablation_retry_priority_v1.json
results/atcc_2x2_ablation_high_clients_retry_priority_v1.json
```

Older `paper_like_atcc_*_v*.json` and `force_tpcc_medium_*` files are historical
tuning outputs. They should be deleted once their conclusions have been copied
into the report or replaced by a final multi-seed matrix.

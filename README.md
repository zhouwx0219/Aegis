# Aegis

Aegis is a research prototype for agent-side transaction execution and adaptive concurrency control over a versioned key-value backend. It combines a Python transaction runtime with a small C++/pybind11 storage extension and provides repeatable YCSB, TPC-C, and enterprise credit-review experiments.

This repository is intended for mechanism evaluation and paper reproduction, not as a production database or transaction service.

## What Is Included

- A versioned object store backed by embedded DBx1000 components.
- Agent transactions with snapshots, read/write sets, conditional commit,
  retry accounting, reasoning phases, token-cost accounting, and optional undo
  logging.
- Traditional CC baselines: OCC, 2PL No-Wait, 2PL Wait-Die, MVCC, Silo,
  TicToc, Bamboo, and Polaris.
- `paper-atcc`, the Aegis paper path with phase-aware protection, dynamic
  priority, Wound-Wait lock handling, and delayed write application.
- Legacy/static/trained ATCC variants used for comparison and ablation.
- YCSB and TPC-C benchmark profiles, mixed agent/background execution, and a
  streaming Credit Review workload.
- Fixed-trace runners for replaying the same concrete workload across Aegis and
  patched DBx1000-family baselines.
- Training, ablation, bounded experiment, summarization, and archived-result
  verification scripts.

## Repository Layout

```text
agent/runtime/       Transaction lifecycle, versioning, ATCC locks, priority,
                     trajectories, undo logging, and commit instrumentation.
agent/cc/            Traditional CC strategies and ATCC policy implementations.
agent/benchmarks/    Concurrent, mixed, and multi-seed benchmark harnesses.
agent/workloads/     YCSB, TPC-C, and Credit Review workload definitions.
agent/cli/           Smoke, comparison, matrix, training, and ablation CLIs.
core/storage/        Versioned KV interfaces and the DBx1000-backed store.
core/bindings/       pybind11 module definition for `cast_core`.
core/intent/         Read/write intent types shared with the native layer.
scripts/unified_trace/
                     Fixed-trace generation, fair replay, paper matrices,
                     Credit Review, sensitivity studies, and ablations.
scripts/external_cc/ Adapters for native Bamboo/Polaris DBx1000 repositories.
tests/               Runtime, CLI, workload, ATCC, and verifier tests.
third_party/dbx1000/ Embedded DBx1000 sources used to build `cast_core`.
results/             Generated local artifacts; ignored by Git.
```

## Requirements

- Linux or WSL. The supplied native build script uses `g++`, `-fPIC`, and
  POSIX threads.
- Python 3.10 or newer, including the matching Python development headers.
- A C++17 compiler.
- `pybind11>=2.10`.

The repository stores C++ source code, not a prebuilt native module. Shared
libraries are platform- and Python-ABI-specific and are ignored by Git, so
`cast_core` must be built with the same Python interpreter that will run Aegis.

## Install And Build

From the repository root in Linux or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
bash build.sh
```

`build.sh` compiles `core/bindings/cast_bindings.cpp`, the versioned storage
implementation, and the required embedded DBx1000 sources into a module named
`cast_core` in the repository root.

Confirm that the Python and native layers load together:

```bash
python3 -c "import cast_core; print(cast_core.__file__)"
python3 -m agent.cli.smoke --json
```

## Test

Run the self-contained runtime and workload tests after building `cast_core`:

```bash
python3 -m unittest \
  tests.test_smoke_runtime \
  tests.test_compare_cli \
  tests.test_credit_review_workload \
  tests.test_paper_atcc_correctness \
  tests.test_verify_aegis_reproduction -v
```

The two archived-matrix test modules additionally require acceptance manifests
and their referenced files under `results/reproduction/`. Those generated
artifacts are intentionally not committed to this repository. Run the complete
discovery suite only after restoring that archive:

```bash
python3 -m unittest discover -v
```

## Quick Start

Compare all default CC strategies on a concurrent YCSB task batch:

```bash
python3 -m agent.cli.compare \
  --workload ycsb \
  --level high \
  --workload-profile paper \
  --cc all \
  --tasks 64 \
  --workers 8 \
  --reasoning-profile agentic \
  --output results/ycsb_compare.json
```

Run mixed long-agent and short-background TPC-C transactions:

```bash
python3 -m agent.cli.mixed \
  --workload tpcc \
  --level high \
  --workload-profile paper \
  --background-mode procedure \
  --cc occ,2pl-wait-die,bamboo,silo,polaris,paper-atcc \
  --clients 40 \
  --agent-ratio 0.8 \
  --duration 5 \
  --retry-until-commit \
  --output results/tpcc_mixed.json
```

Run the built-in paper-style multi-seed matrix:

```bash
python3 -m agent.cli.matrix \
  --paper-style \
  --cc 2pl-wait-die,bamboo,silo,polaris,paper-atcc \
  --duration 5 \
  --output results/paper_matrix.json
```

Use `--policy results/paper_policy.json --policy-mode eval` when evaluating a
previously trained policy artifact.

## Fixed-Trace Evaluation

Fixed traces separate workload generation from execution. A trace contains the
ordered reads and writes, client role, reasoning delays, seed, and workload
metadata for every transaction. Replaying one file across systems avoids
silently comparing different transaction streams.

Generate a trace:

```bash
python3 scripts/unified_trace/generate_castdas_trace.py \
  --output results/traces/ycsb_high_c24.csv \
  --variant ycsb_high_z099 \
  --clients 24 \
  --agent-ratio 1.0 \
  --seed 920104 \
  --transactions-per-worker 128 \
  --reasoning-profile agentic
```

Replay it through the internal fair runner:

```bash
python3 scripts/unified_trace/run_castdas_trace_fair.py \
  --trace results/traces/ycsb_high_c24.csv \
  --output results/ycsb_high_c24_results.csv \
  --cc 2pl-wait-die,bamboo,silo,polaris,paper-atcc \
  --paper-policy results/paper_policy.json \
  --policy-mode eval \
  --max-attempts 1 \
  --measure-seconds 2
```

Paper replay disables retries by default. For a retry experiment, set
`--max-attempts` above `1` and explicitly pass `--allow-retries`.

The higher-level `run_unified_trace_matrix.py` command generates and runs the
standard YCSB/TPC-C variants, client counts, agent ratios, and seeds. It can
also invoke patched native Bamboo and Polaris repositories when those external
checkouts are available.

## Paper ATCC

`paper-atcc` is the current Aegis mechanism. It is separate from the legacy
`dynamic-atcc` strategy retained for earlier experiments.

At a high level, the runtime:

1. Starts a transaction optimistically and records its observed access set.
2. Invokes a phase policy as execution reveals contention and transaction
   shape.
3. Monotonically expands protection over hot/cold reads and writes.
4. Validates already observed reads before retroactively acquiring protection.
5. Resolves protected conflicts with dynamic-priority Wound-Wait.
6. Enters a non-preemptible commit phase, validates remaining optimistic reads,
   installs buffered writes, publishes the commit, and releases protection.

Delayed write application and priority are independent ablation switches. The
runtime also records lock acquisition, lock wait, wounds, validation failures,
commit-phase timing, retries, wasted reasoning, and token-cost metrics.

Train a compiled paper policy from one or more trajectory files:

```bash
python3 scripts/train_paper_atcc.py \
  --trajectory results/training/trajectory.json \
  --output results/paper_policy.json \
  --report results/paper_policy_report.json \
  --generation 1
```

For coordinated or matrix training, use
`scripts/train_paper_atcc_coordinated.py` or
`scripts/train_paper_atcc_matrix.py`.

## Credit Review Workload

Credit Review models streaming enterprise credit decisions whose access sets
are discovered online. Transactions combine company, sector, region,
portfolio, committee, compliance, exposure, and review-queue objects under a
Zipfian company distribution. This workload is used to evaluate expensive
agent reasoning, commit admission, and tail latency without relying on a
predeclared complete write set.

```bash
python3 scripts/unified_trace/run_credit_review_experiment.py \
  --output-dir results/credit_review \
  --paper-policy results/paper_policy.json \
  --clients 8,16,24,32,40 \
  --systems 2pl-wait-die,bamboo,silo,polaris,paper-atcc \
  --repeats 3 \
  --measure-seconds 3
```

## Bounded Experiment Groups

The grouped runner covers scalability, agent/worker decoupling, contention,
transaction shape, read/write ratio, reasoning ratio control, and ATCC
ablations:

```bash
python3 scripts/unified_trace/run_aegis_two_hour_experiments.py \
  --group all \
  --output-dir results/aegis_groups \
  --paper-policy results/paper_policy.json
```

Use `--quick` for one repeat per configuration, `--points` to select specific
parameter values, and `--force` to recompute existing rows. To refresh only
Aegis rows in an existing group, use
`scripts/unified_trace/refresh_aegis_group.py`.

Mechanism-only DWA and priority checks can be run separately:

```bash
python3 scripts/unified_trace/run_atcc_mechanism_microbench.py \
  --output results/atcc_mechanisms.csv \
  --ablation-dir results/atcc_mechanism_ablation
```

## Workload And Retry Semantics

The paper YCSB profile uses 10 operations per transaction. Low, medium, and
high levels vary the read/write ratio and contention distribution; explicit
Zipfian sweeps can override the profile with `--ycsb-zipf-theta` or
`--zipfian`.

The paper TPC-C profile uses NewOrder and Payment transactions with configurable
warehouse counts. Mixed `procedure` mode executes short background procedures
against the same store as agent transactions.

Agent latency is measured from first submission through final outcome. When
retries are enabled, it includes retry backoff and repeated reasoning. Token
metrics use the configured `--tokens-per-operation` value, which defaults to
2703.

Do not mix no-retry and retry results in one claim. The fixed-trace runner uses
one attempt unless retries are explicitly enabled, while the bounded paper
runner defaults to six total attempts (the initial attempt plus five retries).

## External Bamboo And Polaris Baselines

`scripts/external_cc/` patches disposable external DBx1000-family checkouts and
runs CAST-DAS-shaped client mixes in their native engines. This path is a
benchmark adapter; it does not port Aegis transaction semantics or ATCC into
those systems. See `scripts/external_cc/README.md` for the required checkout
names and remote wrapper command.

## Reproduction Verification

The verification scripts check archived file hashes, seed isolation, system
coverage, retry invariants, and recomputed metrics. Generated archives are not
part of the Git tree, so provide a manifest and all files referenced by it:

```bash
python3 scripts/verify_aegis_reproduction.py \
  --manifest results/reproduction/experiment_manifest.json

python3 scripts/verify_five_retry_paper_matrix.py \
  --manifest results/reproduction/five_retry_paper_acceptance_manifest.json \
  --require-all

python3 scripts/verify_zero_retry_paper_matrix.py \
  --manifest results/reproduction/zero_retry_paper_acceptance_manifest.json \
  --require-full-paper-claims
```

## Results Policy

`results/`, root-level paper result JSON files, native modules, build outputs,
and external repositories are ignored by Git. Keep the command line, code
revision, policy file, trace, seeds, retry budget, and manifest together for
every result used in a report. A CSV or JSON file without that provenance is
not sufficient for a reproducible comparison.

# ASTRA Core

ASTRA is a compact agent-side transaction runtime. A task may generate multiple
candidate plans, operate on heterogeneous versioned objects, and commit through
a selectable concurrency-control module.

## Architecture

```text
Agent task -> K candidates -> ASTRA transaction runtime
           -> pluggable CC validation -> atomic version checks and writes
           -> DBx1000 Catalog + table_t + row_t + IndexHash
```

- `core/storage` exposes a narrow `VersionedKVStore` contract. The default
  `Dbx1000VersionedKVStore` stores key, value, existence, and a monotonic version
  in DBx1000 rows and uses DBx1000's hash index for lookup.
- `core/concurrency` defines the `ConcurrencyControl` plugin interface. The
  included `SemanticConcurrencyControl` understands overwrite, ordered or
  commutative append, delta, constrained delta, and CAS. Strict validation
  modules provide OCC and DBx1000-style traditional comparison points.
- `core/txn` defines a `CommitProtocol` interface. The default
  `CostAsymmetricCommit` validates candidates in quality order and performs
  direct commit, semantic resolution, candidate reselection, or an explicit
  regeneration request. Multi-object writes use one atomic version-check batch.
- `agent/runtime` owns snapshots, model/tool traces, candidate construction,
  CC registration, and real regeneration callbacks. Its modules map directly to
  the four pluggable design points:
  `branching.py` for multi-branch transaction semantics, `cc_registry.py` plus
  `core/concurrency` for semantic-aware CC modules, `commit_protocol.py` plus
  `core/txn` for cost-aware commit protocols, and `adaptive.py` for ATCC policy
  tables.
- `agent/workloads` contains provider-neutral Agent-YCSB and Agent-TPCC
  workloads derived from DBx1000's benchmark families, split into faithful and
  semantic agent layers.
- `agent/evaluation/dbx1000_native.py` runs DBx1000's own executable as an
  external native baseline, so native OCC/MVCC/TicToc/Silo/2PL results are not
  confused with ASTRA's agent-side adapters.

DBx1000 is an in-memory OLTP research engine. In ASTRA it is deliberately used
as a process-local, non-durable, versioned KV substrate; agent/runtime owns the
transaction semantics, CC policy selection, read-set validation, regeneration
boundary, and trace. The vendored upstream revision and local embedding changes
are recorded in `third_party/dbx1000/UPSTREAM.md`.

## Build And Test

On Linux or WSL:

```bash
python3 -m pip install -e .
bash build.sh
python3 -m unittest discover -s tests -v
```

`pyproject.toml` declares the Python package metadata, the `pybind11`
dependency, and the `astra-cc-matrix` and `astra-dbx1000-native` console
scripts. The native `cast_core` extension is still built explicitly by
`build.sh`, because the DBx1000 embedding is intentionally kept as a narrow
local adapter.
If importing `agent.runtime` or running the CLI reports that `cast_core` is not
available, rebuild with `bash build.sh` using the same Linux/WSL Python runtime.

## Versioned KV

```python
import cast_core as cc

store = cc.Dbx1000VersionedKVStore(
    max_key_bytes=512,
    max_value_bytes=8192,
    bucket_count=4096,
)
store.put("inventory:42", "10")
snapshot = store.get("inventory:42")
updated = store.put_if_version("inventory:42", snapshot.version, "9")
```

Deletes leave a versioned tombstone, so deleting and recreating a key cannot
reintroduce an old version. The transaction kernel uses `BatchPutIfVersion` for
all-or-nothing validation and writes across multiple keys.

## CC Plugins

The runtime includes `semantic` (also available as the compatibility aliases
`cast` and `semantic-v2`), `occ`, DBx1000-style strict validation adapters
(`mvcc`, `silo`, `tictoc`, and `2pl`), plus an ATCC-shaped `adaptive` policy
table. The default table chooses `semantic` for rebaseable intents, may choose
`2pl` for wide strict read/write footprints, and falls back to `occ`:

```python
import cast_core as cc
from agent.runtime import AgentTransactionManager

manager = AgentTransactionManager()
manager.register_cc("semantic-v2", cc.SemanticConcurrencyControl())
print(manager.cc_strategy("adaptive"))
print(manager.module_catalog()["commit_protocol"]["name"])

manager.register_object("counter", 0, kind="counter")
txn = manager.begin("task-1")
txn.add_candidate("candidate-1", quality=1, gen_cost=0).delta("counter", 1)
result = txn.commit(strategy="adaptive")
```

To add another agent-side CC module, implement `ConcurrencyControl::Name`,
`Family`, optional metadata such as `RequiresObjectLocks`, and `Resolve`, expose
the class in `cast_bindings.cpp`, and register an instance under a runtime name.
`StrictValidationConcurrencyControl` can also be used to register a named
traditional baseline over the same agent read/write-set validation contract.
Storage and commit orchestration do not need to change.

The DBx1000-style names are comparison adapters at the ASTRA layer. They do not
delegate transaction execution to DBx1000's original benchmark threads or lock
managers; DBx1000 remains the versioned KV backend.

For authoritative DBx1000 native CC baselines, use `astra-dbx1000-native`. It
copies the vendored DBx1000 tree into a temporary directory, patches `CC_ALG`,
builds `rundb`, runs the original benchmark path, and parses DBx1000's
`[summary]` output:

```bash
python3 -m agent.evaluation.dbx1000_native \
  --workload ycsb \
  --strategies occ,mvcc,tictoc,silo,no_wait \
  --threads 8 \
  --output dbx1000-ycsb-native.json
```

The adaptive table is explicit and replaceable from Python:

```python
from agent.runtime import AdaptivePolicyRule, AdaptivePolicyTable

manager.set_adaptive_policy(
    AdaptivePolicyTable(
        rules=(
            AdaptivePolicyRule(
                name="strict-wide-to-2pl",
                target_strategy="2pl",
                min_reads=2,
                min_writes=4,
                overwrite_only=True,
            ),
        ),
        fallback_strategy="occ",
    )
)
```

Operation-level ATCC has two execution variants. `adaptive-op` keeps semantic
optimistic resolution and adds commit-phase locks for pessimistic targets.
`adaptive-op-strict` is the pure traditional switch: optimistic targets use
strict OCC, while pessimistic targets are locked before the snapshot and held
through commit. `2pl-pre` is the full pre-snapshot 2PL baseline. Pre-snapshot
strategies must be evaluated with
`agent_path_experiment --execution-mode concurrent`.

See `docs/pre_snapshot_atcc_tpcc_report.md` for training, reproduction, and
large TPC-C NewOrder results.

The online ATCC policy table can be selected with
`agent_path_experiment --operation-policy atcc`. It combines object-role priors
with runtime conflict and lock-wait feedback while staying on the strict
OCC/2PL path. See `docs/online_atcc_experiment_zh.md` for the focused ATCC
TPC-C/YCSB experiments and reproduction commands.

The migrated phase-aware ATCC workflow can be reproduced as a fixed profile
suite. It trains the tabular ATCC policy artifact, evaluates OCC, full
pre-snapshot 2PL, and loaded ATCC on Agent-YCSB/Agent-TPCC low/medium/high
profiles, and writes both JSON artifacts and a Markdown summary table:

```bash
python3 -m agent.evaluation.atcc_profile_runner \
  --profiles all \
  --output-dir results/phase_atcc_profiles
```

For a quick smoke run, lower `--train-task-count`, `--eval-task-count`,
`--eval-repeats`, and `--workers`. The installed console script is
`astra-atcc-profiles`. Phase-aware ATCC policy artifacts are versioned; current
artifacts use schema version 2 and include `class=<object_class>` in Q-table
state keys, so formal profile runs should retrain artifacts after changing the
state dimensions. Retry-evaluation reports include `policy_artifact_schema`
when `--policy-artifact` is used, making legacy or incompatible artifacts visible
in the result JSON. See `docs/phase_atcc_v2_profile_experiment_zh.md` for the
current v2 profile experiment and `docs/phase_atcc_v2_smoke_report_zh.md` for a
small smoke run. The current ATCC module design is summarized in
`docs/atcc_design_zh.md`, and the latest local pressure-scale profile analysis is
in `docs/phase_atcc_v2_scale_experiment_zh.md`. For a workload closer to the
original ATCC paper's agent-starvation setting, see
`docs/atcc_paperlike_starvation_experiment_zh.md`.

## Agent Workloads

Both workloads emit immutable, JSON-serializable `AgentTask` values containing
a natural-language request, task context, K ranked candidates, and typed
operations. They can be generated without an LLM and executed by any adapter
that understands the common operation model.

```python
from agent.runtime import AgentTransactionManager
from agent.workloads import (
    TPCCAgentWorkload,
    YCSBAgentWorkload,
    execute_task,
    register_workload,
)

manager = AgentTransactionManager()
workload = YCSBAgentWorkload()
register_workload(manager, workload)

for task in workload.generate_tasks(10, seed=7):
    result = execute_task(manager, task, cc="semantic")
```

`YCSBAgentWorkload` and `TPCCAgentWorkload` are the semantic agent layers
(`agent-ycsb-semantic` and `agent-tpcc-semantic`): they support K candidates and
semantic intents. `YCSBFaithfulAgentWorkload` and `TPCCFaithfulAgentWorkload`
are the faithful agent layers (`agent-ycsb-faithful` and
`agent-tpcc-faithful`): they use one candidate per task and keep closer
DBx1000-derived request surfaces. TPC-C faithful/native comparability is limited
to DBx1000's Payment/NewOrder surface; order-status, delivery, and stock-level
belong to the semantic agent extension.
Each workload exposes a JSON-serializable manifest describing the DBx1000
source files, preserved benchmark semantics, and ASTRA agent adaptations.

Use `agent.evaluation.run_strategy_matrix` to compare CC strategies on the
same generated tasks. `contention_window > 1` begins multiple transactions
from the same snapshot before committing them, which makes strict OCC-style
and semantic-rebase behavior comparable without changing the workload model.

```python
from agent.evaluation import run_strategy_matrix
from agent.workloads import TPCCAgentWorkload, TPCCConfig

workload = TPCCAgentWorkload(
    TPCCConfig(transaction_mix=(("new_order", 1.0),))
)
summaries = run_strategy_matrix(
    workload,
    ("semantic", "adaptive", "occ", "2pl"),
    task_count=100,
    seed=7,
    contention_window=8,
)
for summary in summaries:
    print(summary.to_dict())
```

The same matrix runner is available as a CLI. After `pip install -e .`, the
equivalent console script is `astra-cc-matrix`.

```bash
python3 -m agent.evaluation.cc_matrix_cli \
  --workload tpcc \
  --workload-layer semantic \
  --transaction-mix new_order:1.0 \
  --strategies semantic,adaptive,occ,2pl \
  --adaptive-policy new-order \
  --task-count 100 \
  --seed 7 \
  --repeats 5 \
  --contention-window 8 \
  --format json

python3 -m agent.evaluation.cc_matrix_cli \
  --workload ycsb \
  --workload-layer faithful \
  --strategies semantic,occ \
  --task-count 100 \
  --repeats 5 \
  --format csv \
  --csv-section aggregates \
  --output ycsb_cc_matrix.csv
```

See `docs/experiments.md` for the full reproduction and analysis guide,
including the NewOrder ATCC-style policy-table sweep.

## Regeneration Boundary

A conflict never relabels an old plan as newly generated. When all candidates
conflict, the C++ kernel returns `regenerate_required`. If the caller supplies a
regenerator, the Python runtime refreshes the snapshot and invokes it; otherwise
the transaction aborts without overwriting concurrent data.

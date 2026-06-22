# ASTRA: Agent-Side Transactions

ASTRA is a research prototype for **agent-side transaction execution and semantic-aware concurrency control** in data agent systems.

The project targets the AI-OLTP gap: recent data-agent systems such as Palimpzest focus on read-heavy AI-OLAP workloads, while many agent tasks also write shared state such as seats, inventory, counters, order status, and text. ASTRA moves the transaction boundary to the agent side, where object semantics and generated candidates are still visible.

## Naming

| Name | Meaning in the paper | Code / artifact |
|---|---|---|
| **ASTRA** | The full system: Agent-Side Transactions | This repository and paper narrative |
| **CAST** | ASTRA's cost-asymmetric commit protocol | `core/txn/cost_asymmetric_commit.h` |
| **HYBRID** | The experimental label for ASTRA's semantic-aware CC strategy | Figures and CSV results |
| **CSI-SS** | The mixed correctness boundary | `docs/ISOLATION_LEVELS.md`, `docs/PROOFS.md` |

We no longer use CAST as the whole-system name. CAST is the commit protocol inside ASTRA: `direct -> merge -> reselect -> regenerate`.

## What Is Implemented

```text
Python agent/runtime layer
  - AgentTransactionManager: begin -> snapshot/read -> model/tool calls
    -> candidate branches -> commit/reject trace
  - Mock and DeepSeek OTA candidate generators
  - Synthetic, true-concurrency, VitaBench-style, and LLM-in-the-loop experiments

pybind11
  - In-process bridge to the C++ kernel (`cast_core`)

C++ kernel
  - VersionedObjectStore: minimal versioned KV reference store
  - WriteIntent + PolicyDispatcher: typed intents and semantic rebase
  - CostAsymmetricCommit: CAST direct/merge/reselect/regenerate
  - HybridDispatcher: per-object/intent HYBRID CC policy
  - EscrowAccount: constrained commutative reservation, no oversell
```

Important scope decision for the CCFA paper: **real persistent DB backend**, **crash recovery**, and an **adaptive optimizer for choosing `k` or policy** are future work. They are not required for the current transaction/CC contribution.

## Repository Layout

```text
cast-das/
├── core/                         # C++ transaction / concurrency kernel
│   ├── storage/                  # versioned KV reference boundary
│   ├── intent/                   # write-intent classification and rebase
│   ├── txn/                      # CAST commit protocol
│   ├── concurrency/              # HYBRID dispatcher and escrow
│   └── bindings/                 # pybind11 module
├── agent/
│   ├── runtime/                  # upper transaction lifecycle
│   ├── llm/                      # DeepSeek OTA agent operator
│   ├── workloads/                # synthetic / VitaBench-style workloads
│   ├── experiments/              # experiments and paper figure generator
│   └── integrations/             # VitaBench OTA integration
├── docs/                         # paper skeleton, proofs, artifact guide
├── scripts/                      # smoke and reproduction entrypoints
├── figures/                      # schematic figures
└── build.sh                      # build `cast_core`
```

## Quick Start

On WSL/Linux:

```bash
python3 -m pip install -r requirements.txt   # needed for legacy PNG scripts
bash build.sh
bash scripts/smoke.sh
```

`bash scripts/smoke.sh` builds the C++ extension, imports `cast_core`, runs the transaction lifecycle checks, runs the end-to-end CAST demo, verifies the correctness boundary, and regenerates the paper-facing SVG figures.

If `matplotlib` is unavailable, the smoke path still works because `agent/experiments/paper_figures.py` uses only the Python standard library.

## Core Paper Figures

The CCFA-facing figure set is generated into:

```text
agent/experiments/results/paper_figures/
```

Regenerate from existing CSV/JSON results:

```bash
python3 agent/experiments/paper_figures.py
```

Current core figures:

1. `fig1_cost_asymmetry.svg`: cost asymmetry, mergeability, and boundary cases.
2. `fig2_true_concurrency.svg`: measured throughput/latency under true concurrency.
3. `fig3_semantic_reselect.svg`: semantic validation and multi-candidate reselect.
4. `fig4_escrow_correctness.svg`: constrained commutative writes and no oversell.
5. `fig5_llm_in_loop.svg`: real DeepSeek trace evidence and replay comparison.
6. `fig6_baseline_family.svg`: OCC/Silo/TicToc/MVCC/2PL/HYBRID baseline family.
7. `fig7_scale_out.svg`: larger task counts and thread scale-out.
8. `fig8_agent_aware_baselines.svg`: OCC+K, HYBRID-K1, merge-all, and safety-aware baselines.
9. `fig9_hotspot_mixed.svg`: hotspot mixed-object workload with throughput and generation-call efficiency.
10. `fig10_vitabench_authoritative.svg`: VitaBench environment-derived OTA write workload over real shared resources.
11. `fig11_rigorous_vitabench.svg`: large-scale benchmark reporting throughput, P95 latency, and SLA success rate.

See `docs/CCFA_ARTIFACT_GUIDE.md` for the claim-to-figure mapping.

## Reproduction

Curated CCFA reproduction flow:

```bash
bash scripts/reproduce_ccfa.sh
```

From Windows PowerShell:

```powershell
.\scripts\reproduce_ccfa.ps1
```

The script always rebuilds the kernel, runs dependency-light checks, reruns the cost-asymmetry sweep, runs the CCFA baseline/scale/agent-aware/hotspot extension experiments, and regenerates the SVG figure set. If `matplotlib` is installed, it also reruns the legacy PNG experiments.

The default quick profile covers the expanded baseline family, up to 32 worker threads, up to 10k synthetic tasks, and 5k booking tasks. Use `bash scripts/reproduce_ccfa.sh large` for the larger stress profile with up to 64 worker threads, 50k synthetic tasks, and 10k booking tasks.

Real DeepSeek calls are intentionally not run by default:

```bash
DEEPSEEK_API_KEY=... python3 agent/experiments/llm_in_the_loop.py all --tasks 60 --k 3 --conc 8
```

The current real run uses DeepSeek `deepseek-chat` with 60 OTA tasks, `K=3`, API
concurrency 8, replay threads 8, 3 replay seeds, and `speed=20`. It produced 0
API errors, mean real generation latency 1.32s, 100% tasks with at least two
alternatives, live HYBRID `oversell=0`, and same-trace replay HYBRID throughput
8.2% above OCC+K and 3.05x 2PL. See
`agent/experiments/results/llm_analysis.md`.

For a no-key dry run:

```bash
bash scripts/reproduce_ccfa.sh llm-mock
```

VitaBench-derived authoritative write workload:

```bash
bash scripts/reproduce_vitabench.sh
```

This command installs the external VitaBench package into `/tmp/vb`, verifies that an official OTA order tool decrements shared `quantity`, runs the CC benchmark on real VitaBench OTA resources, and regenerates `fig10_vitabench_authoritative.svg`.

DBx1000-backed in-memory Vita workload:

```bash
bash scripts/reproduce_dbx1000_vita.sh
```

This builds the vendored DBx1000 ASTRA/Vita runner and compares DBx-style OCC/MVCC/TicToc/Silo/2PL baselines with ASTRA-HYBRID on the same VitaBench-derived shared-resource workload. The vendored DBx1000 tree also registers `CC_ALG=HYBRID` as a native compile-time CC option. See `docs/DBX1000_INTEGRATION.md` for scope, parameters, and caveats.

Additional DBx1000-backed sensitivity sweeps:

```bash
bash scripts/reproduce_dbx1000_vita_sensitivity.sh
```

This varies candidate count and worker threads under high contention and writes `agent/experiments/results/dbx1000_vita_sensitivity_summary.csv`.

DBx1000-backed hot-resource stress:

```bash
bash scripts/reproduce_dbx1000_vita_stress.sh
```

This reports speedup against the best safe DBx1000 baseline, not only OCC+K. The current stress result is the only DBx1000/Vita setting in this artifact with a 50%+ throughput improvement; see `docs/DBX1000_RESEARCH_STORY.md` for the exact scope and caveats.

Full DBx1000-backed `K x contention` matrix:

```bash
bash scripts/reproduce_dbx1000_vita_matrix.sh
```

This runs `K=1/4/8` across low, medium, and high contention and writes `agent/experiments/results/dbx1000_vita_matrix_summary.csv`. See `docs/DBX1000_MATRIX_EXPERIMENTS.md` for parameters, CC policy definitions, and the current results.

Real DeepSeek `K x contention` matrix:

```bash
DEEPSEEK_API_KEY=... bash scripts/reproduce_llm_matrix.sh
```

This runs real DeepSeek calls for `K=1/4/8` across low, medium, and high contention. It distinguishes the traditional branch-per-transaction baseline from the stronger agent-aware OCC ablation. See `docs/REAL_LLM_MATRIX_AND_BASELINES.md`.

Large-scale rigorous benchmark:

```bash
bash scripts/reproduce_rigorous.sh large
```

This runs 30k tasks per seed, 5 seeds, up to 64 worker threads, and regenerates `fig11_rigorous_vitabench.svg`.

## Main Evidence

- CAST replaces expensive regeneration with semantic merge or candidate reuse.
- HYBRID reduces false conflicts compared with syntactic OCC/MVCC-style checks.
- Escrow preserves lower-bound constraints while keeping commutative concurrency.
- The expanded CCFA experiments add OCC/Silo/TicToc/MVCC/2PL baselines, agent-aware OCC+K/HYBRID-K1 controls, larger workload sweeps, and a hotspot mixed-object workload where HYBRID reduces generation calls per task close to 1.0.
- The VitaBench-derived benchmark collects real OTA shared resources and confirms official order-tool quantity decrements; HYBRID improves throughput while keeping zero oversell.
- The rigorous benchmark reports throughput, P95/P99 latency, commit rate, and SLA success rate; at 64 threads HYBRID improves throughput by 54.7% over OCC+K and raises SLA success from 25.5% to 83.4%.
- In real DeepSeek OTA traces, calls are seconds-scale and every task produced multiple alternatives in the recorded run, validating the reselect premise; the current 60-task run has mean c_gen 1.32s and HYBRID replay throughput 8.2% above OCC+K with zero oversell.

The current artifact is close to a CCFA paper prototype: the remaining work is mostly paper writing, final claim selection, and presentation polish rather than missing core mechanisms.

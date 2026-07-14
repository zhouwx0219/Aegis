#!/usr/bin/env python3
"""Train paper ATCC with coordinated phase-path rollouts and a shared Delta-Psys reward."""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cc.atcc.ppo import DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig, audit_policy
from agent.runtime import (
    CompiledPhasePolicy,
    CompiledPolicyEntry,
    PaperRewardConfig,
    PhaseAwareState,
    apply_paper_rewards,
    phase_aware_state_from_dict,
    policy_transition_from_dict,
)
from scripts.train_paper_atcc_matrix import DEFAULT_VARIANTS, coverage_report, transition_dict
from scripts.unified_trace.generate_castdas_trace import VARIANTS


DEFAULT_PATHS = (
    tuple((0, action) for action in range(16))
    + tuple((action, action) for action in range(1, 16))
    + (
        (1, 5),
        (1, 9),
        (2, 6),
        (2, 10),
        (4, 5),
        (4, 12),
        (8, 9),
        (8, 12),
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--clients", default="24,40")
    parser.add_argument("--agent-ratios", default="1.0,0.8")
    parser.add_argument("--seeds", default="810104,810105,810106")
    parser.add_argument("--paths", default=",".join(f"{a}:{b}" for a, b in DEFAULT_PATHS))
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--transactions-per-worker", type=int, default=128)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--min-group-samples", type=int, default=16)
    parser.add_argument("--min-group-actions", type=int, default=2)
    parser.add_argument("--ppo-seed", type=int, default=810100)
    parser.add_argument("--shared-reward-weight", type=float, default=100.0)
    parser.add_argument("--refinement-distance-threshold", type=float)
    parser.add_argument("--disable-occ-cold-start-guard", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    variants = split(args.variants)
    if set(variants) - set(VARIANTS):
        raise SystemExit("unknown workload variant")
    clients = [int(value) for value in split(args.clients)]
    ratios = [float(value) for value in split(args.agent_ratios)]
    seeds = [int(value) for value in split(args.seeds)]
    paths = [tuple(int(part) for part in item.split(":")) for item in split(args.paths)]
    validate_paths(paths)

    output = args.output_dir.resolve()
    trace_dir, run_dir, trajectory_dir, policy_dir = (
        output / "traces", output / "runs", output / "trajectories", output / "behavior_policies"
    )
    for directory in (output, trace_dir, run_dir, trajectory_dir, policy_dir):
        directory.mkdir(parents=True, exist_ok=True)
    policies = write_behavior_policies(policy_dir, paths)
    action_probabilities = behavior_probabilities(paths)

    transitions: list[PolicyTransition] = []
    source_runs = []
    run_index = 0
    for variant in variants:
        for client_count in clients:
            for ratio in ratios:
                for seed in seeds:
                    config_id = f"{variant}_c{client_count}_a{ratio:g}_s{seed}"
                    trace = trace_dir / f"{config_id}.csv"
                    warmup_trace = trace_dir / f"{config_id}.warmup.csv"
                    generate_trace(trace, config_id, variant, client_count, ratio, seed, args.transactions_per_worker)
                    generate_trace(
                        warmup_trace,
                        f"{config_id}-warmup",
                        variant,
                        client_count,
                        ratio,
                        seed + 500_000,
                        args.transactions_per_worker,
                    )
                    coordinated_runs = []
                    for refine_action, commit_action in paths:
                        path_id = f"p{refine_action}_{commit_action}"
                        result = run_dir / f"{config_id}_{path_id}.csv"
                        trajectory = trajectory_dir / f"{config_id}_{path_id}.json"
                        if not (args.resume and result.exists() and trajectory.exists()):
                            run_trace(
                                trace,
                                warmup_trace,
                                result,
                                trajectory,
                                policies[(refine_action, commit_action)],
                                args.duration,
                                args.warmup_seconds,
                            )
                        result_row = next(csv.DictReader(result.open(newline="", encoding="utf-8-sig")))
                        payload = json.loads(trajectory.read_text(encoding="utf-8"))
                        coordinated_runs.append(
                            {
                                "path": (refine_action, commit_action),
                                "path_id": path_id,
                                "result": result_row,
                                "rows": payload.get("transitions", []),
                            }
                        )
                        run_index += 1
                        print(f"[{run_index}] {config_id}:{path_id}", flush=True)
                    shared_scores = normalized_system_scores(coordinated_runs, mixed=ratio < 0.999)
                    for coordinated, score in zip(coordinated_runs, shared_scores):
                        refine_action, commit_action = coordinated["path"]
                        path_id = coordinated["path_id"]
                        for row in coordinated["rows"]:
                            state = phase_aware_state_from_dict(row["state"])
                            probability = behavior_probability(
                                action_probabilities,
                                state.phase,
                                int(state.current_action),
                                int(row["action"]),
                            )
                            transition = policy_transition_from_dict(
                                row, source_id=config_id
                            )
                            transitions.append(
                                dataclasses.replace(
                                    transition,
                                    txn_id=f"{path_id}:{row['txn_id']}",
                                    behavior_probability=probability,
                                    system_delta=(
                                        max(-3.0, min(3.0, float(score)))
                                        if transition.done
                                        else transition.system_delta
                                    ),
                                )
                            )
                        source_runs.append(
                            {
                                "config_id": config_id,
                                "path": [refine_action, commit_action],
                                "shared_system_score": score,
                                "result": coordinated["result"],
                            }
                        )

    transitions, reward_report = apply_paper_rewards(
        transitions,
        PaperRewardConfig(system_weight=float(args.shared_reward_weight)),
    )
    policy = DiscretePPOPolicy(seed=args.ppo_seed)
    config = PPOConfig(
        epochs=args.epochs,
        entropy_weight=args.entropy_weight,
        min_group_samples=args.min_group_samples,
        min_group_actions=args.min_group_actions,
    )
    training = DiscretePPOTrainer(config).train(policy, transitions)
    compiled = policy.compile(
        generation=args.generation,
        refinement_distance_threshold=args.refinement_distance_threshold,
        occ_cold_start_guard=not args.disable_occ_cold_start_guard,
    )
    policy_audit = audit_policy(policy, compiled, transitions, discount=config.discount)
    (output / "paper_policy.json").write_text(
        json.dumps(compiled.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "trajectory.json").write_text(
        json.dumps(
            {
                "artifact_type": "cast-das-paper-atcc-coordinated-trajectories",
                "source_runs": source_runs,
                "transitions": [transition_dict(row) for row in transitions],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    report = {
        "artifact_type": "cast-das-paper-atcc-coordinated-train-report",
        "algorithm": "coordinated-discrete-clipped-ppo",
        "generation": args.generation,
        "training_seeds": seeds,
        "evaluation_seeds_excluded": [920104, 920105, 920106],
        "variants": variants,
        "clients": clients,
        "agent_ratios": ratios,
        "behavior_paths": [list(path) for path in paths],
        "rollout_duration_seconds": args.duration,
        "warmup_seconds": args.warmup_seconds,
        "transactions_per_worker": args.transactions_per_worker,
        "shared_reward_weight": args.shared_reward_weight,
        "shared_reward_applied": True,
        "shared_reward_semantics": "terminal_delta_psys_eta_weight",
        "reward": reward_report,
        "policy_uses_workload_labels": False,
        "outcome_oracle": False,
        "action_space": "lock_protection_mask_4bit",
        "priority_control": "transaction_manager_formula",
        "priority_is_policy_action": False,
        "config": dataclasses.asdict(config),
        "training": training,
        "policy_audit": policy_audit,
        "coverage": coverage_report(transitions),
        "compiled_entries": len(compiled.entries),
        "selective_refinement": bool(compiled.refinement_actor),
        "refinement_distance_threshold": args.refinement_distance_threshold,
        "occ_cold_start_guard": compiled.occ_cold_start_guard,
    }
    (output / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "paper_policy.json")
    return 0


def normalized_system_scores(runs, *, mixed: bool) -> list[float]:
    metrics = []
    for run in runs:
        row = run["result"]
        metrics.append(
            {
                "agent": number(row, "agent_task_tps"),
                "total": number(row, "total_tps"),
                "abort_good": 1.0 - number(row, "agent_attempt_abort_rate"),
                "latency_good": -number(row, "agent_p99_latency_ms"),
                "background": number(row, "background_commit_rate") if mixed else 0.0,
            }
        )
    normalized = {
        key: minmax([row[key] for row in metrics]) for key in metrics[0]
    }
    weights = (
        {"agent": 0.25, "total": 0.30, "abort_good": 0.10, "latency_good": 0.10, "background": 0.25}
        if mixed
        else {"agent": 0.40, "total": 0.25, "abort_good": 0.20, "latency_good": 0.15, "background": 0.0}
    )
    raw = [sum(weights[key] * normalized[key][index] for key in weights) for index in range(len(runs))]
    center = statistics.fmean(raw)
    return [value - center for value in raw]


def minmax(values):
    low, high = min(values), max(values)
    if high <= low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def behavior_probabilities(paths):
    refine_counts = collections.Counter(path[0] for path in paths)
    commit_counts = collections.Counter(paths)
    probabilities = {}
    for refine_action, commit_action in paths:
        probabilities[("explore", 0, 0)] = 1.0
        probabilities[("refine", 0, refine_action)] = refine_counts[refine_action] / len(paths)
        probabilities[("commit", refine_action, commit_action)] = (
            commit_counts[(refine_action, commit_action)] / refine_counts[refine_action]
        )
    return probabilities


def behavior_probability(probabilities, phase, current_action, action):
    key = (str(phase), int(current_action), int(action))
    if key in probabilities:
        return probabilities[key]
    if int(action) == int(current_action):
        # Once a coordinated path has selected a phase action, operation-level
        # decisions deterministically retain it until the next phase boundary.
        return 1.0
    raise KeyError(key)


def write_behavior_policies(directory, paths):
    result = {}
    for index, (refine_action, commit_action) in enumerate(paths):
        policy = CompiledPhasePolicy(
            (
                CompiledPolicyEntry(phase="explore", current_action=0, action=0),
                CompiledPolicyEntry(phase="refine", current_action=0, action=refine_action),
                CompiledPolicyEntry(phase="commit", current_action=refine_action, action=commit_action),
            ),
            generation=1000 + index,
        )
        path = directory / f"path_{refine_action}_{commit_action}.json"
        path.write_text(json.dumps(policy.to_dict(), indent=2) + "\n", encoding="utf-8")
        result[(refine_action, commit_action)] = path
    return result


def generate_trace(path, trace_id, variant, clients, ratio, seed, transactions_per_worker):
    run_checked(
        [sys.executable, str(ROOT / "scripts/unified_trace/generate_castdas_trace.py"), "--output", str(path),
         "--trace-id", trace_id, "--variant", variant, "--clients", str(clients), "--agent-ratio", str(ratio),
         "--seed", str(seed), "--transactions-per-worker", str(transactions_per_worker),
         "--reasoning-profile", "agentic", "--reasoning-scale", "2.0"]
    )


def run_trace(trace, warmup_trace, result, trajectory, policy, duration, warmup_seconds):
    run_checked(
        [sys.executable, str(ROOT / "scripts/unified_trace/run_castdas_trace_fair.py"), "--trace", str(trace),
         "--warmup-trace", str(warmup_trace),
         "--output", str(result), "--cc", "paper-atcc", "--paper-policy", str(policy),
         "--trajectory-output", str(trajectory), "--policy-mode", "eval", "--measure-seconds", str(duration),
         "--warmup-seconds", str(warmup_seconds),
         "--max-attempts", "5"]
    )


def validate_paths(paths):
    for refine_action, commit_action in paths:
        if not 0 <= refine_action <= 15 or not 0 <= commit_action <= 15:
            raise SystemExit("actions must be in [0, 15]")
        if (commit_action | refine_action) != commit_action:
            raise SystemExit("commit action must monotonically include refine action")


def number(row, key):
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_checked(command):
    subprocess.run(command, cwd=ROOT, check=True)


def split(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

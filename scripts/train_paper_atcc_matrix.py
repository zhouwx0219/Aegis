#!/usr/bin/env python3
"""Collect exploratory paper-ATCC trajectories and compile an offline PPO policy."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cc.atcc.ppo import (
    DiscretePPOPolicy,
    DiscretePPOTrainer,
    PPOConfig,
    audit_policy,
    state_key,
)
from agent.runtime import PolicyTransition, apply_paper_rewards, policy_transition_from_dict
from scripts.unified_trace.generate_castdas_trace import VARIANTS


DEFAULT_VARIANTS = (
    "tpcc_low_w100",
    "tpcc_high_w1",
    "ycsb_low",
    "ycsb_medium_z07",
    "ycsb_medium_z08",
    "ycsb_high_z099",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--clients", default="24,40")
    parser.add_argument("--agent-ratios", default="1.0,0.8")
    parser.add_argument("--seeds", default="810104,810105,810106")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--paper-deferred-replay",
        action="store_true",
        help=(
            "Opt into the whole-transaction replay ablation. The paper main "
            "path executes reasoning and accesses in their original order."
        ),
    )
    parser.add_argument("--reasoning-scale", type=float, default=1.0)
    parser.add_argument("--transactions-per-worker", type=int, default=128)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--ppo-seed", type=int, default=810100)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--min-group-samples", type=int, default=16)
    parser.add_argument("--min-group-actions", type=int, default=2)
    parser.add_argument("--exploration-stay-probability", type=float, default=0.0)
    parser.add_argument("--initial-policy", type=Path)
    parser.add_argument("--exploration-epsilon", type=float, default=0.2)
    parser.add_argument("--refinement-distance-threshold", type=float)
    parser.add_argument("--disable-occ-cold-start-guard", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= float(args.exploration_stay_probability) <= 1.0:
        raise SystemExit("--exploration-stay-probability must be in [0, 1]")
    if not 0.0 <= float(args.exploration_epsilon) <= 1.0:
        raise SystemExit("--exploration-epsilon must be in [0, 1]")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")

    variants = split_csv(args.variants)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants: {','.join(unknown)}")
    clients = [int(value) for value in split_csv(args.clients)]
    ratios = [float(value) for value in split_csv(args.agent_ratios)]
    seeds = [int(value) for value in split_csv(args.seeds)]
    output_dir = args.output_dir.resolve()
    trace_dir = output_dir / "traces"
    run_dir = output_dir / "runs"
    trajectory_dir = output_dir / "trajectories"
    for directory in (output_dir, trace_dir, run_dir, trajectory_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    source_runs = []
    transitions = []
    run_index = 0
    for variant in variants:
        for client_count in clients:
            for ratio in ratios:
                for seed in seeds:
                    run_id = f"{variant}_c{client_count}_a{str(ratio).replace('.', 'p')}_s{seed}"
                    trace = trace_dir / f"{run_id}.csv"
                    warmup_trace = trace_dir / f"{run_id}.warmup.csv"
                    result = run_dir / f"{run_id}.csv"
                    trajectory = trajectory_dir / f"{run_id}.json"
                    exploration_seed = seed * 1009 + client_count * 17 + int(ratio * 10) + run_index
                    if not (args.resume and trace.exists()):
                        run_checked(
                            [
                                sys.executable,
                                str(ROOT / "scripts/unified_trace/generate_castdas_trace.py"),
                                "--output", str(trace),
                                "--trace-id", run_id,
                                "--variant", variant,
                                "--clients", str(client_count),
                                "--agent-ratio", str(ratio),
                                "--seed", str(seed),
                                "--transactions-per-worker", str(args.transactions_per_worker),
                                "--reasoning-profile", "agentic",
                                "--reasoning-scale", str(args.reasoning_scale),
                            ]
                        )
                    if not (args.resume and warmup_trace.exists()):
                        run_checked(
                            [
                                sys.executable,
                                str(ROOT / "scripts/unified_trace/generate_castdas_trace.py"),
                                "--output", str(warmup_trace),
                                "--trace-id", f"{run_id}-warmup",
                                "--variant", variant,
                                "--clients", str(client_count),
                                "--agent-ratio", str(ratio),
                                "--seed", str(seed + 500_000),
                                "--transactions-per-worker", str(args.transactions_per_worker),
                                "--reasoning-profile", "agentic",
                                "--reasoning-scale", str(args.reasoning_scale),
                            ]
                        )
                    if not (args.resume and result.exists() and trajectory.exists()):
                        command = [
                            sys.executable,
                            str(ROOT / "scripts/unified_trace/run_castdas_trace_fair.py"),
                            "--trace", str(trace),
                            "--warmup-trace", str(warmup_trace),
                            "--output", str(result),
                            "--cc", "paper-atcc",
                            "--trajectory-output", str(trajectory),
                            "--paper-exploration-seed", str(exploration_seed),
                            "--paper-exploration-stay-probability",
                            str(args.exploration_stay_probability),
                            "--measure-seconds", str(args.duration),
                            "--warmup-seconds", str(args.warmup_seconds),
                            "--max-attempts", str(args.max_attempts),
                            "--disable-atcc-retry-cache",
                        ]
                        if args.initial_policy is not None:
                            command.extend(
                                [
                                    "--paper-policy",
                                    str(args.initial_policy.resolve()),
                                    "--paper-exploration-epsilon",
                                    str(args.exploration_epsilon),
                                ]
                            )
                        if not args.paper_deferred_replay:
                            command.append("--disable-paper-deferred-replay")
                        run_checked(command)
                    if not trajectory.exists():
                        detail = result.read_text(encoding="utf-8") if result.exists() else "missing result"
                        raise RuntimeError(f"trajectory runner failed for {run_id}: {detail}")
                    payload = json.loads(trajectory.read_text(encoding="utf-8"))
                    rows = payload.get("transitions", [])
                    transitions.extend(parse_transition(row, source_id=run_id) for row in rows)
                    source_runs.append(
                        {
                            "run_id": run_id,
                            "variant": variant,
                            "clients": client_count,
                            "agent_ratio": ratio,
                            "training_seed": seed,
                            "exploration_seed": exploration_seed,
                            "transition_count": len(rows),
                            "trace": str(trace),
                            "warmup_trace": str(warmup_trace),
                            "result": str(result),
                        }
                    )
                    run_index += 1
                    print(f"[{run_index}] {run_id}: transitions={len(rows)}", flush=True)

    transitions, reward_report = apply_paper_rewards(transitions)
    policy = DiscretePPOPolicy(seed=args.ppo_seed)
    config = PPOConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
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
    coverage = coverage_report(transitions)

    policy_path = output_dir / "paper_policy.json"
    trajectory_path = output_dir / "trajectory.json"
    report_path = output_dir / "train_report.json"
    coverage_path = output_dir / "coverage.json"
    policy_path.write_text(json.dumps(compiled.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trajectory_path.write_text(
        json.dumps(
            {
                "artifact_type": "cast-das-paper-atcc-exploration-trajectories",
                "source_runs": source_runs,
                "transitions": [transition_dict(row) for row in transitions],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "artifact_type": "cast-das-paper-atcc-train-report",
        "algorithm": "discrete-clipped-ppo",
        "generation": args.generation,
        "ppo_seed": args.ppo_seed,
        "training_seeds": seeds,
        "evaluation_seeds_excluded": [920104, 920105, 920106],
        "variants": variants,
        "clients": clients,
        "agent_ratios": ratios,
        "duration_s_per_run": args.duration,
        "warmup_s_per_run": args.warmup_seconds,
        "max_attempts": args.max_attempts,
        "paper_deferred_replay": bool(args.paper_deferred_replay),
        "access_timing": (
            "whole_transaction_replay_ablation"
            if args.paper_deferred_replay
            else "real_interleaved_operations"
        ),
        "reasoning_scale": args.reasoning_scale,
        "runs": len(source_runs),
        "elapsed_s": time.time() - started,
        "config": dataclasses.asdict(config),
        "training": training,
        "policy_audit": policy_audit,
        "coverage": coverage,
        "compiled_entries": len(compiled.entries),
        "selective_refinement": bool(compiled.refinement_actor),
        "refinement_distance_threshold": args.refinement_distance_threshold,
        "occ_cold_start_guard": compiled.occ_cold_start_guard,
        "atcc_retry_cache_enabled": False,
        "policy_uses_workload_labels": False,
        "outcome_oracle": False,
        "action_space": "lock_protection_mask_4bit",
        "priority_control": "transaction_manager_formula",
        "priority_is_policy_action": False,
        "exploration_stay_probability": args.exploration_stay_probability,
        "initial_policy": str(args.initial_policy.resolve()) if args.initial_policy else "",
        "exploration_epsilon": args.exploration_epsilon,
        "reward": reward_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(policy_path)
    return 0


def coverage_report(transitions):
    states = collections.Counter(state_key(row.state) for row in transitions)
    state_actions = collections.Counter((state_key(row.state), row.action) for row in transitions)
    actions = collections.Counter(row.action for row in transitions)
    phases = collections.defaultdict(collections.Counter)
    probabilities = []
    trajectories = collections.defaultdict(list)
    for row in transitions:
        phases[row.state.phase][row.action] += 1
        probabilities.append(row.behavior_probability)
        trajectories[row.txn_id].append(row.action)
    escalation = collections.Counter(
        "->".join(str(action) for action in actions_for_txn)
        for actions_for_txn in trajectories.values()
        if len(actions_for_txn) > 1
    )
    return {
        "transition_count": len(transitions),
        "transaction_count": len(trajectories),
        "unique_states": len(states),
        "unique_state_action_pairs": len(state_actions),
        "actions_observed": sorted(actions),
        "action_counts": {str(key): value for key, value in sorted(actions.items())},
        "phase_action_counts": {
            phase: {str(key): value for key, value in sorted(counts.items())}
            for phase, counts in sorted(phases.items())
        },
        "behavior_probability": {
            "min": min(probabilities) if probabilities else 0.0,
            "max": max(probabilities) if probabilities else 0.0,
            "mean": statistics.fmean(probabilities) if probabilities else 0.0,
            "all_valid": all(0.0 < value <= 1.0 for value in probabilities),
        },
        "common_action_paths": dict(escalation.most_common(20)),
    }


def parse_transition(row, *, source_id=""):
    return policy_transition_from_dict(row, source_id=str(source_id))


def transition_dict(row):
    payload = dataclasses.asdict(row)
    payload["state"] = dataclasses.asdict(row.state)
    payload["next_state"] = dataclasses.asdict(row.next_state)
    return payload


def run_checked(command):
    subprocess.run(command, cwd=ROOT, check=True)


def split_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

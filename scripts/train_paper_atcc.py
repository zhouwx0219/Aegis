#!/usr/bin/env python3
"""Train and compile the paper ATCC policy from runtime trajectories."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cc.atcc.ppo import (
    DiscretePPOPolicy,
    DiscretePPOTrainer,
    PAPER_MEDOIDS_PER_GROUP,
    PPOConfig,
    audit_policy,
    state_key,
)
from agent.runtime import PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=810104)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--group-learning-rate", type=float, default=0.03)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--min-group-samples", type=int, default=16)
    parser.add_argument("--min-group-actions", type=int, default=2)
    parser.add_argument("--system-weight", type=float, default=10.0)
    parser.add_argument("--lock-weight", type=float, default=5.0)
    parser.add_argument("--abort-weight", type=float, default=80.0)
    parser.add_argument("--coordinated-reward-weight", type=float)
    parser.add_argument("--coordinated-original-weight", type=float, default=100.0)
    parser.add_argument("--refinement-distance-threshold", type=float)
    parser.add_argument(
        "--medoids-per-group",
        type=int,
        default=PAPER_MEDOIDS_PER_GROUP,
        help="Representative state keys retained per phase/action group (paper default: 4).",
    )
    parser.add_argument("--disable-occ-cold-start-guard", action="store_true")
    args = parser.parse_args()

    if args.coordinated_reward_weight is not None:
        raise SystemExit(
            "--coordinated-reward-weight is incompatible with the paper reward formula"
        )
    if args.medoids_per_group <= 0:
        raise SystemExit("--medoids-per-group must be positive")
    transitions = []
    for path in args.trajectory:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("transitions", payload) if isinstance(payload, dict) else payload
        for row in rows:
            source_id = str(row.get("source_id") or path.stem)
            transitions.append(policy_transition_from_dict(row, source_id=source_id))
    transitions, reward_report = apply_paper_rewards(
        transitions,
        PaperRewardConfig(
            abort_weight=float(args.abort_weight),
            lock_weight=float(args.lock_weight),
            system_weight=float(args.system_weight),
        ),
    )
    policy = DiscretePPOPolicy(seed=args.seed)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        group_learning_rate=args.group_learning_rate,
        clip_ratio=args.clip_ratio,
        discount=args.discount,
        entropy_weight=args.entropy_weight,
        epochs=args.epochs,
        min_group_samples=args.min_group_samples,
        min_group_actions=args.min_group_actions,
    )
    training = DiscretePPOTrainer(config).train(policy, transitions)
    compiled = policy.compile(
        generation=args.generation,
        medoids_per_group=args.medoids_per_group,
        refinement_distance_threshold=args.refinement_distance_threshold,
        occ_cold_start_guard=not args.disable_occ_cold_start_guard,
    )
    policy_audit = audit_policy(policy, compiled, transitions, discount=config.discount)
    coverage = coverage_report(transitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compiled.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(
        json.dumps(
            {
                "artifact_type": "cast-das-paper-atcc-train-report",
                "algorithm": "discrete-clipped-ppo",
                "config": dataclasses.asdict(config),
                "seed": args.seed,
                "generation": args.generation,
                "inputs": [str(path) for path in args.trajectory],
                "training": training,
                "policy_audit": policy_audit,
                "coverage": coverage,
                "compiled_entries": len(compiled.entries),
                "medoids_per_group": compiled.medoids_per_group,
                "selective_refinement": bool(compiled.refinement_actor),
                "refinement_distance_threshold": args.refinement_distance_threshold,
                "occ_cold_start_guard": compiled.occ_cold_start_guard,
                "reward": reward_report,
                "action_space": "lock_protection_mask_4bit",
                "priority_control": "transaction_manager_formula",
                "priority_is_policy_action": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


def coordinated_shared_scores(payload) -> dict[tuple[str, str], float]:
    if not isinstance(payload, dict):
        return {}
    scores = {}
    for row in payload.get("source_runs", []):
        path = row.get("path", [])
        if len(path) != 2:
            continue
        path_id = f"p{int(path[0])}_{int(path[1])}"
        scores[(str(row.get("config_id", "")), path_id)] = float(
            row.get("shared_system_score", 0.0)
        )
    return scores


def coverage_report(transitions: list[PolicyTransition]) -> dict[str, object]:
    action_counts: dict[str, int] = {}
    phase_action_counts: dict[str, dict[str, int]] = {}
    probabilities = []
    states = set()
    state_actions = set()
    for transition in transitions:
        action = str(int(transition.action))
        phase = transition.state.phase
        action_counts[action] = action_counts.get(action, 0) + 1
        phase_counts = phase_action_counts.setdefault(phase, {})
        phase_counts[action] = phase_counts.get(action, 0) + 1
        probabilities.append(float(transition.behavior_probability))
        key = state_key(transition.state)
        states.add(key)
        state_actions.add((key, action))
    return {
        "transition_count": len(transitions),
        "unique_states": len(states),
        "unique_state_action_pairs": len(state_actions),
        "actions_observed": sorted(int(action) for action in action_counts),
        "action_counts": action_counts,
        "phase_action_counts": phase_action_counts,
        "behavior_probability": {
            "min": min(probabilities) if probabilities else 0.0,
            "max": max(probabilities) if probabilities else 0.0,
            "mean": sum(probabilities) / len(probabilities) if probabilities else 0.0,
            "all_valid": all(0.0 < value <= 1.0 for value in probabilities),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())

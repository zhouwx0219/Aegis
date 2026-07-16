#!/usr/bin/env python3
"""Train one ATCC policy by replaying the same fixed-trace runtime used for evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cli.train_atcc import make_policy, resolve_trainable_actions  # noqa: E402
from agent.cc.atcc.policy import ATCCPolicyTable  # noqa: E402
from scripts.unified_trace.run_unified_trace_matrix import (  # noqa: E402
    TRAINING_SEEDS,
    VARIANTS,
    split_csv,
    trace_transactions_per_worker,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variants", default="tpcc_high_w1")
    parser.add_argument("--clients", default="24,40")
    parser.add_argument("--agent-ratios", default="1.0,0.8")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in TRAINING_SEEDS))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--reasoning-scale", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    variants = split_csv(args.variants)
    clients = [int(value) for value in split_csv(args.clients)]
    ratios = [float(value) for value in split_csv(args.agent_ratios)]
    seeds = [int(value) for value in split_csv(args.seeds)]
    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unsupported variant: {variant}")

    output_dir = args.output_dir.resolve()
    traces_dir = output_dir / "traces"
    runs_dir = output_dir / "runs"
    logs_dir = output_dir / "logs"
    for directory in (output_dir, traces_dir, runs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    policy = make_policy(
        abort_threshold=0.20,
        min_visits=5,
        protect_cost_threshold_ms=10.0,
        low_conflict_occ_guard=True,
        low_conflict_safe_abort_rate=0.50,
        sparse_state_risk_prior=True,
        commit_value=100.0,
        abort_penalty=80.0,
        reasoning_weight=1.0,
        lock_wait_weight=0.5,
        latency_weight=0.1,
        lock_hold_weight=0.05,
        background_abort_weight=15.0,
        background_tps_loss_weight=5.0,
        trainable_actions=resolve_trainable_actions("mixed", "auto"),
        exploration_coefficient=1.5,
    ).set_mode("train")
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    policy.save_json(args.policy)

    report_rows: list[dict[str, object]] = []
    run_index = 0
    for variant in variants:
        tpw = trace_transactions_per_worker(
            variant,
            seconds=float(args.duration),
            fallback=4,
        )
        for ratio in ratios:
            for client_count in clients:
                for episode in range(int(args.episodes)):
                    seed = seeds[episode % len(seeds)]
                    trace_id = f"train_{variant}_c{client_count}_a{ratio:g}_e{episode}_s{seed}"
                    trace_path = traces_dir / f"{trace_id}.csv"
                    warmup_trace_path = traces_dir / f"{trace_id}.warmup.csv"
                    result_path = runs_dir / f"{trace_id}.csv"
                    generate_cmd = [
                        sys.executable,
                        str(THIS_DIR / "generate_castdas_trace.py"),
                        "--output", str(trace_path),
                        "--trace-id", trace_id,
                        "--variant", variant,
                        "--clients", str(client_count),
                        "--agent-ratio", str(ratio),
                        "--seed", str(seed),
                        "--repeat", str(episode),
                        "--transactions-per-worker", str(tpw),
                        "--reasoning-profile", "agentic",
                        "--reasoning-scale", str(args.reasoning_scale),
                    ]
                    run_cmd = [
                        sys.executable,
                        str(THIS_DIR / "run_castdas_trace_fair.py"),
                        "--trace", str(trace_path),
                        "--warmup-trace", str(warmup_trace_path),
                        "--output", str(result_path),
                        "--cc", "dynamic-atcc",
                        "--policy", str(args.policy),
                        "--policy-mode", "train",
                        "--max-attempts", str(args.max_attempts),
                        "--measure-seconds", str(args.duration),
                        "--warmup-seconds", str(args.warmup_seconds),
                    ]
                    warmup_cmd = list(generate_cmd)
                    warmup_cmd[warmup_cmd.index(str(trace_path))] = str(warmup_trace_path)
                    warmup_cmd[warmup_cmd.index(trace_id)] = f"{trace_id}.warmup"
                    warmup_seed = str(int(seed) + 1_000_000)
                    warmup_cmd[warmup_cmd.index(str(seed))] = warmup_seed
                    subprocess.run(generate_cmd, check=True, stdout=subprocess.DEVNULL)
                    subprocess.run(warmup_cmd, check=True, stdout=subprocess.DEVNULL)
                    subprocess.run(run_cmd, check=True, stdout=subprocess.DEVNULL)
                    with result_path.open(newline="", encoding="utf-8-sig") as handle:
                        row = next(csv.DictReader(handle))
                    report_rows.append(
                        {
                            "run_index": run_index,
                            "variant": variant,
                            "clients": client_count,
                            "agent_ratio": ratio,
                            "seed": seed,
                            "episode": episode,
                            "status": row.get("status", ""),
                            "agent_task_tps": row.get("agent_task_tps", ""),
                            "total_tps": row.get("total_tps", ""),
                            "background_tps": row.get("background_tps", ""),
                            "agent_attempt_abort_rate": row.get("agent_attempt_abort_rate", ""),
                            "raw_action_counts": row.get("raw_action_counts", ""),
                        }
                    )
                    run_index += 1

    trained = ATCCPolicyTable.load_json(args.policy).set_mode("eval")
    trained.save_json(args.policy)
    report = {
        "training_scope": "unified-fixed-trace",
        "runs": run_index,
        "variants": variants,
        "clients": clients,
        "agent_ratios": ratios,
        "seeds": seeds,
        "episodes": int(args.episodes),
        "duration_s": float(args.duration),
        "warmup_seconds": float(args.warmup_seconds),
        "policy_states": len(trained.rows),
        "reward_config": trained.reward_config.to_dict(),
        "trainable_actions": list(trained.trainable_actions),
        "rows": report_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.policy)
    print(args.report)
    print(f"runs={run_index} policy_states={len(trained.rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate the paper-ATCC short benefit gate and emit paper-ready comparison rows."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.raw_csv.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    grouped = collections.defaultdict(list)
    for row in raw_rows:
        grouped[config_key(row)].append(row)

    comparisons = []
    for key, rows in sorted(grouped.items()):
        atcc_rows = [row for row in rows if row.get("cc_label") == "ATCC" or row.get("cc") == "paper-atcc"]
        baselines = [row for row in rows if row not in atcc_rows]
        if len(atcc_rows) != 1 or not baselines:
            continue
        atcc = atcc_rows[0]
        best_agent = max(baselines, key=lambda row: number(row, "agent_task_tps", "agent_tps"))
        best_total = max(baselines, key=lambda row: number(row, "total_tps"))
        best_background = max(baselines, key=lambda row: number(row, "background_tps"))
        low = str(atcc.get("level", "")) == "low"
        high = str(atcc.get("level", "")) == "high"
        mixed = abs(number(atcc, "agent_ratio") - 0.8) < 1e-9
        total_ratio = ratio(number(atcc, "total_tps"), number(best_total, "total_tps"))
        agent_ratio = ratio(
            number(atcc, "agent_task_tps", "agent_tps"),
            number(best_agent, "agent_task_tps", "agent_tps"),
        )
        background_ratio = ratio(
            number(atcc, "background_tps"), number(best_background, "background_tps")
        )
        p99 = number(atcc, "agent_p99_latency_ms")
        baseline_p99 = number(best_agent, "agent_p99_latency_ms")
        comparisons.append(
            {
                "workload_variant": atcc.get("workload_variant", ""),
                "level": atcc.get("level", ""),
                "clients": atcc.get("clients", ""),
                "client_mix": atcc.get("client_mix", ""),
                "agent_ratio": atcc.get("agent_ratio", ""),
                "atcc_agent_task_tps": number(atcc, "agent_task_tps", "agent_tps"),
                "best_agent_baseline": label(best_agent),
                "best_baseline_agent_task_tps": number(best_agent, "agent_task_tps", "agent_tps"),
                "agent_tps_speedup": agent_ratio,
                "atcc_total_tps": number(atcc, "total_tps"),
                "best_total_baseline": label(best_total),
                "best_baseline_total_tps": number(best_total, "total_tps"),
                "total_tps_ratio": total_ratio,
                "atcc_background_tps": number(atcc, "background_tps"),
                "best_background_baseline": label(best_background),
                "best_baseline_background_tps": number(best_background, "background_tps"),
                "background_tps_ratio": background_ratio,
                "atcc_background_commit_rate": number(atcc, "background_commit_rate"),
                "atcc_p99_latency_ms": p99,
                "best_agent_baseline_p99_latency_ms": baseline_p99,
                "p99_ratio": ratio(p99, baseline_p99),
                "low_overhead_pass": not low or total_ratio >= 0.90,
                "high_agent_tps_pass": not high or agent_ratio > 1.0,
                "mixed_background_pass": not mixed or (
                    number(atcc, "background_commit_rate") >= 0.80 and background_ratio >= 0.50
                ),
                "p99_pass": not high or p99 < baseline_p99,
            }
        )

    action_report = analyze_actions(args.trajectory_dir)
    gates = {
        "low_contention_overhead": all(row["low_overhead_pass"] for row in comparisons),
        "high_contention_agent_tps": all(row["high_agent_tps_pass"] for row in comparisons),
        "mixed_background_fairness": all(row["mixed_background_pass"] for row in comparisons),
        "high_contention_p99": all(row["p99_pass"] for row in comparisons),
        "dynamic_actions": len(action_report["actions_observed"]) > 1,
        "zero_partial_larger_transition": action_report["zero_partial_larger_count"] > 0,
    }
    report = {
        "artifact_type": "cast-das-paper-atcc-benefit-gate",
        "configurations": len(comparisons),
        "gates": gates,
        "all_pass": all(gates.values()),
        "actions": action_report,
        "failed_configurations": [
            row for row in comparisons
            if not all(
                row[key] for key in (
                    "low_overhead_pass", "high_agent_tps_pass", "mixed_background_pass", "p99_pass"
                )
            )
        ],
    }
    write_csv(args.output_csv, comparisons)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["gates"], sort_keys=True))
    return 0 if report["all_pass"] else 2


def analyze_actions(directory):
    action_counts = collections.Counter()
    phase_counts = collections.defaultdict(collections.Counter)
    paths = collections.Counter()
    qualifying = 0
    transition_count = 0
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_txn = collections.defaultdict(list)
        for row in payload.get("transitions", []):
            action = int(row["action"])
            phase = str(row["state"]["phase"])
            action_counts[action] += 1
            phase_counts[phase][action] += 1
            by_txn[str(row["txn_id"])].append(action)
            transition_count += 1
        for actions in by_txn.values():
            paths["->".join(str(action) for action in actions)] += 1
            if has_zero_partial_larger(actions):
                qualifying += 1
    return {
        "trajectory_files": len(list(directory.glob("*.json"))),
        "transition_count": transition_count,
        "actions_observed": sorted(action_counts),
        "action_counts": {str(key): value for key, value in sorted(action_counts.items())},
        "phase_action_counts": {
            phase: {str(key): value for key, value in sorted(counts.items())}
            for phase, counts in sorted(phase_counts.items())
        },
        "common_paths": dict(paths.most_common(20)),
        "zero_partial_larger_count": qualifying,
    }


def has_zero_partial_larger(actions):
    for first in range(len(actions) - 2):
        if actions[first] != 0:
            continue
        for second in range(first + 1, len(actions) - 1):
            partial = actions[second]
            if not 0 < partial < 15:
                continue
            for third in range(second + 1, len(actions)):
                larger = actions[third]
                if larger != partial and (larger | partial) == larger:
                    return True
    return False


def config_key(row):
    return tuple(row.get(key, "") for key in ("workload_variant", "clients", "agent_ratio", "seed"))


def number(row, *keys):
    for key in keys:
        try:
            value = row.get(key, "")
            if value not in ("", None):
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def ratio(left, right):
    return left / right if right else 0.0


def label(row):
    return str(row.get("cc_label") or row.get("cc") or "")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

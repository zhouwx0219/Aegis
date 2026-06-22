"""Aggregate DBx1000 ASTRA/Vita CSV outputs into paper-friendly summaries."""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


METRICS = (
    "throughput",
    "task_throughput",
    "latency_ms",
    "booked",
    "no_stock",
    "oversell",
    "regen",
    "reselect",
    "merge",
    "generation_calls_per_task",
)

UNSAFE_POLICIES = {"DBX-merge-all"}
ASTRA_POLICIES = {"ASTRA-HYBRID", "ASTRA-HYBRID-K1"}


def mean_ci(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    mean = statistics.mean(vals)
    half = t95.get(len(vals) - 1, 1.96) * statistics.stdev(vals) / math.sqrt(len(vals))
    return mean, half


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(path: str, scenario: str) -> List[Dict[str, object]]:
    rows = read_rows(path)
    by_policy: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)

    occ_k = mean_ci(float(r["throughput"]) for r in by_policy.get("DBX-OCC+K", []))[0]
    occ_k1 = mean_ci(float(r["throughput"]) for r in by_policy.get("DBX-OCC-K1", []))[0]
    branch_txn = mean_ci(float(r["throughput"]) for r in by_policy.get("branch-txn", []))[0]
    policy_means = {
        policy: mean_ci(float(r["throughput"]) for r in policy_rows)[0]
        for policy, policy_rows in by_policy.items()
    }
    safe_dbx = {
        policy: throughput
        for policy, throughput in policy_means.items()
        if policy.startswith("DBX-") and policy not in UNSAFE_POLICIES
    }
    best_safe_policy = max(safe_dbx, key=safe_dbx.get) if safe_dbx else ""
    best_safe_tp = safe_dbx.get(best_safe_policy, 0.0)
    out: List[Dict[str, object]] = []
    for policy in sorted(by_policy):
        row_out: Dict[str, object] = {"scenario": scenario, "policy": policy}
        for metric in METRICS:
            mean, ci = mean_ci(float(r[metric]) for r in by_policy[policy])
            row_out[metric] = round(mean, 4)
            row_out[f"{metric}_ci"] = round(ci, 4)
        tp = float(row_out["throughput"])
        row_out["speedup_vs_dbx_occ_k"] = round(tp / occ_k, 4) if occ_k else 0.0
        row_out["speedup_vs_dbx_occ_k1"] = round(tp / occ_k1, 4) if occ_k1 else 0.0
        row_out["speedup_vs_branch_txn"] = round(tp / branch_txn, 4) if branch_txn else 0.0
        row_out["best_safe_dbx_policy"] = best_safe_policy
        row_out["best_safe_dbx_throughput"] = round(best_safe_tp, 4)
        row_out["speedup_vs_best_safe_dbx"] = round(tp / best_safe_tp, 4) if best_safe_tp else 0.0
        row_out["improvement_pct_vs_best_safe_dbx"] = (
            round((tp / best_safe_tp - 1.0) * 100.0, 2) if best_safe_tp else 0.0
        )
        out.append(row_out)
    return out


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", default="agent/experiments/results/dbx1000_vita_balanced.csv")
    parser.add_argument("--contention", default="agent/experiments/results/dbx1000_vita_contention.csv")
    parser.add_argument("--out", default="agent/experiments/results/dbx1000_vita_summary.csv")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="SCENARIO=CSV",
        help="Additional or replacement scenario input. When set, only these inputs are aggregated.",
    )
    args = parser.parse_args()

    rows = []
    if args.input:
        for item in args.input:
            if "=" not in item:
                raise SystemExit(f"Expected SCENARIO=CSV for --input, got: {item}")
            scenario, path = item.split("=", 1)
            if not scenario:
                raise SystemExit(f"Empty scenario name in --input: {item}")
            if not os.path.exists(path):
                raise SystemExit(f"Missing DBx1000 Vita result CSV: {path}")
            rows.extend(aggregate(path, scenario))
    else:
        if os.path.exists(args.balanced):
            rows.extend(aggregate(args.balanced, "balanced"))
        if os.path.exists(args.contention):
            rows.extend(aggregate(args.contention, "contention"))
    if not rows:
        raise SystemExit("No DBx1000 Vita result CSVs found.")
    write_csv(args.out, rows)

    scenarios = sorted({str(r["scenario"]) for r in rows})
    for scenario in scenarios:
        subset = [r for r in rows if r["scenario"] == scenario]
        hybrid = next((r for r in subset if r["policy"] == "ASTRA-HYBRID"), None)
        occ = next((r for r in subset if r["policy"] == "DBX-OCC+K"), None)
        best_safe = next(
            (
                r
                for r in subset
                if r["policy"] == (hybrid or {}).get("best_safe_dbx_policy")
            ),
            None,
        )
        if hybrid and occ:
            print(
                f"{scenario}: ASTRA-HYBRID tp={hybrid['throughput']} "
                f"regen={hybrid['regen']} oversell={hybrid['oversell']} "
                f"vs DBX-OCC+K tp={occ['throughput']} regen={occ['regen']}"
            )
        branch = next((r for r in subset if r["policy"] == "branch-txn"), None)
        if hybrid and branch:
            print(
                f"{scenario}: ASTRA-HYBRID speedup_vs_branch_txn="
                f"{hybrid['speedup_vs_branch_txn']} "
                f"branch-txn tp={branch['throughput']} regen={branch['regen']}"
            )
        if hybrid and best_safe:
            print(
                f"{scenario}: ASTRA-HYBRID speedup_vs_best_safe_dbx="
                f"{hybrid['speedup_vs_best_safe_dbx']} "
                f"({hybrid['improvement_pct_vs_best_safe_dbx']}%) "
                f"best={best_safe['policy']} tp={best_safe['throughput']}"
            )


if __name__ == "__main__":
    main()

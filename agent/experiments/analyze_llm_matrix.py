"""Aggregate real DeepSeek K x contention matrix results."""
from __future__ import annotations

import csv
import json
import os
import statistics
from typing import Any, Dict, List


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT_CSV = os.path.join(RESULTS, "llm_matrix_summary.csv")
OUT_MD = os.path.join(RESULTS, "llm_matrix_summary.md")

SCENARIOS = [
    ("low", 1), ("low", 4), ("low", 8),
    ("mid", 1), ("mid", 4), ("mid", 8),
    ("high", 1), ("high", 4), ("high", 8),
]

POLICIES = ["branch-txn", "OCC", "2PL", "merge-all", "HYBRID-K1", "HYBRID"]
SAFE_BASELINES = ["branch-txn", "OCC", "2PL"]


def load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def pctile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    rows: List[Dict[str, Any]] = []
    scenario_summaries: List[Dict[str, Any]] = []
    for level, k in SCENARIOS:
        scenario = f"{level}_k{k}"
        cache_path = os.path.join(RESULTS, f"llm_cache_matrix_{scenario}.json")
        replay_path = os.path.join(RESULTS, f"llm_in_the_loop_matrix_{scenario}.json")
        if not (os.path.exists(cache_path) and os.path.exists(replay_path)):
            raise SystemExit(f"missing matrix files for {scenario}")
        cache = load(cache_path)
        replay = load(replay_path)
        records = cache["records"]
        lat = [float(r.get("c_gen", 0.0)) for r in records if r.get("c_gen", 0.0) > 0]
        distinct = [int(r.get("distinct_flights", 0)) for r in records]

        safe = {
            policy: float(replay[policy]["throughput"][0])
            for policy in SAFE_BASELINES
            if policy in replay
        }
        best_policy = max(safe, key=safe.get)
        best_tp = safe[best_policy]
        hybrid_tp = float(replay["HYBRID"]["throughput"][0])
        branch_tp = float(replay["branch-txn"]["throughput"][0])

        scenario_summaries.append({
            "scenario": scenario,
            "level": level,
            "k": k,
            "tasks": len(records),
            "api_errors": cache.get("errs", 0),
            "mean_c_gen_s": round(mean(lat), 4),
            "p95_c_gen_s": round(pctile(lat, 95), 4),
            "mean_distinct": round(mean(distinct), 4),
            "best_safe_policy": best_policy,
            "best_safe_throughput": round(best_tp, 4),
            "hybrid_throughput": round(hybrid_tp, 4),
            "hybrid_speedup_vs_best_safe": round(hybrid_tp / best_tp, 4) if best_tp else 0.0,
            "hybrid_improvement_pct": round((hybrid_tp / best_tp - 1.0) * 100.0, 2) if best_tp else 0.0,
            "hybrid_speedup_vs_branch_txn": round(hybrid_tp / branch_tp, 4) if branch_tp else 0.0,
            "hybrid_oversell": replay["HYBRID"]["oversell"][0],
            "merge_all_oversell": replay["merge-all"]["oversell"][0],
        })

        for policy in POLICIES:
            r = replay[policy]
            rows.append({
                "scenario": scenario,
                "level": level,
                "k": k,
                "policy": policy,
                "throughput": round(float(r["throughput"][0]), 4),
                "throughput_ci": round(float(r["throughput"][1]), 4),
                "regen": r["regen"],
                "reselect": r["reselect"],
                "no_seat": r["no_seat"],
                "oversell": r["oversell"][0],
                "mean_c_gen_s": round(mean(lat), 4),
                "p95_c_gen_s": round(pctile(lat, 95), 4),
                "mean_distinct": round(mean(distinct), 4),
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Real DeepSeek K x Contention Matrix",
        "",
        "Each cell uses real DeepSeek `deepseek-chat` calls. Replay compares policies on the same recorded candidates and recorded generation latency.",
        "",
        "| Scenario | K | Best safe baseline | HYBRID vs best safe | HYBRID vs branch-txn | mean c_gen(s) | mean distinct | HYBRID oversell | merge-all oversell |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in scenario_summaries:
        lines.append(
            f"| {s['level']} | {s['k']} | {s['best_safe_policy']} | "
            f"{s['hybrid_speedup_vs_best_safe']:.4f} ({s['hybrid_improvement_pct']:.2f}%) | "
            f"{s['hybrid_speedup_vs_branch_txn']:.4f} | "
            f"{s['mean_c_gen_s']:.2f} | {s['mean_distinct']:.2f} | "
            f"{s['hybrid_oversell']:.0f} | {s['merge_all_oversell']:.0f} |"
        )
    lines += [
        "",
        "Baseline meaning:",
        "",
        "- `branch-txn`: traditional branch-per-transaction model: each candidate branch is an independent speculative DB transaction; the winner commits and losers abort. If the winner conflicts, the agent must regenerate rather than semantically reselect in the same transaction.",
        "- `OCC`: agent-aware OCC with the same already generated candidates; this is a stronger ablation baseline, not a native DBx1000 algorithm name.",
        "- `2PL`: lock-based baseline.",
        "- `merge-all`: unsafe upper bound; not a correctness-preserving baseline.",
        "- `HYBRID`: ASTRA intent-aware commit with constrained DELTA, merge/reselect, and lower-bound checks.",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("saved", OUT_CSV)
    print("saved", OUT_MD)
    for s in scenario_summaries:
        print(
            f"{s['scenario']}: HYBRID {s['hybrid_speedup_vs_best_safe']}x "
            f"({s['hybrid_improvement_pct']}%) vs {s['best_safe_policy']}"
        )


if __name__ == "__main__":
    main()

"""Analyze real DeepSeek transaction traces without external dependencies."""
from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(RESULTS, "llm_cache.json")
REPLAY = os.path.join(RESULTS, "llm_in_the_loop.json")
OUT_JSON = os.path.join(RESULTS, "llm_analysis.json")
OUT_MD = os.path.join(RESULTS, "llm_analysis.md")


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    cache = load_json(CACHE)
    replay = load_json(REPLAY)

    records = cache["records"]
    latencies = [float(record["c_gen"]) for record in records if record.get("c_gen", 0) > 0]
    distinct = [int(record.get("distinct_flights", 0)) for record in records]
    total_tokens = [int(record.get("usage", {}).get("total_tokens", 0) or 0) for record in records]
    prompt_tokens = [int(record.get("usage", {}).get("prompt_tokens", 0) or 0) for record in records]
    completion_tokens = [
        int(record.get("usage", {}).get("completion_tokens", 0) or 0) for record in records
    ]

    actions: Dict[str, int] = {}
    for record in records:
        action = record.get("transaction_result", {}).get("action", "missing")
        actions[action] = actions.get(action, 0) + 1

    live = cache.get("live_hybrid", {})
    hybrid_tp = float(replay["HYBRID"]["throughput"][0])
    hybrid_k1_tp = float(replay["HYBRID-K1"]["throughput"][0])
    occ_tp = float(replay["OCC"]["throughput"][0])
    two_pl_tp = float(replay["2PL"]["throughput"][0])
    merge_tp = float(replay["merge-all"]["throughput"][0])

    analysis = {
        "config": {
            "model": cache.get("model"),
            "mock": cache.get("mock"),
            "tasks": len(records),
            "errors": cache.get("errs"),
            "wall_s": cache.get("wall"),
        },
        "generation_latency_s": distribution(latencies),
        "tokens": {
            "total": sum(total_tokens),
            "mean_per_task": statistics.mean(total_tokens) if total_tokens else 0,
            "prompt_total": sum(prompt_tokens),
            "completion_total": sum(completion_tokens),
        },
        "candidates": {
            "mean_distinct": statistics.mean(distinct) if distinct else 0,
            "tasks_with_2plus": sum(value >= 2 for value in distinct),
            "tasks_with_3": sum(value >= 3 for value in distinct),
            "reselectable_fraction": (
                sum(value >= 2 for value in distinct) / len(distinct) if distinct else 0
            ),
        },
        "live_transaction": live,
        "live_actions": actions,
        "replay": replay,
        "derived": {
            "hybrid_vs_occ_throughput_ratio": hybrid_tp / occ_tp,
            "hybrid_vs_occ_improvement_pct": (hybrid_tp / occ_tp - 1) * 100,
            "hybrid_vs_2pl_ratio": hybrid_tp / two_pl_tp,
            "hybrid_k3_vs_k1_improvement_pct": (hybrid_tp / hybrid_k1_tp - 1) * 100,
            "merge_all_vs_hybrid_ratio": merge_tp / hybrid_tp,
            "occ_regen_avoided": replay["OCC"]["regen"] - replay["HYBRID"]["regen"],
            "merge_all_oversell": replay["merge-all"]["oversell"][0],
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    latency = analysis["generation_latency_s"]
    candidate = analysis["candidates"]
    derived = analysis["derived"]
    lines = [
        "# DeepSeek Real Agent Transaction Experiment",
        "",
        f"- Model: {cache.get('model')}; mock={cache.get('mock')}.",
        f"- Tasks: {len(records)}; API errors: {cache.get('errs')}; live wall-clock: {cache.get('wall'):.2f}s.",
        (
            f"- Generation latency: mean {latency['mean']:.2f}s, P50 {latency['median']:.2f}s, "
            f"P95 {latency['p95']:.2f}s, P99 {latency['p99']:.2f}s."
        ),
        f"- Tokens: total {analysis['tokens']['total']}, mean {analysis['tokens']['mean_per_task']:.1f}/task.",
        (
            f"- Candidates: mean {candidate['mean_distinct']:.2f} distinct flights; "
            f"{candidate['reselectable_fraction'] * 100:.1f}% of tasks have at least 2 alternatives."
        ),
        "",
        "## Live End-To-End Transactions",
        "",
        (
            f"- Booked {live.get('booked', 0)}, correctly rejected {live.get('no_seat', 0)}, "
            f"reselect {live.get('reselect', 0)}, merge {live.get('merge', 0)}."
        ),
        (
            f"- Regenerate {live.get('regen', 0)}, oversell {live.get('oversell', 0)}, "
            f"throughput {live.get('throughput', 0):.2f} txns/s."
        ),
        f"- Commit action distribution: {actions}.",
        "",
        "## Same-Trace Replay",
        "",
        "Replay uses the same real DeepSeek candidates and recorded generation latency. "
        "Sleep is uniformly scaled by speed=20, so absolute throughput is a replay-scale "
        "number; policy ratios are the relevant comparison.",
        "",
        "| Policy | Throughput/s | 95%CI | regen | reselect | no-seat reject | oversell |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ["OCC", "2PL", "merge-all", "HYBRID-K1", "HYBRID"]:
        row = replay[policy]
        lines.append(
            f"| {policy} | {row['throughput'][0]:.1f} | +/-{row['throughput'][1]:.1f} | "
            f"{row['regen']:.0f} | {row['reselect']:.0f} | {row['no_seat']:.0f} | "
            f"{row['oversell'][0]:.0f} |"
        )
    lines += [
        "",
        "## Conclusions",
        "",
        (
            f"- HYBRID improves throughput over OCC+K by "
            f"{derived['hybrid_vs_occ_improvement_pct']:.1f}% and reduces average "
            f"regenerate from {replay['OCC']['regen']:.0f} to {replay['HYBRID']['regen']:.0f}."
        ),
        f"- HYBRID throughput is {derived['hybrid_vs_2pl_ratio']:.2f}x 2PL.",
        (
            f"- Under the same semantic route, K3 improves over K1 by "
            f"{derived['hybrid_k3_vs_k1_improvement_pct']:.1f}%, quantifying candidate-reselect benefit."
        ),
        (
            f"- merge-all has higher throughput but produces {derived['merge_all_oversell']:.0f} "
            "oversell events on average, so it is not a correctness-preserving baseline."
        ),
        "- This experiment uses the built-in OTA catalog and real DeepSeek decisions. It is evidence for the agent transaction-layer mechanism, not for production database backend performance.",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print("saved", OUT_JSON)
    print("saved", OUT_MD)


if __name__ == "__main__":
    main()

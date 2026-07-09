"""Report rendering helpers for evaluation modules."""

from __future__ import annotations

from typing import Any, Mapping


def render_atcc_ablation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ATCC Ablation Report",
        "",
        "## Configuration",
        "",
        f"- workloads: {', '.join(report.get('workloads', []))}",
        f"- profiles: {', '.join(report.get('profiles', []))}",
        f"- variants: {', '.join(report.get('variants', []))}",
        f"- seeds: {', '.join(str(seed) for seed in report.get('seeds', []))}",
        f"- task_count: {report.get('task_count')}",
        f"- train_seeds: {', '.join(str(seed) for seed in report.get('train_seeds', []))}",
        f"- train_rounds: {report.get('train_rounds')}",
        f"- train_task_count: {report.get('train_task_count')}",
        f"- train_policy_epsilon: {report.get('train_policy_epsilon')}",
        f"- validation_seeds: {', '.join(str(seed) for seed in report.get('validation_seeds', []))}",
        f"- validation_task_count: {report.get('validation_task_count')}",
        f"- priority_cap: {report.get('priority_cap')}",
        f"- freeze_dynamic_policy: {report.get('freeze_dynamic_policy')}",
        f"- static_preset: {report.get('static_preset')}",
        f"- static_operation_wide_overwrite_threshold: {report.get('static_operation_wide_overwrite_threshold')}",
        f"- static_transaction_wide_write_threshold: {report.get('static_transaction_wide_write_threshold')}",
        "",
        "## Metrics",
        "",
        "| workload | profile | variant | agent tput | total tput | commit rate | attempts/task | p99 latency | p99.99 latency | abort rate | avg token/task | waste token/task | selected from |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("metrics", []):
        values = dict(row)
        values.setdefault("selected_from", "")
        lines.append(
            "| {workload_kind} | {profile} | {variant} | {committed_throughput:.3f} | "
            "{total_throughput:.3f} | {commit_rate:.1%} | {attempts_per_task:.3f} | "
            "{agent_latency_p99_s:.3f} | {agent_latency_p9999_s:.3f} | "
            "{conflict_abort_rate:.1%} | {estimated_tokens_per_task:.1f} | "
            "{estimated_wasted_tokens_per_task:.1f} | {selected_from} |".format(**values)
        )
    selections = report.get("validation_selections", {})
    if selections:
        lines.extend(
            [
                "",
                "## Validation Selections",
                "",
                "| selected variant | source | validated against | criterion |",
                "| --- | --- | --- | --- |",
            ]
        )
        for key, row in sorted(dict(selections).items()):
            lines.append(
                f"| {key} | {row.get('source', '')} | "
                f"{row.get('validated_against', '')} | {row.get('criterion', '')} |"
            )
    lines.extend(
        [
            "",
            "## Ratios",
            "",
            "| workload | profile | comparison | ratio | note |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in report.get("ratios", []):
        ratio = row.get("throughput_ratio")
        ratio_text = "" if ratio in (None, "") else f"{float(ratio):.3f}x"
        lines.append(
            f"| {row['workload_kind']} | {row['profile']} | "
            f"{row['comparison']} | {ratio_text} | {row.get('note', '')} |"
        )
    return "\n".join(lines) + "\n"

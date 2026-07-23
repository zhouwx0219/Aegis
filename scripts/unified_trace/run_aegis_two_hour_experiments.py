#!/usr/bin/env python3
"""Run the bounded Aegis experiment plan and export one raw CSV per figure."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "unified_trace" / "generate_castdas_trace.py"
RUNNER = ROOT / "scripts" / "unified_trace" / "run_castdas_trace_fair.py"
SYSTEMS = "2pl-wait-die,bamboo,silo,polaris,paper-atcc"
SYSTEM_LABELS = {
    "2pl-wait-die": "2PL",
    "bamboo": "Bamboo",
    "silo": "Silo",
    "polaris": "Polaris",
    "paper-atcc": "Aegis",
}
PAPER_MAX_RETRIES = 5
PAPER_MAX_ATTEMPTS = PAPER_MAX_RETRIES + 1


@dataclasses.dataclass(frozen=True)
class Case:
    group: str
    workload: str
    parameter: str
    value: str
    clients: int = 32
    repeat: int = 0
    systems: str = SYSTEMS
    reasoning_scale: float = 1.0
    generator_args: tuple[str, ...] = ()
    switching: str = "dynamic"
    priority: str = "enabled"
    delayed_write_apply: str = "enabled"
    performance_guards: str = "enabled"
    priority_scale: float = 1.0
    policy_invocation_ops: int = 0
    ablation: str = ""
    execution_workers: int = 0
    trace_variant: str = ""

    @property
    def seed(self) -> int:
        value = "paired" if self.parameter == "ablation" else self.value
        payload = f"{self.group}:{self.workload}:{self.parameter}:{value}:{self.clients}:{self.repeat}"
        return 970_000 + zlib.crc32(payload.encode()) % 20_000

    @property
    def case_id(self) -> str:
        raw = f"{self.workload}_{self.parameter}_{self.value}_c{self.clients}_r{self.repeat}_{self.ablation}"
        return "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=group_names() + ["all"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-policy", type=Path, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=0.25)
    parser.add_argument("--measure-seconds", type=float, default=1.5)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=PAPER_MAX_ATTEMPTS,
        help="Total attempts including the initial attempt (paper default: 6).",
    )
    parser.add_argument("--reasoning-profile", default="agentic")
    parser.add_argument("--reasoning-scale-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run every configuration point once instead of using all repeats.",
    )
    parser.add_argument(
        "--points",
        default="",
        help=(
            "Optional comma-separated parameter=value points to run, for example "
            "write_ratio=0.9 or reasoning_scale=4.0,transaction_length=24."
        ),
    )
    parser.add_argument(
        "--workloads",
        default="",
        help="Optional comma-separated workload filter (ycsb,tpcc).",
    )
    parser.add_argument(
        "--systems",
        default="",
        help="Override the default five-system matrix; Aegis-only cases are skipped.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")
    if args.reasoning_scale_multiplier <= 0:
        raise SystemExit("--reasoning-scale-multiplier must be positive")
    points = parse_points(args.points)
    workloads = parse_workloads(args.workloads)

    groups = group_names() if args.group == "all" else [args.group]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        run_group(
            group,
            output_dir=args.output_dir,
            paper_policy=args.paper_policy,
            warmup_seconds=args.warmup_seconds,
            measure_seconds=args.measure_seconds,
            max_attempts=args.max_attempts,
            reasoning_profile=args.reasoning_profile,
            reasoning_scale_multiplier=args.reasoning_scale_multiplier,
            systems=args.systems,
            force=args.force,
            quick=args.quick,
            points=points,
            workloads=workloads,
        )
    return 0


def group_names() -> list[str]:
    return [
        "ycsb_scalability",
        "tpcc_scalability",
        "agent_worker_decoupling",
        "contention_sensitivity",
        "shape_sensitivity",
        "read_write_sensitivity",
        "ratio_control_sensitivity",
        "ablation_tpcc",
        "ablation",
    ]


def parse_points(value: str) -> frozenset[tuple[str, str]]:
    points: set[tuple[str, str]] = set()
    for item in (part.strip() for part in str(value).split(",")):
        if not item:
            continue
        parameter, separator, point_value = item.partition("=")
        if not separator or not parameter.strip() or not point_value.strip():
            raise SystemExit(f"invalid --points item: {item!r}")
        points.add((parameter.strip(), point_value.strip()))
    return frozenset(points)


def parse_workloads(value: str) -> frozenset[str]:
    workloads = frozenset(
        part.strip().lower() for part in str(value).split(",") if part.strip()
    )
    unknown = workloads - {"ycsb", "tpcc"}
    if unknown:
        raise SystemExit(f"invalid --workloads values: {','.join(sorted(unknown))}")
    return workloads


def cases_for(group: str) -> list[Case]:
    cases: list[Case] = []
    if group in {"ycsb_scalability", "tpcc_scalability"}:
        workload = "ycsb" if group.startswith("ycsb") else "tpcc"
        for clients in (8, 16, 24, 32, 40):
            for repeat in range(3):
                cases.append(Case(group, workload, "workers", str(clients), clients, repeat))
        return cases
    if group == "agent_worker_decoupling":
        for workers in (8, 16, 24, 32, 40):
            for repeat in range(3):
                cases.append(
                    Case(
                        group,
                        "ycsb",
                        "execution_workers",
                        str(workers),
                        clients=40,
                        repeat=repeat,
                        execution_workers=workers,
                    )
                )
        return cases
    if group == "contention_sensitivity":
        for value in (0.0, 0.5, 0.8, 0.99, 1.2):
            for repeat in range(3):
                cases.append(Case(group, "ycsb", "zipf_theta", str(value), repeat=repeat,
                                  generator_args=("--ycsb-access-distribution", "zipfian", "--ycsb-zipf-theta", str(value))))
        for value in (8, 32, 128, 512, 2048):
            for repeat in range(3):
                cases.append(Case(group, "ycsb", "hotset_size", str(value), repeat=repeat,
                                  generator_args=("--ycsb-access-distribution", "hotspot", "--ycsb-hotset-size", str(value), "--ycsb-hotspot-access-probability", "0.8")))
        return cases
    if group == "shape_sensitivity":
        for workload in ("ycsb", "tpcc"):
            for value in (0.25, 0.5, 1.0, 2.0, 4.0):
                for repeat in range(3):
                    cases.append(Case(
                        group,
                        workload,
                        "reasoning_scale",
                        str(value),
                        repeat=repeat,
                        reasoning_scale=value,
                    ))
        for value in (4, 8, 12, 16, 24):
            for repeat in range(3):
                cases.append(Case(group, "ycsb", "transaction_length", str(value), repeat=repeat,
                                  generator_args=("--ycsb-operations", str(value))))
        for value in (5, 8, 10, 12, 15):
            for repeat in range(3):
                cases.append(Case(group, "tpcc", "transaction_length", str(value), repeat=repeat,
                                  generator_args=("--tpcc-order-lines", str(value))))
        return cases
    if group in {"read_write_sensitivity", "ratio_control_sensitivity"}:
        for value in (0.1, 0.25, 0.5, 0.75, 0.9):
            for repeat in range(3):
                cases.append(Case(group, "ycsb", "write_ratio", str(value), repeat=repeat,
                                  generator_args=("--ycsb-write-ratio", str(value))))
        if group == "read_write_sensitivity":
            return cases
        for workload in ("ycsb", "tpcc"):
            for value in (0.5, 1.0, 2.0, 4.0):
                for repeat in range(3):
                    cases.append(Case(group, workload, "priority_quantum_scale", str(value), repeat=repeat,
                                      systems="paper-atcc", priority_scale=value))
        length = 10
        for invocations in (1, 2, 3, 6):
            batch = max(1, math.ceil(length / invocations))
            for repeat in range(3):
                cases.append(Case(group, "ycsb", "policy_invocations_per_txn", str(invocations), repeat=repeat,
                                  systems="paper-atcc", policy_invocation_ops=batch,
                                  generator_args=("--policy-invocation-ops", str(batch))))
        return cases
    if group in {"ablation", "ablation_tpcc"}:
        variants = (
            ("Static", "static", "disabled", "disabled"),
            ("Static + DWA", "static", "disabled", "enabled"),
            ("Static + DWA + Priority", "static", "enabled", "enabled"),
            ("Dynamic", "dynamic", "disabled", "disabled"),
            ("Dynamic + DWA", "dynamic", "disabled", "enabled"),
            ("Dynamic + DWA + Priority", "dynamic", "enabled", "enabled"),
        )
        workloads = ("tpcc",) if group == "ablation_tpcc" else ("ycsb", "tpcc")
        repeats = 3 if group == "ablation_tpcc" else 2
        for workload in workloads:
            for label, switching, priority, delayed_write_apply in variants:
                for repeat in range(repeats):
                    cases.append(Case(group, workload, "ablation", label, repeat=repeat,
                                      systems="paper-atcc", switching=switching,
                                      priority=priority,
                                      delayed_write_apply=delayed_write_apply,
                                      performance_guards="disabled",
                                      ablation=label,
                                      trace_variant="tpcc_high_w1" if group == "ablation_tpcc" else ""))
        return cases
    raise ValueError(group)


def run_group(
    group: str,
    *,
    output_dir: Path,
    paper_policy: Path,
    warmup_seconds: float,
    measure_seconds: float,
    max_attempts: int,
    reasoning_profile: str,
    reasoning_scale_multiplier: float,
    systems: str,
    force: bool,
    quick: bool = False,
    points: frozenset[tuple[str, str]] = frozenset(),
    workloads: frozenset[str] = frozenset(),
) -> None:
    case_dir = output_dir / "cases" / group
    trace_dir = output_dir / "traces" / group
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    cases = cases_for(group)
    if workloads:
        cases = [case for case in cases if case.workload in workloads]
    if points:
        cases = [case for case in cases if (case.parameter, case.value) in points]
        if not cases:
            raise SystemExit(f"--points selected no cases for group {group}")
    if quick:
        cases = [case for case in cases if case.repeat == 0]
    if str(systems).strip():
        cases = [
            dataclasses.replace(case, systems=str(systems).strip())
            for case in cases
            if case.systems == SYSTEMS
        ]
    for index, case in enumerate(cases, 1):
        trace = trace_dir / f"{case.case_id}.csv"
        result = case_dir / f"{case.case_id}.csv"
        if force or not trace.exists():
            generate_trace(
                case,
                trace,
                reasoning_profile=reasoning_profile,
                reasoning_scale_multiplier=reasoning_scale_multiplier,
            )
        if force or not valid_result(result, case.systems, max_attempts=max_attempts):
            run_case(
                case,
                trace,
                result,
                paper_policy,
                warmup_seconds,
                measure_seconds,
                max_attempts,
            )
        print(f"[{group}] {index}/{len(cases)} {case.case_id}", flush=True)
    export_group(group, cases, case_dir, output_dir / f"figure_{group}_raw.csv")


def generate_trace(
    case: Case,
    output: Path,
    *,
    reasoning_profile: str,
    reasoning_scale_multiplier: float,
) -> None:
    variant = case.trace_variant or (
        "ycsb_high_z099" if case.workload == "ycsb" else "tpcc_high_w1"
    )
    command = [
        sys.executable, str(GENERATOR), "--output", str(output), "--variant", variant,
        "--clients", str(case.clients), "--agent-ratio", "1.0", "--seed", str(case.seed),
        "--repeat", str(case.repeat), "--transactions-per-worker", "64",
        "--reasoning-profile", str(reasoning_profile),
        "--reasoning-scale", str(case.reasoning_scale * reasoning_scale_multiplier),
        *case.generator_args,
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def run_case(
    case: Case,
    trace: Path,
    output: Path,
    policy: Path,
    warmup: float,
    measure: float,
    max_attempts: int,
) -> None:
    command = [
        sys.executable, str(RUNNER), "--trace", str(trace), "--output", str(output),
        "--cc", case.systems, "--paper-policy", str(policy), "--policy-mode", "eval",
        "--max-attempts", str(max_attempts), "--warmup-seconds", str(warmup),
        "--measure-seconds", str(measure), "--disable-atcc-retry-cache",
        "--paper-switching", case.switching, "--paper-priority", case.priority,
        "--paper-delayed-write-apply", case.delayed_write_apply,
        "--paper-performance-guards", case.performance_guards,
        "--priority-quantum-scale", str(case.priority_scale),
        "--tpcc-replay-capacity", "1",
        "--ycsb-replay-capacity", "1",
    ]
    if max_attempts > 1:
        command.append("--allow-retries")
    if warmup > 0:
        # Warm every system with the same fixed trace. This supplies Aegis with
        # online-observed hotness before measurement and stabilizes the first
        # high-contention cohort without priming a future access-set oracle.
        command.extend(("--warmup-trace", str(trace)))
    if case.execution_workers > 0:
        command.extend(("--execution-workers", str(case.execution_workers)))
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def valid_result(path: Path, systems: str, *, max_attempts: int) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == len([value for value in systems.split(",") if value]) and all(
        row.get("status") == "ok"
        and int(float(row.get("max_attempts", 0) or 0)) == int(max_attempts)
        and int(float(row.get("retry_budget", -1) or -1))
        == max(0, int(max_attempts) - 1)
        and int(float(row.get("tpcc_replay_capacity", 0) or 0)) == 1
        and int(float(row.get("ycsb_replay_capacity", 0) or 0)) == 1
        for row in rows
    )


def export_group(group: str, cases: Iterable[Case], case_dir: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        with (case_dir / f"{case.case_id}.csv").open(newline="", encoding="utf-8-sig") as handle:
            for source in csv.DictReader(handle):
                cc = source.get("cc", "")
                row = dict(source)
                # Paper figures use logical Agent task throughput uniformly.
                # Keep total/native TPS as auxiliary runtime diagnostics only.
                row["throughput"] = source.get("agent_tps", "")
                row["throughput_metric"] = "agent_tps"
                row.update({
                    "experiment": group,
                    "workload": case.workload.upper() if case.workload == "ycsb" else "TPC-C",
                    "parameter": case.parameter,
                    "parameter_value": case.value,
                    "clients": case.clients,
                    "agent_count": case.clients,
                    "worker_count": case.execution_workers or case.clients,
                    "seed": case.seed,
                    "repeat": case.repeat,
                    "system": case.ablation or SYSTEM_LABELS.get(cc, cc),
                    "cc": cc,
                    "paper_switching": case.switching if cc == "paper-atcc" else "",
                    "paper_priority": case.priority if cc == "paper-atcc" else "",
                    "paper_delayed_write_apply": case.delayed_write_apply if cc == "paper-atcc" else "",
                    "paper_performance_guards": case.performance_guards if cc == "paper-atcc" else "",
                    "priority_quantum_scale": case.priority_scale if cc == "paper-atcc" else "",
                    "policy_invocation_ops": case.policy_invocation_ops if cc == "paper-atcc" else "",
                })
                rows.append(row)
    fields = list(rows[0]) if rows else []
    if fields:
        for field in ("throughput", "throughput_metric"):
            fields.remove(field)
        anchor = fields.index("total_tps") if "total_tps" in fields else len(fields)
        fields[anchor:anchor] = ["throughput", "throughput_metric"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

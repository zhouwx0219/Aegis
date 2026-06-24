"""Run DBx1000's native executable as authoritative CC baselines.

ASTRA keeps transaction semantics and concurrency control in the agent runtime.
This module is intentionally separate: it builds and runs the vendored DBx1000
benchmark executable so native DBx1000 CC algorithms can be compared as external
baselines without moving their semantics into the ASTRA versioned-KV backend.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DBX1000_ROOT = _REPO_ROOT / "third_party" / "dbx1000"


@dataclasses.dataclass(frozen=True)
class NativeCCStrategy:
    name: str
    cc_alg: str
    family: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


NATIVE_CC_STRATEGIES: Dict[str, NativeCCStrategy] = {
    "no_wait": NativeCCStrategy("no_wait", "NO_WAIT", "pessimistic", "DBx1000 native no-wait 2PL"),
    "wait_die": NativeCCStrategy("wait_die", "WAIT_DIE", "pessimistic", "DBx1000 native wait-die 2PL"),
    "dl_detect": NativeCCStrategy("dl_detect", "DL_DETECT", "pessimistic", "DBx1000 native deadlock-detecting 2PL"),
    "mvcc": NativeCCStrategy("mvcc", "MVCC", "multiversion", "DBx1000 native MVCC"),
    "occ": NativeCCStrategy("occ", "OCC", "optimistic", "DBx1000 native OCC"),
    "tictoc": NativeCCStrategy("tictoc", "TICTOC", "timestamp", "DBx1000 native TicToc"),
    "silo": NativeCCStrategy("silo", "SILO", "timestamp", "DBx1000 native Silo"),
}


@dataclasses.dataclass(frozen=True)
class Dbx1000NativeConfig:
    workload: str = "YCSB"
    threads: int = 4
    partitions: int = 1
    virtual_partitions: int = 1
    max_txn_per_part: int = 100000
    warmup: int = 0
    ycsb_table_size: int = 1024 * 1024
    ycsb_req_per_query: int = 16
    ycsb_read_perc: float = 0.9
    ycsb_write_perc: float = 0.1
    ycsb_zipf_theta: float = 0.6
    tpcc_warehouses: int = 1
    tpcc_payment_perc: float = 0.5
    extra_args: Tuple[str, ...] = ()

    def workload_macro(self) -> str:
        name = self.workload.strip().upper().replace("-", "")
        if name in {"YCSB", "TPCC"}:
            return name
        if name == "TPC_C":
            return "TPCC"
        raise ValueError(f"unsupported DBx1000 native workload: {self.workload}")

    def runtime_args(self, output_file: Path) -> Tuple[str, ...]:
        args = [
            f"-t{self.threads}",
            f"-p{self.partitions}",
            f"-v{self.virtual_partitions}",
            "-o",
            str(output_file),
        ]
        if self.workload_macro() == "YCSB":
            args.extend(
                [
                    f"-s{self.ycsb_table_size}",
                    f"-R{self.ycsb_req_per_query}",
                    f"-r{self.ycsb_read_perc}",
                    f"-w{self.ycsb_write_perc}",
                    f"-z{self.ycsb_zipf_theta}",
                ]
            )
        else:
            args.extend([f"-n{self.tpcc_warehouses}", f"-Tp{self.tpcc_payment_perc}"])
        args.extend(self.extra_args)
        return tuple(args)

    def to_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["workload"] = self.workload_macro()
        row["extra_args"] = list(self.extra_args)
        return row


@dataclasses.dataclass(frozen=True)
class Dbx1000NativeResult:
    strategy: str
    workload: str
    config: Mapping[str, Any]
    workload_manifest: Mapping[str, Any]
    strategy_metadata: Mapping[str, Any]
    command: Tuple[str, ...]
    elapsed_s: float
    returncode: int
    summary: Mapping[str, Any]
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["command"] = list(self.command)
        return row


def list_native_strategies() -> Dict[str, Dict[str, str]]:
    return {name: strategy.to_dict() for name, strategy in sorted(NATIVE_CC_STRATEGIES.items())}


def run_native_matrix(
    strategies: Iterable[str],
    config: Dbx1000NativeConfig,
    *,
    dbx1000_root: Path = _DBX1000_ROOT,
    keep_build_dir: bool = False,
    build_timeout_s: int = 300,
    run_timeout_s: int = 300,
) -> Sequence[Dbx1000NativeResult]:
    return tuple(
        run_native_strategy(
            strategy,
            config,
            dbx1000_root=dbx1000_root,
            keep_build_dir=keep_build_dir,
            build_timeout_s=build_timeout_s,
            run_timeout_s=run_timeout_s,
        )
        for strategy in strategies
    )


def run_native_strategy(
    strategy_name: str,
    config: Dbx1000NativeConfig,
    *,
    dbx1000_root: Path = _DBX1000_ROOT,
    keep_build_dir: bool = False,
    build_timeout_s: int = 300,
    run_timeout_s: int = 300,
) -> Dbx1000NativeResult:
    strategy = _strategy(strategy_name)
    dbx1000_root = Path(dbx1000_root)
    if not (dbx1000_root / "Makefile").exists():
        raise FileNotFoundError(f"DBx1000 root does not contain Makefile: {dbx1000_root}")

    temp_context = tempfile.TemporaryDirectory(prefix=f"astra-dbx1000-{strategy.name}-")
    try:
        build_dir = Path(temp_context.name) / "dbx1000"
        _copy_dbx1000_tree(dbx1000_root, build_dir)
        _patch_config(
            build_dir / "config.h",
            {
                "WORKLOAD": config.workload_macro(),
                "CC_ALG": strategy.cc_alg,
                "THREAD_CNT": str(config.threads),
                "PART_CNT": str(config.partitions),
                "VIRTUAL_PART_CNT": str(config.virtual_partitions),
                "WARMUP": str(config.warmup),
                "MAX_TXN_PER_PART": str(config.max_txn_per_part),
            },
        )
        build = subprocess.run(
            ["make", "clean", "all"],
            cwd=build_dir,
            text=True,
            capture_output=True,
            timeout=build_timeout_s,
        )
        if build.returncode != 0:
            return _failed_result(strategy, config, ("make", "clean", "all"), build, 0.0)

        output_file = build_dir / "native_summary.txt"
        command = ("./rundb", *config.runtime_args(output_file))
        start = time.perf_counter()
        run = subprocess.run(
            list(command),
            cwd=build_dir,
            text=True,
            capture_output=True,
            timeout=run_timeout_s,
        )
        elapsed = time.perf_counter() - start
        summary_text = output_file.read_text(encoding="utf-8", errors="replace") if output_file.exists() else run.stdout
        return Dbx1000NativeResult(
            strategy=strategy.name,
            workload=config.workload_macro().lower(),
            config=config.to_dict(),
            workload_manifest=_native_manifest(config),
            strategy_metadata=strategy.to_dict(),
            command=command,
            elapsed_s=elapsed,
            returncode=run.returncode,
            summary=parse_summary(summary_text),
            stdout_tail=_tail(run.stdout),
            stderr_tail=_tail(run.stderr),
        )
    finally:
        if keep_build_dir:
            print(f"kept DBx1000 native build directory: {temp_context.name}", file=sys.stderr)
            temp_context._finalizer.detach()  # type: ignore[attr-defined]
        else:
            temp_context.cleanup()


def parse_summary(text: str) -> Dict[str, Any]:
    summary_line = ""
    for line in text.splitlines():
        if line.startswith("[summary]"):
            summary_line = line
    if not summary_line:
        return {}
    fields: Dict[str, Any] = {}
    body = summary_line.removeprefix("[summary]").strip()
    for part in body.split(","):
        if "=" not in part:
            continue
        key, value = [item.strip() for item in part.split("=", 1)]
        fields[key] = _parse_number(value)
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DBx1000 native CC baselines.")
    parser.add_argument("--workload", choices=("ycsb", "tpcc"), default="ycsb")
    parser.add_argument("--strategies", default="occ,mvcc,tictoc,silo,no_wait")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--virtual-partitions", type=int, default=1)
    parser.add_argument("--max-txn-per-part", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--ycsb-table-size", type=int, default=1024 * 1024)
    parser.add_argument("--ycsb-req-per-query", type=int, default=16)
    parser.add_argument("--ycsb-read-perc", type=float, default=0.9)
    parser.add_argument("--ycsb-write-perc", type=float, default=0.1)
    parser.add_argument("--ycsb-zipf-theta", type=float, default=0.6)
    parser.add_argument("--tpcc-warehouses", type=int, default=1)
    parser.add_argument("--tpcc-payment-perc", type=float, default=0.5)
    parser.add_argument("--dbx1000-root", type=Path, default=_DBX1000_ROOT)
    parser.add_argument("--keep-build-dir", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None, *, stdout: Optional[TextIO] = None) -> int:
    args = build_parser().parse_args(argv)
    config = Dbx1000NativeConfig(
        workload=args.workload.upper(),
        threads=args.threads,
        partitions=args.partitions,
        virtual_partitions=args.virtual_partitions,
        max_txn_per_part=args.max_txn_per_part,
        warmup=args.warmup,
        ycsb_table_size=args.ycsb_table_size,
        ycsb_req_per_query=args.ycsb_req_per_query,
        ycsb_read_perc=args.ycsb_read_perc,
        ycsb_write_perc=args.ycsb_write_perc,
        ycsb_zipf_theta=args.ycsb_zipf_theta,
        tpcc_warehouses=args.tpcc_warehouses,
        tpcc_payment_perc=args.tpcc_payment_perc,
    )
    strategies = tuple(part.strip() for part in args.strategies.split(",") if part.strip())
    results = run_native_matrix(
        strategies,
        config,
        dbx1000_root=args.dbx1000_root,
        keep_build_dir=args.keep_build_dir,
    )
    report = {
        "workload_layer": "native",
        "strategies": list(strategies),
        "config": config.to_dict(),
        "results": [result.to_dict() for result in results],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        (stdout or sys.stdout).write(text + "\n")
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _strategy(name: str) -> NativeCCStrategy:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in NATIVE_CC_STRATEGIES:
        raise ValueError(f"unknown DBx1000 native CC strategy: {name}")
    return NATIVE_CC_STRATEGIES[normalized]


def _native_manifest(config: Dbx1000NativeConfig) -> Dict[str, Any]:
    workload = config.workload_macro()
    if workload == "YCSB":
        return {
            "name": "dbx1000-ycsb-native",
            "benchmark_family": "YCSB",
            "source_system": "DBx1000-native",
            "source_files": [
                "third_party/dbx1000/benchmarks/ycsb.h",
                "third_party/dbx1000/benchmarks/ycsb_wl.cpp",
                "third_party/dbx1000/benchmarks/ycsb_txn.cpp",
                "third_party/dbx1000/benchmarks/ycsb_query.cpp",
                "third_party/dbx1000/benchmarks/YCSB_schema.txt",
            ],
            "preserved_semantics": [
                "DBx1000 native YCSB query generator",
                "DBx1000 native transaction execution path",
                "DBx1000 native compile-time CC_ALG selection",
            ],
            "agent_adaptations": [],
            "workload_layer": "native",
            "canonical_name": "dbx1000-ycsb-native",
            "config": config.to_dict(),
        }
    return {
        "name": "dbx1000-tpcc-native",
        "benchmark_family": "TPC-C",
        "source_system": "DBx1000-native",
        "source_files": [
            "third_party/dbx1000/benchmarks/tpcc.h",
            "third_party/dbx1000/benchmarks/tpcc_wl.cpp",
            "third_party/dbx1000/benchmarks/tpcc_txn.cpp",
            "third_party/dbx1000/benchmarks/tpcc_query.cpp",
            "third_party/dbx1000/benchmarks/TPCC_full_schema.txt",
        ],
        "preserved_semantics": [
            "DBx1000 native TPC-C Payment/NewOrder benchmark surface",
            "DBx1000 native transaction execution path",
            "DBx1000 native compile-time CC_ALG selection",
        ],
        "agent_adaptations": [],
        "workload_layer": "native",
        "canonical_name": "dbx1000-tpcc-native",
        "config": config.to_dict(),
    }


def _copy_dbx1000_tree(source: Path, target: Path) -> None:
    def ignore(_: str, names: Sequence[str]) -> set[str]:
        return {name for name in names if name.endswith((".o", ".d")) or name == "rundb"}

    shutil.copytree(source, target, ignore=ignore)


def _patch_config(path: Path, replacements: Mapping[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for macro, value in replacements.items():
        pattern = re.compile(rf"^#define\s+{re.escape(macro)}\s+.*$", re.MULTILINE)
        replacement = f"#define {macro}\t\t\t\t\t{value}"
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise ValueError(f"could not patch DBx1000 config macro: {macro}")
    path.write_text(text, encoding="utf-8")


def _failed_result(
    strategy: NativeCCStrategy,
    config: Dbx1000NativeConfig,
    command: Tuple[str, ...],
    process: subprocess.CompletedProcess[str],
    elapsed_s: float,
) -> Dbx1000NativeResult:
    return Dbx1000NativeResult(
        strategy=strategy.name,
        workload=config.workload_macro().lower(),
        config=config.to_dict(),
        workload_manifest=_native_manifest(config),
        strategy_metadata=strategy.to_dict(),
        command=command,
        elapsed_s=elapsed_s,
        returncode=process.returncode,
        summary={},
        stdout_tail=_tail(process.stdout),
        stderr_tail=_tail(process.stderr),
    )


def _parse_number(value: str) -> Any:
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


if __name__ == "__main__":
    raise SystemExit(main())

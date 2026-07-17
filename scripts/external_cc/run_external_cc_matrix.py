#!/usr/bin/env python3
"""Run patched external DBx1000-family CC systems and emit CAST-DAS CSV."""

import argparse
import csv
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


SUMMARY_RE = re.compile(r"\[summary\]\s+(?P<body>.*)")


SYSTEMS = {
    "bamboo": {
        "source_dir": "Bamboo-Public",
        "work_dir": "Bamboo-Public_castdas",
        "python": "python",
        "algorithms": ["BAMBOO", "SILO", "NO_WAIT", "WAIT_DIE", "WOUND_WAIT"],
        "compat": True,
        "compiler_candidates": [],
    },
    "polaris": {
        "source_dir": "polaris",
        "work_dir": "polaris_castdas",
        "python": "python3",
        "algorithms": ["SILO_PRIO", "SILO", "NO_WAIT", "WAIT_DIE", "WOUND_WAIT", "BAMBOO"],
        "compat": False,
        "compiler_candidates": [
            "/home/chenht/miniconda3-castdas/bin/x86_64-conda-linux-gnu-g++",
            "/usr/bin/g++-13",
            "/usr/bin/g++-12",
            "/usr/bin/g++-11",
            "/usr/bin/g++-10",
        ],
    },
    "plor": {
        "source_dir": "Plor",
        "work_dir": "Plor_castdas",
        "python": "",
        "algorithms": ["PLOR", "SILO", "NO_WAIT", "WAIT_DIE", "WOUND_WAIT", "HLOCK", "MOCC"],
        "compat": True,
        "runner": "plor",
        "compiler_candidates": [],
    },
}


YCSB_LEVELS = {
    "low": {
        "CASTDAS_AGENT_LEVEL": 0,
        "SYNTH_TABLE_SIZE": 5120,
        "REQ_PER_QUERY": 10,
        "READ_PERC": 0.95,
        "ZIPF_THETA": 0.0,
        "SYNTHETIC_YCSB": "false",
    },
    "medium": {
        "CASTDAS_AGENT_LEVEL": 1,
        "SYNTH_TABLE_SIZE": 1280,
        "REQ_PER_QUERY": 10,
        "READ_PERC": 0.90,
        "ZIPF_THETA": 0.7,
        "SYNTHETIC_YCSB": "false",
    },
    "medium_z08": {
        "CASTDAS_AGENT_LEVEL": 1,
        "SYNTH_TABLE_SIZE": 1280,
        "REQ_PER_QUERY": 10,
        "READ_PERC": 0.90,
        "ZIPF_THETA": 0.8,
        "SYNTHETIC_YCSB": "false",
    },
    "high": {
        "CASTDAS_AGENT_LEVEL": 2,
        "SYNTH_TABLE_SIZE": 640,
        "REQ_PER_QUERY": 10,
        "READ_PERC": 0.50,
        "ZIPF_THETA": 0.99,
        "SYNTHETIC_YCSB": "false",
    },
}


TPCC_LEVELS = {
    "low": {"CASTDAS_AGENT_LEVEL": 0, "NUM_WH": 48, "PERC_PAYMENT": 0.45},
    "low_w100": {"CASTDAS_AGENT_LEVEL": 0, "NUM_WH": 100, "PERC_PAYMENT": 0.45},
    "medium": {"CASTDAS_AGENT_LEVEL": 1, "NUM_WH": 2, "PERC_PAYMENT": 0.45},
    "high": {"CASTDAS_AGENT_LEVEL": 2, "NUM_WH": 1, "PERC_PAYMENT": 0.45},
}


VARIANTS = {
    "ycsb_low": ("ycsb", "low", "low"),
    "ycsb_medium_z07": ("ycsb", "medium", "medium"),
    "ycsb_medium_z08": ("ycsb", "medium", "medium_z08"),
    "ycsb_high_z099": ("ycsb", "high", "high"),
    "tpcc_low_w100": ("tpcc", "low", "low_w100"),
    "tpcc_medium": ("tpcc", "medium", "medium"),
    "tpcc_high_w1": ("tpcc", "high", "high"),
}


CSV_FIELDS = [
    "system",
    "cc",
    "workload",
    "workload_variant",
    "level",
    "clients",
    "agent_ratio",
    "duration_s",
    "warmup_s",
    "repeat",
    "status",
    "throughput",
    "txn_cnt",
    "abort_cnt",
    "user_abort_cnt",
    "agent_txn_cnt",
    "agent_abort_cnt",
    "background_txn_cnt",
    "background_abort_cnt",
    "agent_delay_ms",
    "run_seconds",
    "error",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--systems", default="bamboo,polaris")
    parser.add_argument("--workloads", default="ycsb,tpcc")
    parser.add_argument("--levels", default="low,medium,high")
    parser.add_argument("--variants", default="", help="Optional comma-separated named workload variants.")
    parser.add_argument("--client-counts", default="8,16,24,32,40,48")
    parser.add_argument("--agent-ratios", default="1.0,0.8")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run-timeout", type=float, default=0.0, help="Per-row wall-clock timeout in seconds; 0 disables.")
    parser.add_argument("--algorithms", default="", help="Optional comma-separated CC_ALG override.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-script", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    patch_script = args.patch_script or Path(__file__).with_name("patch_dbx1000_agentic.py")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    systems = split_csv(args.systems)
    workloads = split_csv(args.workloads)
    levels = split_csv(args.levels)
    variants = build_variants(args.variants, workloads, levels)
    clients = [int(x) for x in split_csv(args.client_counts)]
    ratios = [float(x) for x in split_csv(args.agent_ratios)]
    alg_override = split_csv(args.algorithms)

    for system in systems:
        meta = SYSTEMS[system]
        work_repo = prepare_repo(root, system, meta, patch_script)
        algorithms = alg_override or meta["algorithms"]
        for variant in variants:
            workload = variant["workload"]
            level = variant["level"]
            level_key = variant["level_key"]
            workload_variant = variant["workload_variant"]
            for repeat in range(int(args.repeats)):
                for client_count in clients:
                    for ratio in ratios:
                        for alg in algorithms:
                            row = run_one(
                                work_repo,
                                system=system,
                                python_cmd=meta["python"],
                                workload=workload,
                                level=level,
                                level_key=level_key,
                                workload_variant=workload_variant,
                                clients=client_count,
                                agent_ratio=ratio,
                                duration=args.duration,
                                warmup=args.warmup,
                                run_timeout=args.run_timeout,
                                repeat=repeat,
                                alg=alg,
                                runner=meta.get("runner", "dbx1000"),
                            )
                            rows.append(row)
                            write_rows(output, rows)
                            if args.smoke:
                                return 0
    return 0


def prepare_repo(root: Path, system: str, meta: dict, patch_script: Path) -> Path:
    src = root / meta["source_dir"]
    dst = root / meta["work_dir"]
    if not src.exists():
        raise SystemExit(f"missing external repo for {system}: {src}")
    if dst.exists():
        shutil.rmtree(str(dst))
    ignore = shutil.ignore_patterns(".git", "outputs", "log", "*.o", "*.d", "rundb", "temp.out")
    shutil.copytree(str(src), str(dst), ignore=ignore)
    (dst / "outputs").mkdir(exist_ok=True)
    (dst / "log").mkdir(exist_ok=True)
    cmd = [sys.executable, str(patch_script), str(dst)]
    if meta.get("runner") == "plor":
        cmd.append("--plor")
    if meta.get("compat"):
        cmd.append("--compat")
    compiler = first_existing(meta.get("compiler_candidates", []))
    if compiler:
        cmd.extend(["--compiler", compiler])
    subprocess.check_call(cmd)
    return dst


def run_one(repo: Path, **kwargs) -> dict:
    if kwargs.get("runner") == "plor":
        return run_one_plor(repo, **kwargs)

    workload = kwargs["workload"]
    level_key = kwargs["level_key"]
    duration = kwargs["duration"]
    config_file = "experiments/default.json" if workload == "ycsb" else "experiments/tpcc.json"
    overrides = common_overrides(**kwargs)
    if workload == "ycsb":
        overrides.update(YCSB_LEVELS[level_key])
        overrides["WORKLOAD"] = "YCSB"
    elif workload == "tpcc":
        overrides.update(TPCC_LEVELS[level_key])
        overrides["WORKLOAD"] = "TPCC"
        overrides["TPCC_USER_ABORT"] = "true"
    else:
        raise ValueError(workload)

    cmd = [kwargs["python_cmd"], "test.py", config_file]
    cmd.extend(f"{key}={value}" for key, value in sorted(overrides.items()))
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo),
        env=run_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    timed_out = False
    try:
        out, _ = proc.communicate(
            timeout=float(kwargs.get("run_timeout", 0.0) or 0.0) or None
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_tree(proc)
        out, _ = proc.communicate()
    elapsed = time.time() - started
    parsed = parse_summary(out)

    row = {
        "system": kwargs["system"],
        "cc": kwargs["alg"],
        "workload": workload,
        "workload_variant": kwargs["workload_variant"],
        "level": kwargs["level"],
        "clients": kwargs["clients"],
        "agent_ratio": kwargs["agent_ratio"],
        "duration_s": duration,
        "warmup_s": kwargs["warmup"],
        "repeat": kwargs["repeat"],
        "status": "ok" if parsed and not timed_out else "error",
        "throughput": parsed.get("throughput", parsed.get("rxn_rate", "")),
        "txn_cnt": parsed.get("txn_cnt", ""),
        "abort_cnt": parsed.get("abort_cnt", ""),
        "user_abort_cnt": parsed.get("user_abort_cnt", ""),
        "agent_txn_cnt": parsed.get("castdas_agent_txn_cnt", ""),
        "agent_abort_cnt": parsed.get("castdas_agent_abort_cnt", ""),
        "background_txn_cnt": parsed.get("castdas_background_txn_cnt", ""),
        "background_abort_cnt": parsed.get("castdas_background_abort_cnt", ""),
        "agent_delay_ms": ns_to_ms(parsed.get("castdas_agent_delay_ns", "")),
        "run_seconds": f"{elapsed:.3f}",
        "error": "" if parsed and not timed_out else ("timeout; " + tail(out) if timed_out else tail(out)),
    }
    log_dir = repo / "outputs" / "castdas_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = (
        f"{kwargs['system']}_{kwargs['alg']}_{kwargs['workload_variant']}_"
        f"c{kwargs['clients']}_a{kwargs['agent_ratio']}_r{kwargs['repeat']}.log"
    )
    (log_dir / log_name).write_text(out)
    return row


def run_one_plor(repo: Path, **kwargs) -> dict:
    """Compile and run Plor, whose native driver is config.h plus rundb."""
    workload = kwargs["workload"]
    level_key = kwargs["level_key"]
    defines = {
        "CORE_CNT": kwargs["clients"],
        "CORO_CNT": 1,
        "CC_ALG": kwargs["alg"],
        "CASTDAS_AGENTIC": "true",
        "CASTDAS_AGENT_RATIO": kwargs["agent_ratio"],
        "CASTDAS_REASONING_SCALE": 1.0,
        "CASTDAS_RUN_SECONDS": max(1, int(round(kwargs["duration"]))),
    }
    if workload == "ycsb":
        defines.update(plor_defines(YCSB_LEVELS[level_key]))
        defines["WORKLOAD"] = "YCSB"
    elif workload == "tpcc":
        defines.update(plor_defines(TPCC_LEVELS[level_key]))
        defines["WORKLOAD"] = "TPCC"
    else:
        raise ValueError(workload)
    update_c_defines(repo / "config.h", defines)

    started = time.time()
    build = subprocess.run(
        ["make", "-j"],
        cwd=str(repo),
        env=run_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    build_output = build.stdout
    summary_path = repo / "outputs" / "castdas_summary.log"
    run_output = ""
    timed_out = False
    if build.returncode == 0:
        cmd = ["./rundb", f"-t{kwargs['clients']}", "-o", str(summary_path)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=run_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        try:
            run_output, _ = proc.communicate(
                timeout=float(kwargs.get("run_timeout", 0.0) or 0.0) or None
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(proc)
            run_output, _ = proc.communicate()
    elapsed = time.time() - started
    summary = summary_path.read_text() if summary_path.exists() else ""
    output = build_output + "\n" + run_output + "\n" + summary
    parsed = parse_summary(output)
    ok = build.returncode == 0 and bool(parsed) and not timed_out
    row = {
        "system": kwargs["system"],
        "cc": kwargs["alg"],
        "workload": workload,
        "workload_variant": kwargs["workload_variant"],
        "level": kwargs["level"],
        "clients": kwargs["clients"],
        "agent_ratio": kwargs["agent_ratio"],
        "duration_s": kwargs["duration"],
        "warmup_s": kwargs["warmup"],
        "repeat": kwargs["repeat"],
        "status": "ok" if ok else "error",
        "throughput": parsed.get("rxn_rate", ""),
        "txn_cnt": parsed.get("txn_cnt", ""),
        "abort_cnt": parsed.get("abort_cnt", ""),
        "user_abort_cnt": "",
        "agent_txn_cnt": parsed.get("castdas_agent_txn_cnt", ""),
        "agent_abort_cnt": parsed.get("castdas_agent_abort_cnt", ""),
        "background_txn_cnt": parsed.get("castdas_background_txn_cnt", ""),
        "background_abort_cnt": parsed.get("castdas_background_abort_cnt", ""),
        "agent_delay_ms": ns_to_ms(parsed.get("castdas_agent_delay_ns", "")),
        "run_seconds": f"{elapsed:.3f}",
        "error": "" if ok else ("timeout; " + tail(output) if timed_out else tail(output)),
    }
    log_dir = repo / "outputs" / "castdas_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = (
        f"{kwargs['system']}_{kwargs['alg']}_{kwargs['workload_variant']}_"
        f"c{kwargs['clients']}_a{kwargs['agent_ratio']}_r{kwargs['repeat']}.log"
    )
    (log_dir / log_name).write_text(output)
    return row


def plor_defines(values: dict) -> dict:
    """Keep only config.h knobs Plor exposes as compile-time constants."""
    supported = {"SYNTH_TABLE_SIZE", "REQ_PER_QUERY", "READ_PERC", "ZIPF_THETA", "NUM_WH", "PERC_PAYMENT"}
    return {key: value for key, value in values.items() if key in supported}


def update_c_defines(path: Path, values: dict) -> None:
    text = path.read_text()
    for key, value in values.items():
        pattern = re.compile(rf"^(#define\s+{re.escape(key)}\s+).*$", re.MULTILINE)
        text, count = pattern.subn(rf"\g<1>{value}", text, count=1)
        if count != 1:
            raise SystemExit(f"cannot set {key} in {path}")
    path.write_text(text)


def common_overrides(**kwargs) -> dict:
    return {
        "THREAD_CNT": kwargs["clients"],
        "CC_ALG": kwargs["alg"],
        "CASTDAS_AGENTIC": "true",
        "CASTDAS_AGENT_RATIO": kwargs["agent_ratio"],
        "CASTDAS_REASONING_SCALE": 1.0,
        "TERMINATE_BY_COUNT": "false",
        "MAX_RUNTIME": max(1, int(round(kwargs["duration"]))),
        "MAX_TXN_PER_PART": 1000000,
        "WARMUP": max(0, int(round(kwargs["warmup"]))),
        "UNSET_NUMA": "true",
        "OUTPUT_TO_FILE": "false",
        "NDEBUG": "true",
    }


def parse_summary(output: str) -> dict:
    result = {}
    for line in output.splitlines():
        match = SUMMARY_RE.search(line.strip())
        if not match:
            continue
        body = match.group("body")
        for token in body.split(","):
            token = token.strip()
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def write_rows(path: Path, rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def split_csv(value: str) -> list:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_variants(variant_arg: str, workloads: list, levels: list) -> list:
    requested = split_csv(variant_arg)
    if requested:
        variants = []
        for name in requested:
            if name not in VARIANTS:
                raise SystemExit(f"unsupported workload variant: {name}")
            workload, level, level_key = VARIANTS[name]
            variants.append(
                {
                    "workload": workload,
                    "level": level,
                    "level_key": level_key,
                    "workload_variant": name,
                }
            )
        return variants
    variants = []
    for workload in workloads:
        level_map = YCSB_LEVELS if workload == "ycsb" else TPCC_LEVELS
        for level in levels:
            if level not in level_map:
                continue
            variants.append(
                {
                    "workload": workload,
                    "level": level,
                    "level_key": level,
                    "workload_variant": f"{workload}_{level}",
                }
            )
    return variants


def first_existing(paths: list) -> str:
    for path in paths:
        if Path(path).exists():
            return str(path)
    return ""


def run_env() -> dict:
    env = dict(os.environ)
    conda_lib = "/home/chenht/miniconda3-castdas/lib"
    if Path(conda_lib).exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = conda_lib if not existing else conda_lib + ":" + existing
    return env


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return


def tail(text: str, limit: int = 800) -> str:
    clean = " | ".join(line.strip() for line in text.splitlines()[-20:])
    return clean[-limit:]


def ns_to_ms(value: str) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value) / 1000000.0:.3f}"
    except ValueError:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())

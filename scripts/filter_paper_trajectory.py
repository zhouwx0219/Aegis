#!/usr/bin/env python3
"""Filter a paper-ATCC trajectory artifact by its training source identifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-contains", required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    marker = str(args.source_contains)
    transitions = [
        row
        for row in payload.get("transitions", [])
        if marker in str(row.get("source_id", ""))
    ]
    source_runs = [
        row
        for row in payload.get("source_runs", [])
        if marker in str(row.get("run_id", row.get("config_id", "")))
    ]
    result = {
        **{key: value for key, value in payload.items() if key not in {"transitions", "source_runs"}},
        "artifact_type": "cast-das-paper-atcc-filtered-trajectories",
        "source_filter": marker,
        "source_runs": source_runs,
        "transitions": transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{args.output}: transitions={len(transitions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

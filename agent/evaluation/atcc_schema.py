"""Shared schema metadata for phase-aware ATCC artifacts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


ATCC_STATE_SCHEMA: Dict[str, Any] = {
    "name": "phase-aware-atcc-object-class-state",
    "version": 2,
    "dimensions": (
        "workload",
        "task",
        "class",
        "phase",
        "reads",
        "writes",
        "hotR",
        "hotW",
        "retry",
        "interval",
        "priority",
        "globalObs",
        "globalAbort",
        "globalLockWait",
        "globalLatency",
        "intent",
    ),
    "notes": (
        "State keys include object class so TPCC NewOrder roles such as "
        "next_order_id, stock quantity, stock ytd, and orders append train "
        "separate Q-table rows."
    ),
}


def atcc_state_schema() -> Dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in ATCC_STATE_SCHEMA.items()
    }


def atcc_artifact_schema_status(
    artifact: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected = atcc_state_schema()
    if artifact is None:
        return {
            "loaded": False,
            "expected_state_schema": expected,
            "compatible": True,
        }

    schema = dict(artifact.get("atcc_state_schema", {}) or {})
    version = schema.get("version")
    dimensions = tuple(str(value) for value in schema.get("dimensions", ()) or ())
    expected_dimensions = tuple(str(value) for value in expected["dimensions"])
    missing_dimensions = [
        dimension
        for dimension in expected_dimensions
        if dimension not in dimensions
    ]
    compatible = (
        int(version or 0) == int(expected["version"])
        and not missing_dimensions
    )
    return {
        "loaded": True,
        "artifact_type": str(artifact.get("artifact_type", "")),
        "artifact_version": int(artifact.get("artifact_version", 0) or 0),
        "state_schema_name": str(schema.get("name", "")),
        "state_schema_version": int(version or 0),
        "expected_state_schema_version": int(expected["version"]),
        "state_schema_dimensions": list(dimensions),
        "expected_state_schema_dimensions": list(expected_dimensions),
        "missing_expected_dimensions": missing_dimensions,
        "compatible": bool(compatible),
    }

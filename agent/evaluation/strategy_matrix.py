"""Shared strategy labels and experiment matrix helpers.

This module keeps experiment naming separate from runner implementation.  The
runtime modules still own the actual CC and ATCC behavior; evaluation code uses
these helpers only to build reproducible matrices and reports.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Sequence, Tuple


TRADITIONAL_CC_STRATEGIES: Tuple[str, ...] = (
    "occ",
    "2pl-nowait",
    "2pl-wait-die",
    "mvcc-full",
    "silo-full",
    "tictoc-full",
)

ATCC_STRATEGIES: Tuple[str, ...] = (
    "adaptive-op-strict",
    "transaction-atcc-strict",
)

ATCC_FAMILY_STRATEGIES: Tuple[str, ...] = (
    "adaptive-hybrid",
)

ABLATION_VARIANTS: Tuple[str, ...] = (
    "op-static",
    "op-static-priority",
    "op-dynamic",
    "op-dynamic-priority",
    "tx-static",
    "tx-static-priority",
    "tx-dynamic",
    "tx-dynamic-priority",
)

DEFAULT_ABLATION_BASELINES: Tuple[str, ...] = ("occ", "mvcc-full", "tictoc-full")

STATIC_PRESETS: Tuple[str, ...] = ("conservative", "threshold32", "naive")
STATIC_OPERATION_WIDE_OVERWRITE_THRESHOLD = 32
STATIC_OPERATION_THRESHOLD32_WIDE_OVERWRITE_THRESHOLD = 12
STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD = 64
STATIC_TRANSACTION_WIDE_WRITE_THRESHOLD = 32
STATIC_OPERATION_NAIVE_WIDE_WRITE_THRESHOLD = 16
STATIC_TRANSACTION_NAIVE_WIDE_WRITE_THRESHOLD = 16


@dataclasses.dataclass(frozen=True)
class AblationVariantSpec:
    name: str
    scope: str
    dynamic: bool
    priority: bool

    @property
    def strategy(self) -> str:
        return "adaptive-op-strict" if self.scope == "op" else "transaction-atcc-strict"

    @property
    def mechanism(self) -> str:
        return "dynamic" if self.dynamic else "static"


def normalize_static_preset(value: str) -> str:
    preset = str(value or "conservative").strip().lower()
    if preset not in STATIC_PRESETS:
        raise ValueError(f"unsupported static preset: {value}")
    return preset


def static_operation_wide_overwrite_threshold(static_preset: str) -> int:
    preset = normalize_static_preset(static_preset)
    if preset == "naive":
        return STATIC_OPERATION_NAIVE_WIDE_WRITE_THRESHOLD
    if preset == "threshold32":
        return STATIC_OPERATION_THRESHOLD32_WIDE_OVERWRITE_THRESHOLD
    return STATIC_OPERATION_WIDE_OVERWRITE_THRESHOLD


def static_transaction_wide_write_threshold(static_preset: str) -> int:
    preset = normalize_static_preset(static_preset)
    if preset == "naive":
        return STATIC_TRANSACTION_NAIVE_WIDE_WRITE_THRESHOLD
    if preset == "threshold32":
        return STATIC_TRANSACTION_WIDE_WRITE_THRESHOLD
    return STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD


def ablation_variant_spec(name: str) -> AblationVariantSpec:
    normalized = str(name).strip().lower()
    if normalized not in ABLATION_VARIANTS:
        raise ValueError(f"unsupported ablation variant: {name}")
    scope, mechanism, *rest = normalized.split("-")
    return AblationVariantSpec(
        name=normalized,
        scope=scope,
        dynamic=mechanism == "dynamic",
        priority=bool(rest and rest[0] == "priority"),
    )


def ablation_variant_metadata(variant: str, strategy: str) -> Dict[str, Any]:
    if str(variant) == "baseline":
        return {
            "ablation_scope": "baseline",
            "ablation_mechanism": "baseline",
            "ablation_priority": False,
            "policy_variant": "baseline:" + str(strategy),
        }
    spec = ablation_variant_spec(str(variant))
    return {
        "ablation_scope": spec.scope,
        "ablation_mechanism": spec.mechanism,
        "ablation_priority": spec.priority,
    }


def split_csv(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def select_named_values(value: str, all_values: Sequence[str]) -> Tuple[str, ...]:
    normalized = str(value).strip().lower()
    if normalized == "all":
        return tuple(all_values)
    return (normalized,)


def select_ablation_variants(value: str) -> Tuple[str, ...]:
    selected = ABLATION_VARIANTS if str(value).strip().lower() == "all" else split_csv(value)
    for name in selected:
        ablation_variant_spec(name)
    return tuple(str(name).strip().lower() for name in selected)


def priority_cap_arg(value: int) -> Optional[int]:
    cap = int(value)
    return None if cap < 0 else cap


def workload_kind_from_name(workload_name: str) -> str:
    text = str(workload_name).lower()
    if "ycsb" in text:
        return "ycsb"
    if "tpcc" in text:
        return "tpcc"
    return text


def profile_name_from_workload(workload_name: str) -> str:
    text = str(workload_name).lower()
    for profile in ("low", "medium", "high"):
        if profile in text:
            return profile
    return ""


def bucket_count(value: int) -> str:
    number = max(0, int(value))
    if number == 0:
        return "0"
    if number == 1:
        return "1"
    if number <= 3:
        return "2-3"
    if number <= 7:
        return "4-7"
    if number <= 15:
        return "8-15"
    return "16+"


def bucket_latency_s(value: float) -> str:
    ms = max(0.0, float(value)) * 1000.0
    if ms <= 0.0:
        return "0ms"
    if ms <= 10.0:
        return "<=10ms"
    if ms <= 50.0:
        return "<=50ms"
    if ms <= 100.0:
        return "<=100ms"
    return ">100ms"


def coarse_interval_s(value: float) -> str:
    ms = max(0.0, float(value)) * 1000.0
    if ms <= 0.0:
        return "0ms"
    if ms <= 50.0:
        return "<=50ms"
    return ">50ms"

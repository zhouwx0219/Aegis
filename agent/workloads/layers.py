"""Named workload layers used by ASTRA experiments.

The layers keep the system boundary explicit:

* native: DBx1000's own benchmark executable and native CC algorithms.
* faithful: agent runtime over versioned KV, with one candidate and DBx1000-like
  request families for apples-to-apples comparison.
* semantic: full data-agent workload with K candidates and semantic intents.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Tuple

from .base import WorkloadManifest
from .tpcc import TPCCAgentWorkload, TPCCConfig, TPCCFaithfulAgentWorkload
from .ycsb import YCSBAgentWorkload, YCSBConfig, YCSBFaithfulAgentWorkload


WORKLOAD_LAYERS: Tuple[str, ...] = ("native", "faithful", "semantic")


def build_agent_workload(
    family: str,
    layer: str = "semantic",
    *,
    ycsb_config: YCSBConfig | None = None,
    tpcc_config: TPCCConfig | None = None,
) -> Any:
    """Return an agent-executable workload for the requested non-native layer."""

    family = family.strip().lower()
    layer = layer.strip().lower()
    if layer == "native":
        raise ValueError("native DBx1000 workloads run through agent.evaluation.dbx1000_native")
    if family == "ycsb":
        config = ycsb_config or YCSBConfig()
        if layer == "faithful":
            return YCSBFaithfulAgentWorkload(config)
        if layer == "semantic":
            return YCSBAgentWorkload(config)
    if family in {"tpcc", "tpc-c"}:
        config = tpcc_config or TPCCConfig()
        if layer == "faithful":
            return TPCCFaithfulAgentWorkload(config)
        if layer == "semantic":
            return TPCCAgentWorkload(config)
    raise ValueError(f"unsupported workload layer: family={family}, layer={layer}")


def native_workload_manifest(
    family: str, config: Mapping[str, Any] | None = None
) -> WorkloadManifest:
    """Describe the DBx1000-native workload layer for experiment manifests."""

    family = family.strip().lower()
    config_dict: Dict[str, Any] = dict(config or {})
    if family == "ycsb":
        return WorkloadManifest(
            name="dbx1000-ycsb-native",
            benchmark_family="YCSB",
            source_system="DBx1000-native",
            source_files=(
                "third_party/dbx1000/benchmarks/ycsb.h",
                "third_party/dbx1000/benchmarks/ycsb_wl.cpp",
                "third_party/dbx1000/benchmarks/ycsb_txn.cpp",
                "third_party/dbx1000/benchmarks/ycsb_query.cpp",
                "third_party/dbx1000/benchmarks/YCSB_schema.txt",
            ),
            preserved_semantics=(
                "DBx1000 native YCSB query generator",
                "DBx1000 native transaction execution path",
                "DBx1000 native compile-time CC_ALG selection",
            ),
            agent_adaptations=(),
            workload_layer="native",
            canonical_name="dbx1000-ycsb-native",
            config=config_dict,
        )
    if family in {"tpcc", "tpc-c"}:
        return WorkloadManifest(
            name="dbx1000-tpcc-native",
            benchmark_family="TPC-C",
            source_system="DBx1000-native",
            source_files=(
                "third_party/dbx1000/benchmarks/tpcc.h",
                "third_party/dbx1000/benchmarks/tpcc_wl.cpp",
                "third_party/dbx1000/benchmarks/tpcc_txn.cpp",
                "third_party/dbx1000/benchmarks/tpcc_query.cpp",
                "third_party/dbx1000/benchmarks/TPCC_full_schema.txt",
            ),
            preserved_semantics=(
                "DBx1000 native TPC-C Payment/NewOrder benchmark surface",
                "DBx1000 native transaction execution path",
                "DBx1000 native compile-time CC_ALG selection",
            ),
            agent_adaptations=(),
            workload_layer="native",
            canonical_name="dbx1000-tpcc-native",
            config=config_dict,
        )
    raise ValueError(f"unsupported native workload family: {family}")


def layer_summary() -> Dict[str, Dict[str, str]]:
    return {
        "native": {
            "executor": "DBx1000 rundb",
            "cc_owner": "DBx1000 native CC implementation",
            "purpose": "authoritative DBx1000 baseline",
        },
        "faithful": {
            "executor": "ASTRA agent runtime",
            "cc_owner": "agent-side CC over versioned KV",
            "purpose": "DBx1000-derived request shape without K-candidate semantics",
        },
        "semantic": {
            "executor": "ASTRA agent runtime",
            "cc_owner": "agent-side semantic/adaptive CC over versioned KV",
            "purpose": "full data-agent workload with K candidates and intents",
        },
    }


def config_to_dict(config: Any) -> Dict[str, Any]:
    return dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)

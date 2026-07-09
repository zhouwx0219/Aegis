"""Workload builders used by evaluation runners.

The workload modules own the semantic task definitions.  This factory only
centralizes profile-to-config mappings so runners do not duplicate them.
"""

from __future__ import annotations

from typing import Dict, Tuple

from agent.workloads import AgentWorkload, TPCCConfig, YCSBConfig, build_agent_workload


def build_profile_workload(workload_kind: str, profile_name: str) -> AgentWorkload:
    workload = str(workload_kind).strip().lower()
    profile = str(profile_name).strip().lower()
    if workload == "ycsb":
        configs = ycsb_profile_configs()
        if profile not in configs:
            raise ValueError(f"unsupported YCSB profile: {profile_name}")
        return build_agent_workload("ycsb", "semantic", ycsb_config=configs[profile])
    if workload == "tpcc":
        configs = tpcc_profile_configs()
        if profile not in configs:
            raise ValueError(f"unsupported TPCC profile: {profile_name}")
        return build_agent_workload("tpcc", "semantic", tpcc_config=configs[profile])
    raise ValueError(f"unsupported workload kind: {workload_kind}")


def ycsb_profile_configs() -> Dict[str, YCSBConfig]:
    return {
        "low": YCSBConfig(
            record_count=512,
            field_count=10,
            requests_per_task=10,
            candidates_per_task=3,
            read_weight=0.95,
            update_weight=0.05,
            zipf_theta=0.0,
            hotspot_fraction=0.0,
            hotspot_access_probability=0.0,
        ),
        "medium": YCSBConfig(
            record_count=128,
            field_count=10,
            requests_per_task=10,
            candidates_per_task=3,
            read_weight=0.90,
            update_weight=0.10,
            zipf_theta=0.7,
            hotspot_fraction=0.10,
            hotspot_access_probability=0.50,
        ),
        "high": YCSBConfig(
            record_count=64,
            field_count=10,
            requests_per_task=10,
            candidates_per_task=3,
            read_weight=0.50,
            update_weight=0.50,
            zipf_theta=0.99,
            hotspot_fraction=0.10,
            hotspot_access_probability=0.75,
        ),
    }


def tpcc_profile_configs() -> Dict[str, TPCCConfig]:
    new_order: Tuple[Tuple[str, float], ...] = (("new_order", 1.0),)
    return {
        "low": TPCCConfig(
            warehouses=8,
            districts_per_warehouse=5,
            customers_per_district=100,
            items=500,
            order_lines=5,
            transaction_mix=new_order,
        ),
        "medium": TPCCConfig(
            warehouses=2,
            districts_per_warehouse=3,
            customers_per_district=60,
            items=200,
            order_lines=8,
            transaction_mix=new_order,
        ),
        "high": TPCCConfig(
            warehouses=1,
            districts_per_warehouse=2,
            customers_per_district=40,
            items=100,
            order_lines=10,
            transaction_mix=new_order,
        ),
    }

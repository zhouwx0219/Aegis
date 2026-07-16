"""Workloads for CAST-DAS experiments."""

from .base import (
    AgentOperation,
    AgentTask,
    AgentWorkload,
    ObjectSpec,
    apply_operation,
    execute_task,
    populate_task_transaction,
    prepare_task_transaction,
    register_workload,
)
from .tpcc import TPCCConfig, TPCCWorkload, tpcc_config, with_warehouses
from .ycsb import YCSBConfig, YCSBWorkload, ycsb_config


def build_workload(
    family: str,
    level: str,
    profile: str = "small",
    *,
    ycsb_zipf_theta: float | None = None,
    tpcc_warehouses: int | None = None,
) -> AgentWorkload:
    family = str(family).strip().lower()
    profile = str(profile).strip().lower() or "small"
    if family == "ycsb":
        return YCSBWorkload(ycsb_config(level, profile, zipf_theta=ycsb_zipf_theta))
    if family in {"tpcc", "tpc-c"}:
        return TPCCWorkload(with_warehouses(tpcc_config(level, profile), tpcc_warehouses))
    raise ValueError(f"unsupported workload: {family}")


__all__ = [
    "AgentOperation",
    "AgentTask",
    "AgentWorkload",
    "ObjectSpec",
    "apply_operation",
    "execute_task",
    "populate_task_transaction",
    "prepare_task_transaction",
    "register_workload",
    "build_workload",
    "TPCCConfig",
    "TPCCWorkload",
    "tpcc_config",
    "with_warehouses",
    "YCSBConfig",
    "YCSBWorkload",
    "ycsb_config",
]

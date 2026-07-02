"""ATCC family-level strategy selection for agent workloads.

The operation-level ATCC table decides which objects need pessimistic
protection.  This module sits one level above it and selects the execution
family for a whole agent task so read-heavy profiles can avoid ATCC fixed costs
while write-hotspot profiles keep the stronger operation-level protection.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence


@dataclasses.dataclass(frozen=True)
class ATCCFamilyDecision:
    requested_strategy: str
    selected_strategy: str
    rule: str
    signals: Mapping[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_strategy": self.requested_strategy,
            "selected_strategy": self.selected_strategy,
            "rule": self.rule,
            "signals": dict(self.signals),
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class ATCCFamilyPolicyTable:
    """Explainable ATCC policy table for choosing a CC family per agent task."""

    read_heavy_write_ratio: float = 0.25
    hot_write_ratio_threshold: float = 0.30
    hotspot_probability_threshold: float = 0.70
    read_heavy_strategy: str = "tictoc-full"
    cold_read_heavy_strategy: str = "tictoc-full"
    adaptive_cold_mvcc_write_ratio_threshold: float = 0.05
    adaptive_hotspot_mvcc_write_ratio_threshold: float = 0.11
    cold_hotspot_probability_threshold: float = 0.01
    hot_write_strategy: str = "adaptive-op-strict"
    fallback_strategy: str = "tictoc-full"
    tpcc_low_contention_strategy: str = ""
    tpcc_low_contention_min_distinct_order_counters: int = 0
    tpcc_window_distinct_order_counters: int = 0
    requested_strategy: str = "adaptive-hybrid"

    @classmethod
    def default(cls) -> "ATCCFamilyPolicyTable":
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ATCCFamilyPolicyTable":
        allowed_strategies = {
            "mvcc-full",
            "tictoc-full",
            "adaptive-op-strict",
            "transaction-atcc-strict",
            "silo-full",
            "occ",
        }
        adaptive_read_heavy_strategies = {"adaptive-mvcc-tictoc"}

        def strategy(
            name: str,
            fallback: str,
            *,
            allow_adaptive_read_heavy: bool = False,
        ) -> str:
            value = str(data.get(name, fallback)).strip().lower()
            allowed = set(allowed_strategies)
            if allow_adaptive_read_heavy:
                allowed.update(adaptive_read_heavy_strategies)
            if value not in allowed:
                raise ValueError(f"unsupported family strategy for {name}: {value}")
            return value

        return cls(
            read_heavy_write_ratio=float(
                data.get("read_heavy_write_ratio", cls.read_heavy_write_ratio)
            ),
            hot_write_ratio_threshold=float(
                data.get("hot_write_ratio_threshold", cls.hot_write_ratio_threshold)
            ),
            hotspot_probability_threshold=float(
                data.get(
                    "hotspot_probability_threshold",
                    cls.hotspot_probability_threshold,
                )
            ),
            read_heavy_strategy=strategy(
                "read_heavy_strategy",
                cls.read_heavy_strategy,
                allow_adaptive_read_heavy=True,
            ),
            cold_read_heavy_strategy=strategy(
                "cold_read_heavy_strategy",
                str(data.get("read_heavy_strategy", cls.cold_read_heavy_strategy)),
                allow_adaptive_read_heavy=True,
            ),
            adaptive_cold_mvcc_write_ratio_threshold=float(
                data.get(
                    "adaptive_cold_mvcc_write_ratio_threshold",
                    cls.adaptive_cold_mvcc_write_ratio_threshold,
                )
            ),
            adaptive_hotspot_mvcc_write_ratio_threshold=float(
                data.get(
                    "adaptive_hotspot_mvcc_write_ratio_threshold",
                    cls.adaptive_hotspot_mvcc_write_ratio_threshold,
                )
            ),
            cold_hotspot_probability_threshold=float(
                data.get(
                    "cold_hotspot_probability_threshold",
                    cls.cold_hotspot_probability_threshold,
                )
            ),
            hot_write_strategy=strategy(
                "hot_write_strategy",
                cls.hot_write_strategy,
            ),
            fallback_strategy=strategy("fallback_strategy", cls.fallback_strategy),
            tpcc_low_contention_strategy=(
                strategy(
                    "tpcc_low_contention_strategy",
                    cls.tpcc_low_contention_strategy,
                )
                if str(data.get("tpcc_low_contention_strategy", "")).strip()
                else ""
            ),
            tpcc_low_contention_min_distinct_order_counters=int(
                data.get(
                    "tpcc_low_contention_min_distinct_order_counters",
                    cls.tpcc_low_contention_min_distinct_order_counters,
                )
                or 0
            ),
            requested_strategy=str(
                data.get("requested_strategy", cls.requested_strategy)
            ).strip().lower(),
        )

    def select_task(
        self,
        task: Any,
        *,
        workload_kind: str = "",
    ) -> ATCCFamilyDecision:
        signals = _task_signals(task)
        workload = str(workload_kind or getattr(task, "workload", "")).lower()
        if "ycsb" in workload:
            if (
                signals["write_ratio"] >= self.hot_write_ratio_threshold
                and signals["hotspot_access_probability"]
                >= self.hotspot_probability_threshold
            ):
                return ATCCFamilyDecision(
                    requested_strategy=self.requested_strategy,
                    selected_strategy=self.hot_write_strategy,
                    rule="ycsb-hot-write-atcc",
                    signals=signals,
                    reason=(
                        "YCSB task has enough write or hotspot pressure to "
                        "justify operation-level ATCC protection."
                    ),
                )
            if (
                signals["write_count"] > 0.0
                and signals["hotspot_access_probability"]
                >= max(0.70, self.hotspot_probability_threshold)
                and signals["hot_write_ratio"] > 0.0
            ):
                return ATCCFamilyDecision(
                    requested_strategy=self.requested_strategy,
                    selected_strategy=self.hot_write_strategy,
                    rule="ycsb-high-hotspot-write-atcc",
                    signals=signals,
                    reason=(
                        "YCSB task has a hot write under high hotspot pressure; "
                        "use operation-level ATCC even when this individual task "
                        "looks read-heavy."
                    ),
                )
            if signals["write_ratio"] <= self.read_heavy_write_ratio:
                if (
                    signals["hotspot_access_probability"]
                    <= self.cold_hotspot_probability_threshold
                ):
                    adaptive = self._adaptive_read_heavy_decision(
                        signals,
                        cold=True,
                    )
                    if adaptive is not None:
                        return adaptive
                    return ATCCFamilyDecision(
                        requested_strategy=self.requested_strategy,
                        selected_strategy=self.cold_read_heavy_strategy,
                        rule="ycsb-cold-read-heavy",
                        signals=signals,
                        reason=(
                            "YCSB task is read-heavy without hotspot pressure; "
                            "use the cold-read family learned for low-conflict "
                            "agent workloads."
                        ),
                    )
                adaptive = self._adaptive_read_heavy_decision(signals, cold=False)
                if adaptive is not None:
                    return adaptive
                return ATCCFamilyDecision(
                    requested_strategy=self.requested_strategy,
                    selected_strategy=self.read_heavy_strategy,
                    rule="ycsb-read-heavy-tictoc",
                    signals=signals,
                    reason=(
                        "YCSB task is read-heavy with hotspot pressure below the "
                        "hot-write threshold; use the learned read-heavy family "
                        "to avoid ATCC refresh and prelock fixed costs."
                    ),
                )
        if "tpcc" in workload:
            if (
                self.tpcc_low_contention_strategy
                and self.tpcc_window_distinct_order_counters
                >= self.tpcc_low_contention_min_distinct_order_counters
                > 0
            ):
                return ATCCFamilyDecision(
                    requested_strategy=self.requested_strategy,
                    selected_strategy=self.fallback_strategy,
                    rule="tpcc-low-contention-fast-through",
                    signals=signals,
                    reason=(
                        "TPC-C task window has enough distinct order counters; "
                        "use the configured low-contention fast-through family."
                    ),
                )
            return ATCCFamilyDecision(
                requested_strategy=self.requested_strategy,
                selected_strategy=self.fallback_strategy,
                rule="tpcc-hot-write-atcc",
                signals=signals,
                reason=(
                    "TPC-C task window does not meet the low-contention "
                    "distinct-counter threshold; use the configured hot-write "
                    "family for order-counter and stock-update contention."
                ),
            )
        return ATCCFamilyDecision(
            requested_strategy=self.requested_strategy,
            selected_strategy=self.fallback_strategy,
            rule="fallback-traditional",
            signals=signals,
            reason="No ATCC hot-write rule matched; use the traditional fallback.",
        )

    def _adaptive_read_heavy_decision(
        self,
        signals: Mapping[str, float],
        *,
        cold: bool,
    ) -> ATCCFamilyDecision | None:
        configured = (
            self.cold_read_heavy_strategy if cold else self.read_heavy_strategy
        )
        if configured != "adaptive-mvcc-tictoc":
            return None
        threshold = (
            self.adaptive_cold_mvcc_write_ratio_threshold
            if cold
            else self.adaptive_hotspot_mvcc_write_ratio_threshold
        )
        use_mvcc = signals["write_ratio"] >= threshold
        family = "mvcc-full" if use_mvcc else "tictoc-full"
        rule_prefix = "ycsb-cold-read-heavy" if cold else "ycsb-read-heavy"
        rule_suffix = "mvcc" if use_mvcc else "tictoc"
        pressure = "cold" if cold else "hotspot"
        return ATCCFamilyDecision(
            requested_strategy=self.requested_strategy,
            selected_strategy=family,
            rule=f"{rule_prefix}-adaptive-{rule_suffix}",
            signals=signals,
            reason=(
                f"YCSB {pressure} read-heavy task uses adaptive MVCC/TicToc "
                f"selection: write_ratio={signals['write_ratio']:.3f}, "
                f"mvcc_threshold={threshold:.3f}."
            ),
        )

    def resolve_for_task_window(
        self,
        tasks: Sequence[Any],
        *,
        workload_kind: str = "",
    ) -> "ATCCFamilyPolicyTable":
        workload = str(workload_kind or "").lower()
        if "tpcc" in workload:
            distinct_order_counters = _distinct_tpcc_order_counters(tasks)
            if (
                self.tpcc_low_contention_strategy
                and self.tpcc_low_contention_min_distinct_order_counters > 0
            ):
                if (
                    distinct_order_counters
                    >= self.tpcc_low_contention_min_distinct_order_counters
                ):
                    return dataclasses.replace(
                        self,
                        fallback_strategy=self.tpcc_low_contention_strategy,
                        tpcc_window_distinct_order_counters=distinct_order_counters,
                    )
                return dataclasses.replace(
                    self,
                    fallback_strategy=self.hot_write_strategy,
                    tpcc_window_distinct_order_counters=distinct_order_counters,
                )
            if distinct_order_counters != self.tpcc_window_distinct_order_counters:
                return dataclasses.replace(
                    self,
                    tpcc_window_distinct_order_counters=distinct_order_counters,
                )
            return self
        if "ycsb" not in workload:
            return self
        cold_strategy = self._resolve_adaptive_read_heavy_strategy(
            self.cold_read_heavy_strategy,
            _read_heavy_write_ratios_for_window(self, tasks, cold=True),
            threshold=self.adaptive_cold_mvcc_write_ratio_threshold,
        )
        read_heavy_strategy = self._resolve_adaptive_read_heavy_strategy(
            self.read_heavy_strategy,
            _read_heavy_write_ratios_for_window(self, tasks, cold=False),
            threshold=self.adaptive_hotspot_mvcc_write_ratio_threshold,
        )
        if (
            cold_strategy == self.cold_read_heavy_strategy
            and read_heavy_strategy == self.read_heavy_strategy
        ):
            return self
        return dataclasses.replace(
            self,
            cold_read_heavy_strategy=cold_strategy,
            read_heavy_strategy=read_heavy_strategy,
        )

    @staticmethod
    def _resolve_adaptive_read_heavy_strategy(
        strategy: str,
        write_ratios: Sequence[float],
        *,
        threshold: float,
    ) -> str:
        if strategy != "adaptive-mvcc-tictoc" or not write_ratios:
            return strategy
        average = sum(float(value) for value in write_ratios) / len(write_ratios)
        return "mvcc-full" if average >= threshold else "tictoc-full"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def is_atcc_family_strategy(strategy: str) -> bool:
    return str(strategy or "").strip().lower() in {
        "adaptive-hybrid",
        "atcc-hybrid",
        "adaptive-family",
        "atcc-family",
    }


def _task_signals(task: Any) -> dict[str, float]:
    candidates: Sequence[Any] = tuple(getattr(task, "candidates", ()) or ())
    operations = tuple(
        operation
        for candidate in candidates
        for operation in getattr(candidate, "operations", ()) or ()
    )
    writes = sum(1 for operation in operations if getattr(operation, "kind", "") != "read")
    reads = sum(1 for operation in operations if getattr(operation, "kind", "") == "read")
    total = reads + writes
    context = dict(getattr(task, "context", {}) or {})
    hot_record_count = int(context.get("hot_record_count", 0) or 0)
    hot_operations = sum(
        1
        for operation in operations
        if _is_ycsb_hot_object(str(getattr(operation, "object_id", "")), hot_record_count)
    )
    hot_writes = sum(
        1
        for operation in operations
        if getattr(operation, "kind", "") != "read"
        and _is_ycsb_hot_object(str(getattr(operation, "object_id", "")), hot_record_count)
    )
    return {
        "read_count": float(reads),
        "write_count": float(writes),
        "operation_count": float(total),
        "write_ratio": writes / total if total else 0.0,
        "hot_operation_ratio": hot_operations / total if total else 0.0,
        "hot_write_ratio": hot_writes / writes if writes else 0.0,
        "hotspot_access_probability": float(
            context.get("hotspot_access_probability", 0.0) or 0.0
        ),
        "hotspot_fraction": float(context.get("hotspot_fraction", 0.0) or 0.0),
    }


def _read_heavy_write_ratios_for_window(
    policy: ATCCFamilyPolicyTable,
    tasks: Sequence[Any],
    *,
    cold: bool,
) -> Tuple[float, ...]:
    ratios = []
    for task in tasks:
        signals = _task_signals(task)
        if not _is_ycsb_read_heavy_for_family(policy, signals):
            continue
        is_cold = (
            signals["hotspot_access_probability"]
            <= policy.cold_hotspot_probability_threshold
        )
        if is_cold != cold:
            continue
        ratios.append(float(signals["write_ratio"]))
    return tuple(ratios)


def _distinct_tpcc_order_counters(tasks: Sequence[Any]) -> int:
    counters = set()
    for task in tasks:
        for candidate in getattr(task, "candidates", ()) or ():
            for operation in getattr(candidate, "operations", ()) or ():
                object_id = str(getattr(operation, "object_id", ""))
                if (
                    getattr(operation, "kind", "") != "read"
                    and object_id.startswith("tpcc:district:")
                    and object_id.endswith(":next_order_id")
                ):
                    counters.add(object_id)
    return len(counters)


def _is_ycsb_read_heavy_for_family(
    policy: ATCCFamilyPolicyTable,
    signals: Mapping[str, float],
) -> bool:
    if (
        signals["write_ratio"] >= policy.hot_write_ratio_threshold
        and signals["hotspot_access_probability"]
        >= policy.hotspot_probability_threshold
    ):
        return False
    if (
        signals["write_count"] > 0.0
        and signals["hotspot_access_probability"]
        >= max(0.70, policy.hotspot_probability_threshold)
        and signals["hot_write_ratio"] > 0.0
    ):
        return False
    return signals["write_ratio"] <= policy.read_heavy_write_ratio


def _is_ycsb_hot_object(object_id: str, hot_record_count: int) -> bool:
    if hot_record_count <= 0 or not object_id.startswith("ycsb:record:"):
        return False
    parts = object_id.split(":")
    if len(parts) < 3:
        return False
    try:
        return int(parts[2]) < hot_record_count
    except ValueError:
        return False

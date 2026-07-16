"""ATCC telemetry helpers for cost-aware policy construction."""

from __future__ import annotations

import dataclasses
from typing import Dict

from agent.cc.atcc.policy import ATCCPolicyRow, ATCCPolicyTable


@dataclasses.dataclass
class ATCCStateStats:
    visits: int = 0
    commits: int = 0
    aborts: int = 0

    occ_visits: int = 0
    occ_aborts: int = 0
    protect_visits: int = 0
    protect_aborts: int = 0
    avg_elapsed_ms: float = 0.0
    avg_lock_wait_ms: float = 0.0
    avg_reasoning_delay_ms: float = 0.0
    avg_abort_cost_ms: float = 0.0

    def observe(
        self,
        *,
        action: str,
        committed: bool,
        elapsed_ms: float = 0.0,
        lock_wait_ms: float = 0.0,
        reasoning_delay_ms: float = 0.0,
    ) -> None:
        self.visits += 1
        normalized_action = str(action).strip().lower() or "occ"
        if normalized_action == "protect":
            self.protect_visits += 1
        else:
            self.occ_visits += 1
        if committed:
            self.commits += 1
        else:
            self.aborts += 1
            if normalized_action == "protect":
                self.protect_aborts += 1
            else:
                self.occ_aborts += 1
        self.avg_elapsed_ms = running_average(self.avg_elapsed_ms, self.visits, elapsed_ms)
        self.avg_lock_wait_ms = running_average(self.avg_lock_wait_ms, self.visits, lock_wait_ms)
        self.avg_reasoning_delay_ms = running_average(
            self.avg_reasoning_delay_ms,
            self.visits,
            reasoning_delay_ms,
        )
        if not committed:
            abort_cost_ms = float(elapsed_ms) + float(reasoning_delay_ms)
            self.avg_abort_cost_ms = running_average(
                self.avg_abort_cost_ms,
                self.aborts,
                abort_cost_ms,
            )

    def to_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ATCCTelemetry:
    states: Dict[str, ATCCStateStats] = dataclasses.field(default_factory=dict)

    def observe(
        self,
        state_key: str,
        *,
        action: str = "occ",
        committed: bool,
        elapsed_ms: float = 0.0,
        lock_wait_ms: float = 0.0,
        reasoning_delay_ms: float = 0.0,
    ) -> None:
        self.states.setdefault(str(state_key), ATCCStateStats()).observe(
            action=action,
            committed=committed,
            elapsed_ms=elapsed_ms,
            lock_wait_ms=lock_wait_ms,
            reasoning_delay_ms=reasoning_delay_ms,
        )

    def train_policy(
        self,
        *,
        abort_threshold: float = 0.20,
        min_visits: int = 5,
        protect_cost_threshold_ms: float = 10.0,
    ) -> ATCCPolicyTable:
        policy = ATCCPolicyTable(
            abort_threshold=float(abort_threshold),
            min_visits=int(min_visits),
            protect_cost_threshold_ms=float(protect_cost_threshold_ms),
        )
        for state_key, stats in self.states.items():
            row = policy.rows.setdefault(state_key, ATCCPolicyRow())
            row.visits = stats.visits
            row.commits = stats.commits
            row.aborts = stats.aborts
            row.occ_visits = stats.occ_visits
            row.occ_aborts = stats.occ_aborts
            row.protect_visits = stats.protect_visits
            row.protect_aborts = stats.protect_aborts
            row.avg_elapsed_ms = stats.avg_elapsed_ms
            row.avg_lock_wait_ms = stats.avg_lock_wait_ms
            row.avg_reasoning_delay_ms = stats.avg_reasoning_delay_ms
            row.avg_abort_cost_ms = stats.avg_abort_cost_ms
            policy._refresh_decision(state_key, row)
        return policy


def running_average(current: float, count_after: int, value: float) -> float:
    count = max(1, int(count_after))
    return float(current) + (float(value) - float(current)) / count

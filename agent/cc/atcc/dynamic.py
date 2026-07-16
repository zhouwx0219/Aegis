"""Single dynamic ATCC strategy backed by a policy table."""

from __future__ import annotations

import threading
import zlib
from typing import Any, Optional

from agent.cc.atcc.actions import (
    LOCK_BEFORE_COMMIT,
    LOCK_HOT_BEFORE_COMMIT,
    LOCK_WRITE_SET,
    OCC,
    RESERVE_HOT,
    RESERVE_HOT_RW,
    RESERVE_HOT_RW_K,
    RESERVE_READ_WRITE_SET,
    RETRY_PROTECT,
    WRITE_VALIDATE,
    action_spec,
    normalize_action,
)
from agent.cc.atcc.common import ATCCDecision, decision_from_preplan
from agent.cc.atcc.features import ATCCFeatures, agent_cost_bucket, contention_bucket, extract_features
from agent.cc.atcc.policy import ATCCPolicyTable, ATCCPolicyRow, priority_from_cost
from agent.cc.base import CCPlan, ConcurrencyControl


HIGH_BACKGROUND_WORKERS = 6
LOW_AGENT_RETRY_COUNT = 0
BP_MODE_MIN_WINDOWS = 3
HOT_RW_K_TARGET_LIMIT = 3
BP_QUEUE_PRESSURE_THRESHOLD = 2


class DynamicATCC(ConcurrencyControl):
    name = "dynamic-atcc"
    family = "atcc"
    description = "Dynamic ATCC that chooses OCC or protection from a trained policy table."

    def __init__(
        self,
        *,
        name: str = "dynamic-atcc",
        policy: Optional[ATCCPolicyTable] = None,
        decision_mode: str = "trained",
        priority_enabled: bool = True,
        runtime_guards_enabled: bool = True,
        bp_background_threshold: int = HIGH_BACKGROUND_WORKERS,
        bp_min_windows: int = BP_MODE_MIN_WINDOWS,
        bp_queue_pressure_threshold: int = BP_QUEUE_PRESSURE_THRESHOLD,
        hot_rw_k_target_limit: int = HOT_RW_K_TARGET_LIMIT,
    ):
        self.name = name
        self.policy = policy or ATCCPolicyTable()
        self.decision_mode = str(decision_mode).strip().lower() or "trained"
        if self.decision_mode not in {"trained", "static"}:
            raise ValueError(f"unsupported ATCC decision mode: {decision_mode}")
        self.priority_enabled = bool(priority_enabled)
        self.runtime_guards_enabled = bool(runtime_guards_enabled)
        self.bp_background_threshold = max(0, int(bp_background_threshold))
        self.bp_min_windows = max(1, int(bp_min_windows))
        self.bp_queue_pressure_threshold = max(0, int(bp_queue_pressure_threshold))
        self.hot_rw_k_target_limit = max(1, int(hot_rw_k_target_limit))
        self._bp_mode_lock = threading.Lock()
        self._bp_mode_active = False
        self._bp_mode_windows = 0

    def decide(self, features: ATCCFeatures) -> ATCCDecision:
        target_limit = self.hot_rw_k_target_limit
        state_key = self.state_key_for(features)
        if self.decision_mode == "static":
            policy_action = static_threshold_action(features)
            action, action_source = policy_action, "static-threshold"
        else:
            policy_action = self.policy.action_for(state_key)
            if self.runtime_guards_enabled:
                action, action_source = self._paper_like_action(policy_action, features)
            else:
                action, action_source = policy_action, "policy"
        bp_mode = self._update_bp_mode(features) if self.runtime_guards_enabled else self._inactive_bp_mode(features)
        pre_override_action = normalize_action(action)
        pre_override_action_source = action_source
        if self.runtime_guards_enabled:
            action, action_source, override_reason = self._post_policy_override(
                action,
                action_source,
                features,
                bp_mode=bp_mode,
            )
        else:
            action, action_source, override_reason = normalize_action(action), action_source, ""
        spec = action_spec(action, retry_count=features.retry_count)
        if spec.lock_scope == "hot":
            targets = features.hot_targets
            if should_bound_hot_reservation_targets(features, targets):
                targets = tuple(hot_rw_k_targets(features, limit=hot_reservation_target_limit(features)))
        elif spec.lock_scope == "hot-rw":
            targets = tuple(sorted(set(features.hot_targets) | set(features.hot_read_targets)))
        elif spec.lock_scope == "hot-rw-k":
            targets = hot_rw_k_targets(features, limit=target_limit)
        elif spec.lock_scope == "read-write-set":
            targets = tuple(sorted(set(features.read_targets) | set(features.write_targets)))
        elif spec.lock_scope == "write-set":
            targets = features.write_targets
        else:
            targets = ()
        lock_scope = spec.lock_scope
        lock_phase = spec.lock_phase
        if spec.uses_locking and not targets:
            targets = features.write_targets
            lock_scope = "write-set" if targets else "none"
            lock_phase = spec.lock_phase if targets else "none"
        if self.priority_enabled:
            priority, priority_reason = self._priority_for(features, state_key=state_key)
        else:
            priority, priority_reason = 0, "priority-disabled"
        admission_yield_ms = self.policy.admission_yield_for(state_key)
        if int(features.background_workers) <= 0:
            admission_yield_ms = 0
        if lock_scope == "none" or not targets:
            priority = 0
            priority_reason = "not-locking"
        risk = risk_score(features)
        metadata = features.to_dict()
        metadata.update(
            {
                "risk_score": risk,
                "risk_bucket": risk_bucket(risk),
                "execution_path": execution_path(spec.action, lock_phase),
                "contention_bucket": contention_bucket(features.hot_write_count, features.write_count),
                "hot_read_count": features.hot_read_count,
                "hot_access_count": features.hot_access_count,
                "policy_action": normalize_action(policy_action),
                "pre_override_action": pre_override_action,
                "pre_override_action_source": pre_override_action_source,
                "selected_action_source": action_source,
                "post_policy_override": bool(override_reason),
                "post_policy_override_reason": override_reason,
                "post_policy_override_target_limit": (
                    target_limit if normalize_action(action) == RESERVE_HOT_RW_K else 0
                ),
                "background_pressure_high": bool(bp_mode["pressure_high"]),
                "background_pressure_background_high": bool(bp_mode["background_high"]),
                "background_pressure_queue_high": bool(bp_mode["queue_high"]),
                "background_pressure_threshold": self.bp_background_threshold,
                "background_pressure_queue_threshold": self.bp_queue_pressure_threshold,
                "reservation_queue_pressure": int(bp_mode["queue_pressure"]),
                "reservation_convoy_active": bool(bp_mode["convoy_high"]),
                "reservation_convoy_queue_target_count": int(bp_mode["convoy_queue_target_count"]),
                "reservation_convoy_front_waiter_count": int(bp_mode["convoy_front_waiter_count"]),
                "reservation_convoy_pressure": int(bp_mode["convoy_pressure"]),
                "bp_pressure_gate_high": bool(bp_mode["pressure_gate_high"]),
                "bp_mode_active": bool(bp_mode["active"]),
                "bp_mode_entered": bool(bp_mode["entered"]),
                "bp_mode_exited_after_decision": bool(bp_mode["exited_after_decision"]),
                "bp_mode_windows": int(bp_mode["windows"]),
                "bp_mode_min_windows": self.bp_min_windows,
                "priority_reason": priority_reason,
                "admission_yield_ms": int(admission_yield_ms),
                "decision_mode": self.decision_mode,
                "priority_enabled": self.priority_enabled,
                "runtime_guards_enabled": self.runtime_guards_enabled,
            }
        )
        return ATCCDecision(
            action=spec.action,
            targets=targets,
            priority=priority,
            state_key=state_key,
            reason=decision_reason(spec.action, action_source),
            lock_scope=lock_scope,
            lock_phase=lock_phase,
            metadata=metadata,
        )

    def state_key_for(self, features: ATCCFeatures) -> str:
        return features.state_key

    def _post_policy_override(
        self,
        action: str,
        action_source: str,
        features: ATCCFeatures,
        *,
        bp_mode: dict[str, object],
    ) -> tuple[str, str, str]:
        normalized = normalize_action(action)
        available = {normalize_action(item) for item in self.policy.trainable_actions}
        if (
            normalized == LOCK_BEFORE_COMMIT
            and not self.policy.training
            and bool(bp_mode["active"])
            and LOCK_HOT_BEFORE_COMMIT in available
        ):
            reason = (
                "bp_mode_active_before_commit_scope;"
                f"background_workers={int(features.background_workers)};"
                f"queue_pressure={int(bp_mode['queue_pressure'])};"
                f"convoy_pressure={int(bp_mode['convoy_pressure'])}"
            )
            return (
                LOCK_HOT_BEFORE_COMMIT,
                f"{action_source}+post-policy-background-pressure",
                reason,
            )
        if normalized != RESERVE_READ_WRITE_SET:
            return normalized, action_source, ""
        if self.policy.training:
            return normalized, action_source, ""
        if normalize_action(RESERVE_HOT_RW) not in available:
            return normalized, action_source, ""
        if bool(bp_mode["active"]):
            reason = (
                "bp_mode_active;"
                f"background_workers={int(features.background_workers)};"
                f"queue_pressure={int(bp_mode['queue_pressure'])};"
                f"convoy_pressure={int(bp_mode['convoy_pressure'])};"
                f"k={self.hot_rw_k_target_limit};"
                f"windows={int(bp_mode['windows'])}"
            )
            return (
                RESERVE_HOT_RW_K,
                f"{action_source}+post-policy-background-pressure",
                reason,
            )
        return normalized, action_source, ""

    def _update_bp_mode(self, features: ATCCFeatures) -> dict[str, object]:
        background_workers = int(features.background_workers)
        queue_pressure = int(features.reservation_queue_pressure)
        background_high = background_workers >= self.bp_background_threshold
        queue_high = (
            background_workers > 0
            and self.bp_queue_pressure_threshold > 0
            and queue_pressure >= self.bp_queue_pressure_threshold
        )
        full_set_target_count = len(set(features.read_targets) | set(features.write_targets))
        convoy_high = (
            bool(features.reservation_convoy_active)
            and full_set_target_count > self.hot_rw_k_target_limit
        )
        pressure_gate_high = background_high or queue_high
        pressure_high = convoy_high and pressure_gate_high
        retry_low = int(features.retry_count) <= LOW_AGENT_RETRY_COUNT
        with self._bp_mode_lock:
            entered = False
            exited_after_decision = False
            if not self._bp_mode_active and pressure_high and retry_low:
                self._bp_mode_active = True
                self._bp_mode_windows = 0
                entered = True

            active_for_decision = bool(self._bp_mode_active)
            if active_for_decision:
                self._bp_mode_windows += 1
                windows = int(self._bp_mode_windows)
                if not pressure_high and windows >= self.bp_min_windows:
                    self._bp_mode_active = False
                    self._bp_mode_windows = 0
                    exited_after_decision = True
            else:
                windows = 0

        return {
            "active": active_for_decision,
            "entered": entered,
            "exited_after_decision": exited_after_decision,
            "windows": windows,
            "pressure_high": pressure_high,
            "background_high": background_high,
            "queue_high": queue_high,
            "queue_pressure": queue_pressure,
            "convoy_high": convoy_high,
            "convoy_queue_target_count": int(features.reservation_convoy_queue_target_count),
            "convoy_front_waiter_count": int(features.reservation_convoy_front_waiter_count),
            "convoy_pressure": int(features.reservation_convoy_pressure),
            "pressure_gate_high": pressure_gate_high,
            "retry_low": retry_low,
        }

    def _inactive_bp_mode(self, features: ATCCFeatures) -> dict[str, object]:
        return {
            "active": False,
            "entered": False,
            "exited_after_decision": False,
            "windows": 0,
            "pressure_high": False,
            "background_high": False,
            "queue_high": False,
            "queue_pressure": int(features.reservation_queue_pressure),
            "convoy_high": False,
            "convoy_queue_target_count": int(features.reservation_convoy_queue_target_count),
            "convoy_front_waiter_count": int(features.reservation_convoy_front_waiter_count),
            "convoy_pressure": int(features.reservation_convoy_pressure),
            "pressure_gate_high": False,
            "retry_low": int(features.retry_count) <= LOW_AGENT_RETRY_COUNT,
        }

    def plan(self, txn: Any) -> CCPlan:
        decision = decision_from_preplan(txn) or self.decide(extract_features(txn))
        spec = action_spec(decision.action, retry_count=int(getattr(txn, "metadata", {}).get("retry_count", 0) or 0))
        return CCPlan(
            strategy=self.name,
            family=self.family,
            lock_targets=decision.targets,
            validate_reads=bool(spec.validate_reads),
            validate_writes=bool(spec.validate_writes),
            metadata={
                "lock_table": "exclusive",
                "wait": True,
                "priority": decision.priority,
                "atcc_action": decision.action,
                "atcc_state_key": decision.state_key,
                "atcc_reason": decision.reason,
                "atcc_lock_scope": decision.lock_scope,
                "atcc_lock_phase": decision.lock_phase,
                "atcc_validate_reads": bool(spec.validate_reads),
                "atcc_validate_writes": bool(spec.validate_writes),
                "atcc_features": dict(decision.metadata),
            },
        )

    def observe(self, plan: Any, result: Any, txn: Any = None) -> None:
        if self.decision_mode == "static":
            return
        txn_metadata = dict(getattr(txn, "metadata", {}) or {})
        state_key = str(getattr(plan, "metadata", {}).get("atcc_state_key", "") or "")
        if not state_key:
            return
        plan_metadata = dict(getattr(plan, "metadata", {}) or {})
        agentic = dict(txn_metadata.get("agentic", {}) or {})
        feature_metadata = dict(plan_metadata.get("atcc_features", {}) or {})
        atcc_runtime = dict(txn_metadata.get("atcc_runtime", {}) or {})
        reasoning_delay_ms = float(
            feature_metadata.get(
                "reasoning_delay_ms",
                agentic.get("reasoning_delay_ms", 0.0),
            )
            or 0.0
        )
        committed = bool(getattr(result, "committed", False))
        skipped_reasoning_ms = float(
            plan_metadata.get(
                "atcc_skipped_reasoning_ms",
                atcc_runtime.get("skipped_reasoning_ms", 0.0),
            )
            or 0.0
        )
        self.policy.observe(
            state_key,
            action=str(getattr(plan, "metadata", {}).get("atcc_action", "occ") or "occ"),
            committed=committed,
            elapsed_ms=(
                float(getattr(result, "elapsed_s", 0.0) or 0.0) * 1000.0
                + float(feature_metadata.get("admission_yield_ms", 0) or 0)
            ),
            lock_wait_ms=(
                float(getattr(result, "lock_wait_s", 0.0) or 0.0) * 1000.0
                + float(atcc_runtime.get("lock_wait_ms", 0.0) or 0.0)
            ),
            lock_hold_ms=float(atcc_runtime.get("lock_hold_ms", 0.0) or 0.0),
            reasoning_delay_ms=reasoning_delay_ms,
            wasted_reasoning_ms=0.0 if committed else max(0.0, reasoning_delay_ms - skipped_reasoning_ms),
            skipped_reasoning_ms=skipped_reasoning_ms,
            background_aborts=float(atcc_runtime.get("background_aborts", 0.0) or 0.0),
            background_tps_loss=float(atcc_runtime.get("background_tps_loss", 0.0) or 0.0),
            admission_yield_ms=int(feature_metadata.get("admission_yield_ms", 0) or 0),
        )

    def _paper_like_action(self, action: str, features: ATCCFeatures) -> tuple[str, str]:
        normalized = normalize_action(action)
        if self.policy.training:
            return normalized, "policy-training-exploration"
        available = {normalize_action(item) for item in self.policy.trainable_actions}
        risk = risk_score(features)
        retrying = int(features.retry_count) > 0
        read_stale_risk = read_stale_risk_score(features)
        read_sensitive = read_stale_risk >= 3
        level = str(features.level).strip().lower()
        if (
            int(features.background_workers) > 0
            and self._trusted_policy_action(features, normalized, risk=risk, retrying=retrying)
        ):
            return normalized, "policy-trained"
        if level == "low" and normalize_action(WRITE_VALIDATE) in available:
            if normalized != WRITE_VALIDATE:
                return WRITE_VALIDATE, "runtime-low-conflict-write-validate"
            return normalized, "policy"
        if (
            not retrying
            and level == "medium"
            and str(features.workload).strip().lower() == "tpcc"
            and features.hot_write_count > 0
            and features.write_count > 0
        ):
            for candidate in tpcc_medium_protection_candidates(features, retrying=False):
                if atcc_candidate_available(candidate, available):
                    if normalized != candidate:
                        return candidate, "runtime-tpcc-medium-write-set-protect"
                    return normalized, "policy"
        if (
            read_sensitive
            and not retrying
            and level == "medium"
            and features.write_count <= 2
        ):
            candidate = WRITE_VALIDATE if normalize_action(WRITE_VALIDATE) in available else OCC
            if normalized != candidate:
                return candidate, "runtime-medium-read-write-validate"
            return normalized, "policy"
        if read_sensitive and risk >= 4:
            severe_read_risk = read_stale_risk >= 5 and level == "high"
            candidates = (RESERVE_READ_WRITE_SET, RESERVE_HOT_RW) if severe_read_risk else (
                RESERVE_HOT_RW,
                LOCK_BEFORE_COMMIT,
                RESERVE_READ_WRITE_SET,
            )
            for candidate in candidates:
                if normalize_action(candidate) in available:
                    if normalized != candidate:
                        return candidate, "runtime-hot-read-protect"
                    return normalized, "policy"
        if (
            not retrying
            and str(features.level).strip().lower() == "medium"
            and features.read_count <= 0
            and features.hot_write_count > 0
            and risk >= 5
        ):
            for candidate in (LOCK_BEFORE_COMMIT, RESERVE_HOT, RESERVE_HOT_RW, LOCK_WRITE_SET):
                if normalize_action(candidate) in available:
                    if normalized != candidate:
                        return candidate, "runtime-hot-write-deferred-protect"
                    return normalized, "policy"
        high_risk_first_attempt = (
            not retrying
            and risk >= 6
            and normalize_action(LOCK_BEFORE_COMMIT) in available
        )
        if high_risk_first_attempt and normalized in {
            OCC,
            WRITE_VALIDATE,
            LOCK_WRITE_SET,
            RETRY_PROTECT,
            RESERVE_HOT,
            RESERVE_HOT_RW,
        }:
            return LOCK_BEFORE_COMMIT, "runtime-risk-deferred-protect"
        if (
            retrying
            and level == "medium"
            and str(features.workload).strip().lower() == "tpcc"
            and features.hot_write_count > 0
            and features.write_count > 0
        ):
            for candidate in tpcc_medium_protection_candidates(features, retrying=True):
                if atcc_candidate_available(candidate, available):
                    if normalized != candidate:
                        return candidate, "runtime-tpcc-medium-retry-write-set-protect"
                    return normalized, "policy"
        if retrying and risk >= 4:
            candidates = (
                (RESERVE_HOT, LOCK_BEFORE_COMMIT, RESERVE_HOT_RW, RETRY_PROTECT, LOCK_WRITE_SET)
                if features.read_count <= 0
                else (RESERVE_HOT_RW, LOCK_BEFORE_COMMIT, RETRY_PROTECT, LOCK_WRITE_SET)
            )
            for candidate in candidates:
                if normalize_action(candidate) in available:
                    if normalized != candidate:
                        return candidate, "runtime-retry-priority-protect"
                    return normalized, "policy"
        if (
            not retrying
            and normalized == LOCK_WRITE_SET
            and normalize_action(LOCK_BEFORE_COMMIT) in available
        ):
            return LOCK_BEFORE_COMMIT, "runtime-deferred-lock-hold"
        if normalized in {OCC, WRITE_VALIDATE} and risk >= 8:
            for candidate in (LOCK_BEFORE_COMMIT, RESERVE_HOT_RW, RESERVE_HOT):
                if normalize_action(candidate) in available:
                    return candidate, "runtime-risk-protect"
        return normalized, "policy"

    def _trusted_policy_action(
        self,
        features: ATCCFeatures,
        action: str,
        *,
        risk: int,
        retrying: bool,
    ) -> bool:
        row = self.policy.rows.get(str(features.state_key))
        if not isinstance(row, ATCCPolicyRow):
            return False
        min_visits = max(1, int(getattr(self.policy, "min_visits", 1) or 1))
        if row.visits < min_visits:
            return False
        normalized = normalize_action(action)
        stats = row.actions.get(normalized)
        if stats is None or stats.visits <= 0:
            return False
        abort_rate = stats.aborts / stats.visits if stats.visits else 0.0
        level = str(features.level).strip().lower()
        admission_retry = (
            retrying
            and str(features.previous_failure_reason).strip().lower() == "reservation-timeout"
        )
        if (
            level == "medium"
            and features.read_count <= 0
            and features.hot_write_count > 0
            and risk >= 5
            and normalized not in {
                LOCK_BEFORE_COMMIT,
                LOCK_HOT_BEFORE_COMMIT,
                RESERVE_HOT,
                RESERVE_HOT_RW_K,
            }
        ):
            return False
        if normalized == OCC:
            safe_abort_rate = float(getattr(self.policy, "low_conflict_safe_abort_rate", 0.5) or 0.5)
            if retrying and risk >= 6 and not admission_retry:
                return False
            return stats.commits > 0 and abort_rate <= safe_abort_rate
        if normalized == WRITE_VALIDATE:
            if level == "high" and retrying and risk >= 6 and not admission_retry:
                return False
            return stats.commits > 0 and (not admission_retry or abort_rate <= 0.5)
        if normalized == RESERVE_READ_WRITE_SET and level != "high":
            return False
        if normalized == RESERVE_HOT and level == "high" and features.read_count <= 0:
            return False
        return stats.commits > 0

    def _priority_for(self, features: ATCCFeatures, *, state_key: str) -> tuple[int, str]:
        row = self.policy.rows.get(str(state_key))
        row_priority = int(getattr(row, "priority", 0) or 0) if isinstance(row, ATCCPolicyRow) else 0
        if not self.runtime_guards_enabled:
            return row_priority, "policy-row" if row_priority > 0 else "policy-default"
        expected_cost = expected_abort_cost_ms(features)
        cost_priority = priority_from_cost(expected_cost) if expected_cost > 0 else 0
        retry_priority = min(9, int(features.retry_count) * 3)
        risk_priority = min(9, risk_score(features))
        priority = max(row_priority, cost_priority, retry_priority, risk_priority)
        if priority == row_priority and row_priority > 0:
            return priority, "policy-row"
        if priority == retry_priority and retry_priority > 0:
            return priority, "retry-count"
        if priority == cost_priority and cost_priority > 0:
            return priority, "reasoning-cost"
        if priority == risk_priority and risk_priority > 0:
            return priority, "risk"
        return 0, "default"


def static_threshold_action(features: ATCCFeatures) -> str:
    """Deterministic ATCC baseline for policy-table ablations."""

    level = str(features.level).strip().lower()
    retrying = int(features.retry_count) > 0
    risk = risk_score(features)
    read_risk = read_stale_risk_score(features)
    if level == "low":
        return WRITE_VALIDATE
    if retrying and risk >= 4:
        if features.read_count > 0:
            return RESERVE_HOT_RW
        return RETRY_PROTECT
    if read_risk >= 5 and level == "high":
        return RESERVE_READ_WRITE_SET
    if read_risk >= 3 and risk >= 4:
        return RESERVE_HOT_RW
    if risk >= 6:
        return LOCK_BEFORE_COMMIT
    if level == "medium" and features.hot_write_count > 0 and risk >= 5:
        return LOCK_BEFORE_COMMIT
    if features.hot_write_count > 0 and risk >= 4:
        return RESERVE_HOT
    if features.write_count > 0:
        return WRITE_VALIDATE
    return OCC


def tpcc_medium_protection_candidates(features: ATCCFeatures, *, retrying: bool) -> tuple[str, ...]:
    """Choose a TPC-C medium protection shape from the transaction footprint."""

    task_type = str(features.task_type).strip().lower()
    read_sensitive = read_stale_risk_score(features) >= 3
    write_count = int(features.write_count)
    hot_writes = int(features.hot_write_count)
    if read_sensitive:
        return (RESERVE_HOT_RW, LOCK_BEFORE_COMMIT, RESERVE_READ_WRITE_SET, LOCK_WRITE_SET)
    if task_type == "payment" or (write_count <= 3 and hot_writes <= 2):
        if retrying:
            return (RETRY_PROTECT, RESERVE_HOT, LOCK_BEFORE_COMMIT, LOCK_WRITE_SET)
        return (RESERVE_HOT, LOCK_BEFORE_COMMIT, LOCK_WRITE_SET)
    if retrying:
        return (LOCK_BEFORE_COMMIT, RETRY_PROTECT, RESERVE_HOT_RW_K, RESERVE_HOT_RW, LOCK_WRITE_SET)
    return (RESERVE_HOT_RW_K, RESERVE_HOT, LOCK_BEFORE_COMMIT, RESERVE_HOT_RW, LOCK_WRITE_SET)


def should_bound_hot_reservation_targets(features: ATCCFeatures, targets: Any) -> bool:
    if len(tuple(targets or ())) <= 1:
        return False
    if str(features.workload).strip().lower() != "tpcc":
        return False
    if str(features.level).strip().lower() != "high":
        return False
    if int(features.background_workers) <= 0:
        return False
    if int(features.read_count) > 0:
        return False
    return True


def hot_reservation_target_limit(features: ATCCFeatures) -> int:
    return 2


def atcc_candidate_available(candidate: str, available: set[str]) -> bool:
    normalized = normalize_action(candidate)
    if normalized == RESERVE_HOT_RW_K:
        return bool({RESERVE_HOT_RW, RESERVE_READ_WRITE_SET} & set(available))
    return normalized in available


def expected_abort_cost_ms(features: ATCCFeatures) -> float:
    level_factor = {
        "low": 0.25,
        "medium": 0.65,
        "high": 1.0,
    }.get(str(features.level).strip().lower(), 0.5)
    hot_factor = 0.35 * float(features.hot_write_count) + 0.15 * float(features.hot_read_count)
    retry_factor = 0.5 * float(max(0, features.retry_count))
    write_factor = 0.10 * float(max(0, features.write_count - 1))
    probability = min(1.0, level_factor + hot_factor + retry_factor + write_factor)
    return probability * float(features.reasoning_delay_ms)


def risk_score(features: ATCCFeatures) -> int:
    score = 0
    level = str(features.level).strip().lower()
    if level == "medium":
        score += 1
    elif level == "high":
        score += 3
    contention = contention_bucket(features.hot_write_count, features.write_count)
    score += {
        "cold": 0,
        "warm": 1,
        "hot": 3,
        "extreme": 4,
    }.get(contention, 0)
    cost = agent_cost_bucket(features.reasoning_delay_ms)
    score += {
        "short": 0,
        "medium": 1,
        "long": 2,
        "very-long": 3,
    }.get(cost, 0)
    score += min(2, read_stale_risk_score(features))
    if features.write_count >= 5:
        score += 1
    if features.retry_count > 0:
        score += min(3, int(features.retry_count) + 1)
    return min(9, score)


def read_stale_risk_score(features: ATCCFeatures) -> int:
    score = 0
    if features.read_count <= 0:
        return 0
    if features.hot_read_count > 0:
        score += 3
    if features.read_count >= 3:
        score += 1
    if features.read_count >= 7:
        score += 1
    level = str(features.level).strip().lower()
    if level == "medium":
        score += 1
    elif level == "high":
        score += 2
    if features.retry_count > 0:
        score += 1
    return score


def hot_rw_k_targets(features: ATCCFeatures, *, limit: int) -> tuple[str, ...]:
    cap = max(1, int(limit))
    writes = sorted(unique_ordered(features.hot_targets), key=hot_target_rank_key)
    reads = sorted(
        unique_ordered(target for target in features.hot_read_targets if str(target) not in set(writes)),
        key=hot_target_rank_key,
    )
    ordered = sorted(unique_ordered(tuple(writes) + tuple(reads)), key=hot_target_rank_key)
    pressure_aware = (
        bool(int(features.target_selection_seed))
        or any(int(features.reservation_queue_lengths.get(target, 0) or 0) > 0 for target in ordered)
        or bool(set(ordered) & set(features.reservation_owner_targets))
        or bool(set(ordered) & set(features.reservation_writer_targets))
    )
    if not pressure_aware:
        selected: list[str] = []
        for target in ordered:
            if target and target not in selected:
                selected.append(target)
            if len(selected) >= cap:
                break
        return tuple(selected)

    selected = []
    write_candidates = sorted(writes, key=lambda target: target_pressure_key(features, target, kind_rank=0))
    read_candidates = sorted(reads, key=lambda target: target_pressure_key(features, target, kind_rank=1))
    if write_candidates:
        selected.append(write_candidates[0])
    if len(selected) < cap and read_candidates:
        selected.append(read_candidates[0])
    remaining = [
        target
        for target in sorted(
            tuple(write_candidates[1:]) + tuple(read_candidates[1:]),
            key=lambda item: target_pressure_key(
                features,
                item,
                kind_rank=0 if item in set(writes) else 1,
            ),
        )
        if target not in selected
    ]
    for target in remaining:
        selected.append(target)
        if len(selected) >= cap:
            break
    return tuple(selected[:cap])


def unique_ordered(values: Any) -> list[str]:
    ordered: list[str] = []
    for target in values:
        text = str(target)
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def target_pressure_key(features: ATCCFeatures, target: str, *, kind_rank: int) -> tuple[int, int, int, int, int]:
    text = str(target)
    queue_length = int(features.reservation_queue_lengths.get(text, 0) or 0)
    active_owner = 1 if text in set(features.reservation_owner_targets) else 0
    active_writer = 1 if text in set(features.reservation_writer_targets) else 0
    return (
        active_owner + active_writer,
        queue_length,
        hot_target_domain_rank(text),
        int(kind_rank),
        stable_target_rank(int(features.target_selection_seed), text),
    )


def hot_target_rank_key(target: str) -> tuple[int, str]:
    text = str(target)
    return (hot_target_domain_rank(text), text)


def hot_target_domain_rank(target: str) -> int:
    text = str(target)
    if "next_order_id" in text:
        return 0
    if text.endswith(":orders"):
        return 1
    if text.startswith("tpcc:warehouse:") and text.endswith(":ytd"):
        return 2
    if ":stock:" in text and text.endswith(":quantity"):
        return 3
    if ":stock:" in text:
        return 4
    if ":record:" in text:
        return 5
    return 6


def stable_target_rank(seed: int, target: str) -> int:
    payload = f"{int(seed)}:{target}".encode("utf-8", errors="ignore")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def risk_bucket(score: int) -> str:
    value = int(score)
    if value <= 2:
        return "low"
    if value <= 5:
        return "medium"
    if value <= 7:
        return "high"
    return "critical"


def decision_reason(action: str, action_source: str) -> str:
    if action == OCC:
        return "dynamic-occ-fast-path"
    if action == WRITE_VALIDATE:
        return "dynamic-write-validate-fast-path"
    if action_source == "policy":
        return "dynamic-policy"
    return str(action_source)


def execution_path(action: str, lock_phase: str) -> str:
    if action == OCC or str(lock_phase) == "none":
        if action == WRITE_VALIDATE:
            return "snapshot-write-validate"
        return "optimistic"
    if str(lock_phase) == "before-commit":
        return "deferred-protect"
    if str(lock_phase) == "reserve":
        if action == RESERVE_HOT_RW_K:
            return "hot-set-reservation-k"
        return "hot-set-reservation"
    return "early-protect"

"""Single dynamic ATCC strategy backed by a policy table."""

from __future__ import annotations

from typing import Any, Optional

from agent.cc.atcc.actions import (
    LOCK_BEFORE_COMMIT,
    LOCK_WRITE_SET,
    OCC,
    RESERVE_HOT,
    RESERVE_HOT_RW,
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
    ):
        self.name = name
        self.policy = policy or ATCCPolicyTable()
        self.decision_mode = str(decision_mode).strip().lower() or "trained"
        if self.decision_mode not in {"trained", "static"}:
            raise ValueError(f"unsupported ATCC decision mode: {decision_mode}")
        self.priority_enabled = bool(priority_enabled)

    def decide(self, features: ATCCFeatures) -> ATCCDecision:
        state_key = self.state_key_for(features)
        if self.decision_mode == "static":
            policy_action = static_threshold_action(features)
            action, action_source = policy_action, "static-threshold"
        else:
            policy_action = self.policy.action_for(state_key)
            action, action_source = self._paper_like_action(policy_action, features)
        spec = action_spec(action, retry_count=features.retry_count)
        if spec.lock_scope == "hot":
            targets = features.hot_targets
        elif spec.lock_scope == "hot-rw":
            targets = tuple(sorted(set(features.hot_targets) | set(features.hot_read_targets)))
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
                "selected_action_source": action_source,
                "priority_reason": priority_reason,
                "decision_mode": self.decision_mode,
                "priority_enabled": self.priority_enabled,
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
            elapsed_ms=float(getattr(result, "elapsed_s", 0.0) or 0.0) * 1000.0,
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
        if level == "low" and normalize_action(WRITE_VALIDATE) in available:
            if normalized != WRITE_VALIDATE:
                return WRITE_VALIDATE, "runtime-low-conflict-write-validate"
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
        if self._trusted_policy_action(features, normalized, risk=risk, retrying=retrying):
            return normalized, "policy-trusted"
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
        if (
            level == "medium"
            and features.read_count <= 0
            and features.hot_write_count > 0
            and risk >= 5
            and normalized != LOCK_BEFORE_COMMIT
        ):
            return False
        if normalized == OCC:
            safe_abort_rate = float(getattr(self.policy, "low_conflict_safe_abort_rate", 0.5) or 0.5)
            if retrying and risk >= 6:
                return False
            return stats.commits > 0 and abort_rate <= safe_abort_rate and stats.avg_reward >= 0.0
        if normalized == WRITE_VALIDATE:
            if level == "high" and retrying and risk >= 6:
                return False
            return stats.commits > 0 and stats.avg_reward >= 0.0
        if normalized == RESERVE_READ_WRITE_SET and level != "high":
            return False
        if normalized == RESERVE_HOT and level == "high" and features.read_count <= 0:
            return False
        if normalized != LOCK_BEFORE_COMMIT and stats.avg_reward < 0.0:
            return False
        return stats.commits > 0 and stats.avg_reward >= row.avg_reward - 1.0

    def _priority_for(self, features: ATCCFeatures, *, state_key: str) -> tuple[int, str]:
        row = self.policy.rows.get(str(state_key))
        row_priority = int(getattr(row, "priority", 0) or 0) if isinstance(row, ATCCPolicyRow) else 0
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
        return "hot-set-reservation"
    return "early-protect"

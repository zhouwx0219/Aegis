"""Serializable policy table for dynamic ATCC."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, Mapping

from agent.cc.atcc.actions import (
    LOCK_BEFORE_COMMIT,
    LOCK_HOT,
    LOCK_WRITE_SET,
    OCC,
    RESERVE_HOT,
    RESERVE_HOT_RW,
    RESERVE_READ_WRITE_SET,
    RETRY_PROTECT,
    TRAINABLE_ACTIONS,
    WRITE_VALIDATE,
    normalize_action,
)
from agent.cc.atcc.reward import ATCCRewardConfig


@dataclasses.dataclass
class ATCCActionStats:
    visits: int = 0
    commits: int = 0
    aborts: int = 0
    avg_elapsed_ms: float = 0.0
    avg_lock_wait_ms: float = 0.0
    avg_lock_hold_ms: float = 0.0
    avg_reasoning_delay_ms: float = 0.0
    avg_wasted_reasoning_ms: float = 0.0
    avg_skipped_reasoning_ms: float = 0.0
    avg_reward: float = 0.0

    def observe(
        self,
        *,
        committed: bool,
        elapsed_ms: float,
        lock_wait_ms: float,
        lock_hold_ms: float,
        reasoning_delay_ms: float,
        wasted_reasoning_ms: float,
        skipped_reasoning_ms: float,
        reward: float,
    ) -> None:
        self.visits += 1
        if committed:
            self.commits += 1
        else:
            self.aborts += 1
        self.avg_elapsed_ms = running_average(self.avg_elapsed_ms, self.visits, elapsed_ms)
        self.avg_lock_wait_ms = running_average(self.avg_lock_wait_ms, self.visits, lock_wait_ms)
        self.avg_lock_hold_ms = running_average(self.avg_lock_hold_ms, self.visits, lock_hold_ms)
        self.avg_reasoning_delay_ms = running_average(
            self.avg_reasoning_delay_ms,
            self.visits,
            reasoning_delay_ms,
        )
        self.avg_wasted_reasoning_ms = running_average(
            self.avg_wasted_reasoning_ms,
            self.visits,
            wasted_reasoning_ms,
        )
        self.avg_skipped_reasoning_ms = running_average(
            self.avg_skipped_reasoning_ms,
            self.visits,
            skipped_reasoning_ms,
        )
        self.avg_reward = running_average(self.avg_reward, self.visits, reward)

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ATCCActionStats":
        return cls(
            visits=int(data.get("visits", 0) or 0),
            commits=int(data.get("commits", 0) or 0),
            aborts=int(data.get("aborts", 0) or 0),
            avg_elapsed_ms=float(data.get("avg_elapsed_ms", 0.0) or 0.0),
            avg_lock_wait_ms=float(data.get("avg_lock_wait_ms", 0.0) or 0.0),
            avg_lock_hold_ms=float(data.get("avg_lock_hold_ms", 0.0) or 0.0),
            avg_reasoning_delay_ms=float(data.get("avg_reasoning_delay_ms", 0.0) or 0.0),
            avg_wasted_reasoning_ms=float(data.get("avg_wasted_reasoning_ms", 0.0) or 0.0),
            avg_skipped_reasoning_ms=float(data.get("avg_skipped_reasoning_ms", 0.0) or 0.0),
            avg_reward=float(data.get("avg_reward", 0.0) or 0.0),
        )


@dataclasses.dataclass
class ATCCPolicyRow:
    action: str = OCC
    priority: int = 0
    visits: int = 0
    aborts: int = 0
    commits: int = 0
    avg_reward: float = 0.0
    actions: Dict[str, ATCCActionStats] = dataclasses.field(default_factory=dict)

    # Legacy aggregate fields kept for old tests/artifact readability.
    occ_visits: int = 0
    occ_aborts: int = 0
    protect_visits: int = 0
    protect_aborts: int = 0
    avg_elapsed_ms: float = 0.0
    avg_lock_wait_ms: float = 0.0
    avg_reasoning_delay_ms: float = 0.0
    avg_abort_cost_ms: float = 0.0

    def stats_for(self, action: str) -> ATCCActionStats:
        return self.actions.setdefault(normalize_action(action), ATCCActionStats())

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": normalize_action(self.action),
            "priority": int(self.priority),
            "visits": int(self.visits),
            "aborts": int(self.aborts),
            "commits": int(self.commits),
            "avg_reward": float(self.avg_reward),
            "actions": {
                action: stats.to_dict()
                for action, stats in sorted(self.actions.items())
            },
            "occ_visits": int(self.occ_visits),
            "occ_aborts": int(self.occ_aborts),
            "protect_visits": int(self.protect_visits),
            "protect_aborts": int(self.protect_aborts),
            "avg_elapsed_ms": float(self.avg_elapsed_ms),
            "avg_lock_wait_ms": float(self.avg_lock_wait_ms),
            "avg_reasoning_delay_ms": float(self.avg_reasoning_delay_ms),
            "avg_abort_cost_ms": float(self.avg_abort_cost_ms),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ATCCPolicyRow":
        action_stats = {
            normalize_action(key): ATCCActionStats.from_dict(value if isinstance(value, Mapping) else {})
            for key, value in dict(data.get("actions", {}) or {}).items()
        }
        row = cls(
            action=normalize_action(str(data.get("action", OCC) or OCC)),
            priority=int(data.get("priority", 0) or 0),
            visits=int(data.get("visits", 0) or 0),
            aborts=int(data.get("aborts", 0) or 0),
            commits=int(data.get("commits", 0) or 0),
            avg_reward=float(data.get("avg_reward", 0.0) or 0.0),
            actions=action_stats,
            occ_visits=int(data.get("occ_visits", 0) or 0),
            occ_aborts=int(data.get("occ_aborts", 0) or 0),
            protect_visits=int(data.get("protect_visits", 0) or 0),
            protect_aborts=int(data.get("protect_aborts", 0) or 0),
            avg_elapsed_ms=float(data.get("avg_elapsed_ms", 0.0) or 0.0),
            avg_lock_wait_ms=float(data.get("avg_lock_wait_ms", 0.0) or 0.0),
            avg_reasoning_delay_ms=float(data.get("avg_reasoning_delay_ms", 0.0) or 0.0),
            avg_abort_cost_ms=float(data.get("avg_abort_cost_ms", 0.0) or 0.0),
        )
        if not row.actions:
            seed_legacy_actions(row)
        return row


@dataclasses.dataclass
class ATCCPolicyTable:
    rows: Dict[str, ATCCPolicyRow] = dataclasses.field(default_factory=dict)
    default_action: str = OCC
    abort_threshold: float = 0.20
    min_visits: int = 5
    protect_cost_threshold_ms: float = 10.0
    low_conflict_occ_guard: bool = True
    low_conflict_safe_abort_rate: float = 0.50
    sparse_state_risk_prior: bool = True
    reward_config: ATCCRewardConfig = dataclasses.field(default_factory=ATCCRewardConfig)
    trainable_actions: tuple[str, ...] = TRAINABLE_ACTIONS
    exploration_coefficient: float = 1.5
    frozen: bool = False
    training: bool = False

    def set_frozen(self, frozen: bool) -> "ATCCPolicyTable":
        self.frozen = bool(frozen)
        return self

    def set_mode(self, mode: str) -> "ATCCPolicyTable":
        normalized = str(mode).strip().lower()
        self.frozen = normalized == "eval"
        self.training = normalized == "train"
        return self

    def action_for(self, state_key: str) -> str:
        row = self.rows.get(str(state_key))
        if row is None and self.training:
            row = self.rows.setdefault(
                str(state_key),
                ATCCPolicyRow(action=normalize_action(self.default_action)),
            )
        occ_stats = row.actions.get(OCC) if row is not None else None
        if self._should_keep_low_conflict_occ(state_key, occ_stats):
            return OCC
        if self.training and row is not None:
            exploratory = self._exploration_action(row)
            if exploratory:
                return exploratory
        if row is None or row.visits < self.min_visits:
            return self._sparse_state_action(state_key)
        return normalize_action(row.action)

    def observe(
        self,
        state_key: str,
        *,
        action: str,
        committed: bool,
        elapsed_ms: float = 0.0,
        lock_wait_ms: float = 0.0,
        lock_hold_ms: float = 0.0,
        reasoning_delay_ms: float = 0.0,
        wasted_reasoning_ms: float | None = None,
        skipped_reasoning_ms: float = 0.0,
        background_aborts: float = 0.0,
        background_tps_loss: float = 0.0,
    ) -> None:
        if self.frozen:
            return
        key = str(state_key)
        normalized_action = normalize_action(action)
        row = self.rows.setdefault(key, ATCCPolicyRow(action=normalize_action(self.default_action)))
        row.visits += 1
        if committed:
            row.commits += 1
        else:
            row.aborts += 1

        wasted = (
            float(wasted_reasoning_ms)
            if wasted_reasoning_ms is not None
            else (0.0 if committed else float(reasoning_delay_ms))
        )
        reward = self.reward_config.reward(
            committed=bool(committed),
            elapsed_ms=float(elapsed_ms),
            lock_wait_ms=float(lock_wait_ms),
            wasted_reasoning_ms=wasted,
            lock_hold_ms=float(lock_hold_ms),
            background_aborts=float(background_aborts),
            background_tps_loss=float(background_tps_loss),
        )
        stats = row.stats_for(normalized_action)
        stats.observe(
            committed=bool(committed),
            elapsed_ms=float(elapsed_ms),
            lock_wait_ms=float(lock_wait_ms),
            lock_hold_ms=float(lock_hold_ms),
            reasoning_delay_ms=float(reasoning_delay_ms),
            wasted_reasoning_ms=wasted,
            skipped_reasoning_ms=float(skipped_reasoning_ms),
            reward=reward,
        )

        if normalized_action == OCC:
            row.occ_visits += 1
            if not committed:
                row.occ_aborts += 1
        elif action_uses_protection(normalized_action):
            row.protect_visits += 1
            if not committed:
                row.protect_aborts += 1

        row.avg_elapsed_ms = running_average(row.avg_elapsed_ms, row.visits, elapsed_ms)
        row.avg_lock_wait_ms = running_average(row.avg_lock_wait_ms, row.visits, lock_wait_ms)
        row.avg_reasoning_delay_ms = running_average(
            row.avg_reasoning_delay_ms,
            row.visits,
            reasoning_delay_ms,
        )
        if not committed:
            row.avg_abort_cost_ms = running_average(
                row.avg_abort_cost_ms,
                row.aborts,
                float(elapsed_ms) + float(reasoning_delay_ms),
            )
        row.avg_reward = running_average(row.avg_reward, row.visits, reward)
        self._refresh_decision(key, row)

    def _refresh_decision(self, state_key: str, row: ATCCPolicyRow) -> None:
        eligible = {
            action: stats
            for action, stats in row.actions.items()
            if stats.visits >= self.min_visits
        }
        occ_stats = row.actions.get(OCC)
        if self._should_keep_low_conflict_occ(state_key, occ_stats):
            row.action = OCC
            row.priority = 0
            return
        if eligible:
            best_action, best_stats = max(
                eligible.items(),
                key=lambda item: (item[1].avg_reward, item[1].commits, -item[1].avg_lock_wait_ms),
            )
            row.action = normalize_action(best_action)
            row.priority = priority_from_stats(best_stats)
            return

        if occ_stats and occ_stats.visits >= max(1, min(self.min_visits, 2)):
            occ_abort_rate = occ_stats.aborts / occ_stats.visits if occ_stats.visits else 0.0
            expected_abort_cost = occ_abort_rate * (
                occ_stats.avg_wasted_reasoning_ms + occ_stats.avg_elapsed_ms
            )
            if (
                occ_abort_rate >= self.abort_threshold
                or expected_abort_cost >= self.protect_cost_threshold_ms
            ):
                row.action = LOCK_BEFORE_COMMIT
                row.priority = priority_from_cost(expected_abort_cost)
                return
        row.action = normalize_action(self.default_action)
        row.priority = 0

    def _should_keep_low_conflict_occ(
        self,
        state_key: str,
        occ_stats: ATCCActionStats | None,
    ) -> bool:
        if not self.low_conflict_occ_guard:
            return False
        if "level=low" not in str(state_key):
            return False
        if occ_stats is None or occ_stats.visits < max(1, min(self.min_visits, 2)):
            return True
        occ_abort_rate = occ_stats.aborts / occ_stats.visits if occ_stats.visits else 0.0
        return occ_abort_rate <= float(self.low_conflict_safe_abort_rate)

    def _sparse_state_action(self, state_key: str) -> str:
        if not self.sparse_state_risk_prior:
            return normalize_action(self.default_action)
        parts = parse_state_key(state_key)
        level = parts.get("level", "")
        contention = parts.get("contention", "")
        agent_cost = parts.get("agent_cost", "")
        hot_reads = parts.get("hot_reads", "")
        read_set = parts.get("read_set", "")
        retry = parts.get("retry", "")
        if retry == "retry":
            if hot_reads in {"some", "many"}:
                if level == "medium" and read_set in {"medium", "large"}:
                    return self._first_available(
                        (WRITE_VALIDATE, RESERVE_HOT_RW, RESERVE_READ_WRITE_SET, RETRY_PROTECT)
                    )
                if level == "high" and read_set in {"large", "medium"}:
                    return self._first_available(
                        (RESERVE_READ_WRITE_SET, RESERVE_HOT_RW, RETRY_PROTECT, LOCK_BEFORE_COMMIT)
                    )
                return self._first_available(
                    (RESERVE_HOT_RW, RESERVE_READ_WRITE_SET, RETRY_PROTECT, LOCK_BEFORE_COMMIT)
                )
            return self._first_available(
                (RETRY_PROTECT, LOCK_BEFORE_COMMIT, RESERVE_HOT_RW, LOCK_WRITE_SET)
            )
        hot = contention in {"hot", "extreme"}
        expensive = agent_cost in {"long", "very-long"}
        read_hot = hot_reads in {"some", "many"}
        read_heavy = read_set in {"medium", "large"}
        if read_hot and (level in {"medium", "high"} or read_heavy):
            if level == "medium" and read_heavy:
                return self._first_available(
                    (WRITE_VALIDATE, RESERVE_HOT_RW, RESERVE_READ_WRITE_SET, LOCK_BEFORE_COMMIT)
                )
            if level == "high" and read_heavy:
                return self._first_available(
                    (RESERVE_READ_WRITE_SET, RESERVE_HOT_RW, LOCK_BEFORE_COMMIT, LOCK_WRITE_SET)
                )
            return self._first_available(
                (RESERVE_HOT_RW, RESERVE_READ_WRITE_SET, LOCK_BEFORE_COMMIT, LOCK_WRITE_SET)
            )
        if level == "high" and (hot or expensive):
            return self._first_available(
                (LOCK_BEFORE_COMMIT, RESERVE_HOT_RW, RESERVE_HOT, LOCK_WRITE_SET)
            )
        if level == "medium" and hot and expensive:
            if read_set in {"none", ""}:
                return self._first_available(
                    (RESERVE_HOT, LOCK_BEFORE_COMMIT, RESERVE_HOT_RW, LOCK_WRITE_SET)
                )
            return self._first_available(
                (LOCK_BEFORE_COMMIT, RESERVE_HOT_RW, RESERVE_HOT, LOCK_WRITE_SET)
            )
        if contention == "extreme":
            return self._first_available(
                (LOCK_BEFORE_COMMIT, RESERVE_HOT_RW, RESERVE_HOT, LOCK_WRITE_SET)
            )
        return normalize_action(self.default_action)

    def _first_available(self, candidates: tuple[str, ...]) -> str:
        available = {normalize_action(action) for action in self.trainable_actions}
        for action in candidates:
            normalized = normalize_action(action)
            if normalized in available:
                return normalized
        return normalize_action(self.default_action)

    def _exploration_action(self, row: ATCCPolicyRow) -> str:
        for action in self.trainable_actions:
            if row.actions.get(action, ATCCActionStats()).visits < self.min_visits:
                return action
        total_visits = max(
            1,
            sum(
                row.actions.get(action, ATCCActionStats()).visits
                for action in self.trainable_actions
            ),
        )
        best_action = normalize_action(row.action)
        best_score = float("-inf")
        for action in self.trainable_actions:
            stats = row.actions.get(action, ATCCActionStats())
            visits = max(1, stats.visits)
            score = stats.avg_reward + self.exploration_coefficient * math.sqrt(
                math.log(total_visits + 1) / visits
            )
            if score > best_score:
                best_score = score
                best_action = action
        return normalize_action(best_action)

    def to_dict(self) -> Dict[str, object]:
        return {
            "artifact_type": "cast-das-atcc-policy",
            "version": 2,
            "default_action": normalize_action(self.default_action),
            "abort_threshold": self.abort_threshold,
            "min_visits": self.min_visits,
            "protect_cost_threshold_ms": self.protect_cost_threshold_ms,
            "low_conflict_occ_guard": bool(self.low_conflict_occ_guard),
            "low_conflict_safe_abort_rate": float(self.low_conflict_safe_abort_rate),
            "sparse_state_risk_prior": bool(self.sparse_state_risk_prior),
            "reward_config": self.reward_config.to_dict(),
            "trainable_actions": list(self.trainable_actions),
            "exploration_coefficient": float(self.exploration_coefficient),
            "rows": {key: row.to_dict() for key, row in sorted(self.rows.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ATCCPolicyTable":
        rows = {
            str(key): ATCCPolicyRow.from_dict(value if isinstance(value, Mapping) else {})
            for key, value in dict(data.get("rows", {}) or {}).items()
        }
        policy = cls(
            rows=rows,
            default_action=normalize_action(str(data.get("default_action", OCC) or OCC)),
            abort_threshold=float(data.get("abort_threshold", 0.20) or 0.20),
            min_visits=int(data.get("min_visits", 5) or 5),
            protect_cost_threshold_ms=float(data.get("protect_cost_threshold_ms", 10.0) or 10.0),
            low_conflict_occ_guard=bool(data.get("low_conflict_occ_guard", True)),
            low_conflict_safe_abort_rate=float(data.get("low_conflict_safe_abort_rate", 0.50) or 0.50),
            sparse_state_risk_prior=bool(data.get("sparse_state_risk_prior", True)),
            reward_config=ATCCRewardConfig.from_dict(
                data.get("reward_config", {}) if isinstance(data.get("reward_config", {}), Mapping) else {}
            ),
            trainable_actions=tuple(
                normalize_action(action)
                for action in data.get("trainable_actions", TRAINABLE_ACTIONS)
            ),
            exploration_coefficient=float(data.get("exploration_coefficient", 1.5) or 1.5),
        )
        for state_key, row in policy.rows.items():
            policy._refresh_decision(state_key, row)
        return policy

    @classmethod
    def load_json(cls, path: Path) -> "ATCCPolicyTable":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_legacy_actions(row: ATCCPolicyRow) -> None:
    if row.occ_visits:
        row.actions[OCC] = ATCCActionStats(
            visits=row.occ_visits,
            commits=max(0, row.occ_visits - row.occ_aborts),
            aborts=row.occ_aborts,
            avg_elapsed_ms=row.avg_elapsed_ms,
            avg_lock_wait_ms=row.avg_lock_wait_ms,
            avg_reasoning_delay_ms=row.avg_reasoning_delay_ms,
            avg_wasted_reasoning_ms=row.avg_abort_cost_ms,
            avg_reward=-row.avg_abort_cost_ms if row.occ_aborts else row.avg_reward,
        )
    if row.protect_visits:
        row.actions[LOCK_WRITE_SET] = ATCCActionStats(
            visits=row.protect_visits,
            commits=max(0, row.protect_visits - row.protect_aborts),
            aborts=row.protect_aborts,
            avg_elapsed_ms=row.avg_elapsed_ms,
            avg_lock_wait_ms=row.avg_lock_wait_ms,
            avg_reasoning_delay_ms=row.avg_reasoning_delay_ms,
            avg_reward=row.avg_reward,
        )


def action_uses_protection(action: str) -> bool:
    return normalize_action(action) in {
        LOCK_HOT,
        RESERVE_HOT,
        RESERVE_HOT_RW,
        RESERVE_READ_WRITE_SET,
        LOCK_WRITE_SET,
        LOCK_BEFORE_COMMIT,
        "retry-protect",
    }


def parse_state_key(state_key: str) -> Dict[str, str]:
    parts: Dict[str, str] = {}
    for item in str(state_key).split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[str(key)] = str(value)
    return parts


def running_average(current: float, count_after: int, value: float) -> float:
    count = max(1, int(count_after))
    return float(current) + (float(value) - float(current)) / count


def priority_from_stats(stats: ATCCActionStats) -> int:
    abort_cost = stats.avg_wasted_reasoning_ms + stats.avg_elapsed_ms
    return priority_from_cost(abort_cost)


def priority_from_cost(expected_abort_cost_ms: float) -> int:
    cost = float(expected_abort_cost_ms)
    if cost < 10:
        return 1
    if cost < 30:
        return 3
    if cost < 80:
        return 6
    return 9

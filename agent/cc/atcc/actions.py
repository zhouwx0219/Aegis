"""ATCC action schema."""

from __future__ import annotations

import dataclasses
from typing import Iterable, Tuple


OCC = "occ"
WRITE_VALIDATE = "write-validate"
LOCK_HOT = "lock-hot"
RESERVE_HOT = "reserve-hot"
RESERVE_HOT_RW = "reserve-hot-rw"
RESERVE_HOT_RW_K = "reserve-hot-rw-k"
RESERVE_READ_WRITE_SET = "reserve-read-write-set"
LOCK_WRITE_SET = "lock-write-set"
LOCK_BEFORE_COMMIT = "lock-before-commit"
LOCK_HOT_BEFORE_COMMIT = "lock-hot-before-commit"
RETRY_PROTECT = "retry-protect"

TRAINABLE_ACTIONS: Tuple[str, ...] = (
    OCC,
    WRITE_VALIDATE,
    LOCK_HOT,
    RESERVE_HOT,
    RESERVE_HOT_RW,
    RESERVE_READ_WRITE_SET,
    LOCK_WRITE_SET,
    LOCK_BEFORE_COMMIT,
    LOCK_HOT_BEFORE_COMMIT,
    RETRY_PROTECT,
)

MIXED_TRAINABLE_ACTIONS: Tuple[str, ...] = (
    OCC,
    WRITE_VALIDATE,
    RESERVE_HOT,
    RESERVE_HOT_RW,
    RESERVE_HOT_RW_K,
    RESERVE_READ_WRITE_SET,
    LOCK_BEFORE_COMMIT,
    LOCK_HOT_BEFORE_COMMIT,
    RETRY_PROTECT,
)

RUNTIME_ACTIONS: Tuple[str, ...] = (
    RESERVE_HOT_RW_K,
)

KNOWN_ACTIONS = set(TRAINABLE_ACTIONS) | set(RUNTIME_ACTIONS)

LEGACY_ACTIONS = {
    "protect": LOCK_WRITE_SET,
    "pessimistic": LOCK_WRITE_SET,
    "optimistic": OCC,
    "mvcc": WRITE_VALIDATE,
    "snapshot": WRITE_VALIDATE,
    "snapshot-write": WRITE_VALIDATE,
}


@dataclasses.dataclass(frozen=True)
class ATCCActionSpec:
    action: str
    lock_scope: str = "none"
    lock_phase: str = "none"
    retry_only: bool = False
    validate_reads: bool = True
    validate_writes: bool = True

    @property
    def uses_locking(self) -> bool:
        return self.lock_scope != "none"

    @property
    def begins_locked(self) -> bool:
        return self.uses_locking and self.lock_phase == "begin"

    @property
    def locks_before_commit(self) -> bool:
        return self.uses_locking and self.lock_phase == "before-commit"


def normalize_action(action: str) -> str:
    normalized = str(action).strip().lower().replace("_", "-")
    normalized = LEGACY_ACTIONS.get(normalized, normalized)
    if normalized not in KNOWN_ACTIONS:
        return OCC
    return normalized


def action_spec(action: str, *, retry_count: int = 0) -> ATCCActionSpec:
    normalized = normalize_action(action)
    if normalized == WRITE_VALIDATE:
        return ATCCActionSpec(normalized, validate_reads=False, validate_writes=True)
    if normalized == LOCK_HOT:
        return ATCCActionSpec(normalized, lock_scope="hot", lock_phase="begin")
    if normalized == RESERVE_HOT:
        return ATCCActionSpec(normalized, lock_scope="hot", lock_phase="reserve")
    if normalized == RESERVE_HOT_RW:
        return ATCCActionSpec(normalized, lock_scope="hot-rw", lock_phase="reserve")
    if normalized == RESERVE_HOT_RW_K:
        return ATCCActionSpec(normalized, lock_scope="hot-rw-k", lock_phase="reserve")
    if normalized == RESERVE_READ_WRITE_SET:
        return ATCCActionSpec(normalized, lock_scope="read-write-set", lock_phase="reserve")
    if normalized == LOCK_WRITE_SET:
        return ATCCActionSpec(normalized, lock_scope="write-set", lock_phase="begin")
    if normalized == LOCK_BEFORE_COMMIT:
        return ATCCActionSpec(normalized, lock_scope="write-set", lock_phase="before-commit")
    if normalized == LOCK_HOT_BEFORE_COMMIT:
        return ATCCActionSpec(normalized, lock_scope="hot", lock_phase="before-commit")
    if normalized == RETRY_PROTECT:
        if int(retry_count) > 0:
            return ATCCActionSpec(normalized, lock_scope="write-set", lock_phase="begin", retry_only=True)
        return ATCCActionSpec(normalized, lock_scope="none", lock_phase="none", retry_only=True)
    return ATCCActionSpec(OCC)


def expands_to_locking(action: str, *, retry_count: int = 0) -> bool:
    return action_spec(action, retry_count=retry_count).uses_locking


def target_scope(action: str, *, retry_count: int = 0) -> str:
    return action_spec(action, retry_count=retry_count).lock_scope


def all_actions(value: str | Iterable[str] | None = None) -> Tuple[str, ...]:
    if value is None:
        return TRAINABLE_ACTIONS
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw = [str(item).strip() for item in value if str(item).strip()]
    actions = tuple(dict.fromkeys(normalize_action(item) for item in raw))
    return actions or TRAINABLE_ACTIONS

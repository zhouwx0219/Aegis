"""Shared DTOs for ATCC decisions."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Tuple

from agent.cc.atcc.actions import action_spec, normalize_action


@dataclasses.dataclass(frozen=True)
class ATCCDecision:
    action: str
    targets: Tuple[str, ...]
    priority: int = 0
    state_key: str = ""
    reason: str = ""
    lock_scope: str = "none"
    lock_phase: str = "none"
    metadata: Dict[str, object] = dataclasses.field(default_factory=dict)

    @property
    def uses_locking(self) -> bool:
        return self.lock_scope != "none" and bool(self.targets)

    @property
    def begins_locked(self) -> bool:
        return self.uses_locking and self.lock_phase == "begin"

    @property
    def locks_before_commit(self) -> bool:
        return self.uses_locking and self.lock_phase == "before-commit"


def decision_from_preplan(txn: Any) -> ATCCDecision | None:
    metadata = dict(getattr(txn, "metadata", {}) or {})
    preplan = dict(metadata.get("atcc_preplan", {}) or {})
    if not preplan:
        return None
    action = normalize_action(str(preplan.get("action", "occ") or "occ"))
    spec = action_spec(action, retry_count=int(metadata.get("retry_count", 0) or 0))
    return ATCCDecision(
        action=action,
        targets=tuple(str(target) for target in preplan.get("targets", ()) or ()),
        priority=int(preplan.get("priority", 0) or 0),
        state_key=str(preplan.get("state_key", "") or ""),
        reason=str(preplan.get("reason", "preplanned-atcc") or "preplanned-atcc"),
        lock_scope=str(preplan.get("lock_scope", spec.lock_scope) or spec.lock_scope),
        lock_phase=str(preplan.get("lock_phase", spec.lock_phase) or spec.lock_phase),
        metadata=dict(preplan.get("metadata", {}) or {}),
    )

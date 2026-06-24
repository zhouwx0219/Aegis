"""Multi-branch transaction semantics and candidate construction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from agent.native import load_cast_core
from agent.runtime.types import SnapshotValue

cc = load_cast_core()


class CandidateDraft:
    """One generated candidate and its buffered writes."""

    def __init__(
        self,
        txn: Any,
        branch_id: str,
        quality: float,
        gen_cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.txn = txn
        self.branch_id = branch_id
        self.quality = float(quality)
        self.gen_cost = float(gen_cost)
        self.metadata = dict(metadata or {})
        self._writes: List[Any] = []
        self._intent_names: List[str] = []

    def _snapshot(self, object_id: str) -> SnapshotValue:
        with self.txn._lifecycle_lock:
            self.txn._ensure_active()
            if object_id not in self.txn.snapshot:
                raise KeyError(f"object was not in transaction snapshot: {object_id}")
            return self.txn.snapshot[object_id]

    def _write(
        self, object_id: str, branch_value: str, intent: Any, name: str
    ) -> "CandidateDraft":
        with self.txn._lifecycle_lock:
            snap = self._snapshot(object_id)
            if any(write.object_id == object_id for write in self._writes):
                raise ValueError(f"candidate already writes object: {object_id}")
            write = cc.BranchWrite()
            write.object_id = object_id
            write.kind = self.txn.manager.kind_of(object_id)
            write.base_value = snap.value
            write.base_version = snap.version
            write.branch_value = str(branch_value)
            write.intent = intent
            self._writes.append(write)
            self._intent_names.append(name)
            return self

    def overwrite(self, object_id: str, new_value: str) -> "CandidateDraft":
        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = cc.IntentType.kOverwrite
        return self._write(object_id, str(new_value), intent, "OVERWRITE")

    def append(
        self, object_id: str, payload: str, *, commutative: bool = False
    ) -> "CandidateDraft":
        snap = self._snapshot(object_id)
        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = cc.IntentType.kAppend
        intent.payload = str(payload)
        intent.commutative = bool(commutative)
        name = "APPEND_COMMUTATIVE" if commutative else "APPEND_ORDERED"
        return self._write(object_id, snap.value + str(payload), intent, name)

    def delta(
        self,
        object_id: str,
        amount: int,
        *,
        constrained: bool = False,
        lower_bound: int = 0,
    ) -> "CandidateDraft":
        snap = self._snapshot(object_id)
        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = cc.IntentType.kDelta
        intent.payload = str(int(amount))
        intent.constrained = bool(constrained)
        intent.lower_bound = int(lower_bound)
        name = "DELTA_CONSTRAINED" if constrained else "DELTA"
        return self._write(object_id, str(int(snap.value) + int(amount)), intent, name)

    def cas(self, object_id: str, expected: str, new_value: str) -> "CandidateDraft":
        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = cc.IntentType.kCas
        condition = cc.Condition()
        condition.type = cc.ConditionType.kValueEquals
        condition.expected_value = str(expected)
        intent.condition = condition
        return self._write(object_id, str(new_value), intent, "CAS")

    def to_core(self) -> Any:
        branch = cc.SpeculativeBranch()
        branch.branch_id = self.branch_id
        branch.read_set = self.txn._read_set_for_core()
        branch.writes = self._writes
        branch.gen_cost = self.gen_cost
        branch.quality = self.quality
        return branch

    def to_trace(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "quality": self.quality,
            "gen_cost": self.gen_cost,
            "metadata": self.metadata,
            "intents": list(self._intent_names),
        }


class BranchSemantics:
    """Pluggable policy for turning agent candidates into commit branches."""

    name = "branch-semantics"
    family = "multi-branch"
    description = "Build commit branches for an agent task."

    def to_core_branches(self, txn: Any) -> List[Any]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
        }


class QualityRankedBranchSemantics(BranchSemantics):
    """Default K-candidate semantics: expose all candidates to ranked commit."""

    name = "quality-ranked"
    description = (
        "Multi-branch transaction semantics with quality-ranked candidate "
        "selection, semantic reselect, and regeneration boundaries."
    )

    def to_core_branches(self, txn: Any) -> List[Any]:
        return [candidate.to_core() for candidate in txn.candidates]


class FirstCandidateBranchSemantics(BranchSemantics):
    """Compatibility semantics for one-branch, traditional transaction paths."""

    name = "first-candidate"
    description = "Commit only the first candidate as a traditional transaction."

    def to_core_branches(self, txn: Any) -> List[Any]:
        candidates: Sequence[Any] = txn.candidates
        if not candidates:
            return []
        return [candidates[0].to_core()]

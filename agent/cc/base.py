"""Concurrency-control interfaces for agent-side transactions."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, Tuple


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""
    conflict_object_ids: Tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class CCPlan:
    strategy: str
    family: str
    lock_targets: Tuple[str, ...] = ()
    validate_reads: bool = True
    validate_writes: bool = True
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class ConcurrencyControl:
    """Base class for runtime-owned concurrency-control strategies."""

    name = "cc"
    family = "generic"
    description = "agent-side concurrency-control strategy"

    def plan(self, txn: Any) -> CCPlan:
        return CCPlan(strategy=self.name, family=self.family)

    def validate(self, txn: Any, store: Any) -> ValidationResult:
        plan = self.plan(txn)
        return validate_versions(
            txn,
            store,
            validate_reads=plan.validate_reads,
            validate_writes=plan.validate_writes,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
        }


def validate_versions(
    txn: Any,
    store: Any,
    *,
    validate_reads: bool,
    validate_writes: bool,
) -> ValidationResult:
    conflicts = set()
    if validate_reads:
        for object_id, read in getattr(txn, "read_set", {}).items():
            if int(store.get_version(str(object_id))) != int(read.version):
                conflicts.add(str(object_id))
    if validate_writes:
        for object_id, write in getattr(txn, "write_set", {}).items():
            if int(store.get_version(str(object_id))) != int(write.base_version):
                conflicts.add(str(object_id))
    if conflicts:
        return ValidationResult(
            False,
            "version conflict",
            tuple(sorted(conflicts)),
        )
    return ValidationResult(True)


def unique_targets(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))

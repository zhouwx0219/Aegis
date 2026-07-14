"""Traditional CC strategies owned by the agent runtime."""

from __future__ import annotations

from typing import Any, Iterable

from agent.cc.base import CCPlan, ConcurrencyControl, ValidationResult, unique_targets
from agent.cc.locks import LockConflict


class OccConcurrencyControl(ConcurrencyControl):
    name = "occ"
    family = "optimistic"
    description = "Strict OCC validation over read and write versions."


class PaperATCCConcurrencyControl(ConcurrencyControl):
    name = "paper-atcc"
    family = "paper-atcc"
    description = "Phase-aware operation-level ATCC with runtime-owned locking and unified commit."

    def plan(self, txn: Any) -> CCPlan:
        return CCPlan(
            strategy=self.name,
            family=self.family,
            validate_reads=True,
            validate_writes=True,
            metadata={"paper_atcc": True},
        )


class TwoPhaseLockingConcurrencyControl(ConcurrencyControl):
    family = "pessimistic"

    def __init__(self, name: str, policy: str):
        self.name = str(name)
        self.policy = str(policy)
        self.description = f"Strict 2PL with {self.policy} deadlock handling."

    def plan(self, txn: Any) -> CCPlan:
        targets = unique_targets(
            list(getattr(txn, "read_set", {}).keys())
            + list(getattr(txn, "write_set", {}).keys())
        )
        return CCPlan(
            strategy=self.name,
            family=self.family,
            lock_targets=targets,
            validate_reads=True,
            validate_writes=True,
            metadata={"lock_table": "2pl", "policy": self.policy},
        )


class MvccConcurrencyControl(ConcurrencyControl):
    name = "mvcc"
    family = "mvcc"
    description = "Serializable MVCC fallback with full observed-version validation."

    def plan(self, txn: Any) -> CCPlan:
        return CCPlan(
            strategy=self.name,
            family=self.family,
            validate_reads=True,
            validate_writes=True,
            metadata={"isolation": "serializable", "adapter": "cast-das-versioned-kv"},
        )


class SiloConcurrencyControl(ConcurrencyControl):
    name = "silo"
    family = "silo"
    description = "Silo-style commit with write-set locking and full validation."

    def plan(self, txn: Any) -> CCPlan:
        return CCPlan(
            strategy=self.name,
            family=self.family,
            lock_targets=unique_targets(getattr(txn, "write_set", {}).keys()),
            validate_reads=True,
            validate_writes=True,
            metadata={"lock_table": "exclusive", "wait": True},
        )


class TicTocConcurrencyControl(ConcurrencyControl):
    name = "tictoc"
    family = "tictoc"
    description = "Serializable TicToc adapter with write locking and full validation."

    def plan(self, txn: Any) -> CCPlan:
        return CCPlan(
            strategy=self.name,
            family=self.family,
            lock_targets=unique_targets(getattr(txn, "write_set", {}).keys()),
            validate_reads=True,
            validate_writes=True,
            metadata={
                "lock_table": "exclusive",
                "wait": True,
                "isolation": "serializable",
                "adapter": "cast-das-versioned-kv",
            },
        )


class PolarisConcurrencyControl(ConcurrencyControl):
    name = "polaris"
    family = "polaris"
    description = "Polaris/SILO_PRIO-style write locking with retry priority and full validation."

    def plan(self, txn: Any) -> CCPlan:
        priority = polaris_priority(txn)
        return CCPlan(
            strategy=self.name,
            family=self.family,
            lock_targets=unique_targets(getattr(txn, "write_set", {}).keys()),
            validate_reads=True,
            validate_writes=True,
            metadata={
                "lock_table": "exclusive",
                "wait": True,
                "priority": priority,
                "polaris_priority": priority,
            },
        )


class BambooConcurrencyControl(ConcurrencyControl):
    name = "bamboo"
    family = "bamboo"
    description = "Bamboo-style early-retire baseline approximated with short write-set locks and full validation."

    def plan(self, txn: Any) -> CCPlan:
        return CCPlan(
            strategy=self.name,
            family=self.family,
            lock_targets=unique_targets(getattr(txn, "write_set", {}).keys()),
            validate_reads=True,
            validate_writes=True,
            metadata={
                "lock_table": "exclusive",
                "wait": True,
                "bamboo_early_retire": True,
            },
        )


def polaris_priority(txn: Any) -> int:
    metadata = dict(getattr(txn, "metadata", {}) or {})
    context = dict(metadata.get("context", {}) or {})
    retry_count = int(metadata.get("retry_count", context.get("retry_count", 0)) or 0)
    return max(0, min(9, retry_count * 3))


def lock_conflict_result(exc: LockConflict) -> ValidationResult:
    return ValidationResult(False, exc.reason, exc.targets)


def read_write_targets(txn: Any) -> Iterable[str]:
    yield from getattr(txn, "read_set", {}).keys()
    yield from getattr(txn, "write_set", {}).keys()

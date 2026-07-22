"""Bamboo-style early-retire coordination for the unified agent runtime."""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .locks import LockConflict


@dataclasses.dataclass(frozen=True)
class BambooSnapshot:
    value: Any
    version: int
    exists: bool = True


@dataclasses.dataclass(frozen=True)
class _RetiredWrite:
    owner_id: int
    snapshot: BambooSnapshot


@dataclasses.dataclass
class _TransactionState:
    txn: Any
    status: str = "active"
    dependencies: Set[int] = dataclasses.field(default_factory=set)
    dependency_targets: Dict[int, Set[str]] = dataclasses.field(default_factory=dict)
    dependents: Set[int] = dataclasses.field(default_factory=set)
    retired_targets: Set[str] = dataclasses.field(default_factory=set)
    held_targets: Set[str] = dataclasses.field(default_factory=set)
    finalized: bool = False


class BambooRetireTable:
    """Short S/X locks plus speculative versions and commit dependencies.

    A writer retires its private version immediately after the write operation.
    A later Bamboo transaction may consume that version, but it becomes commit-
    dependent on the writer. Dependencies are resolved before validation and a
    failed writer cascades to transactions that observed its retired version.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._holders: Dict[str, List[Tuple[int, str]]] = {}
        self._retired: Dict[str, List[_RetiredWrite]] = {}
        self._states: Dict[int, _TransactionState] = {}

    def begin_access(
        self,
        txn: Any,
        object_id: str,
        *,
        write: bool,
    ) -> tuple[Optional[BambooSnapshot], float]:
        key = str(object_id)
        owner_id = id(txn)
        mode = "x" if write else "s"
        started_at = time.perf_counter()
        with self._condition:
            state = self._state_locked(txn)
            while not self._compatible_locked(key, owner_id=owner_id, mode=mode):
                self._ensure_active_locked(state, (key,))
                self._condition.wait(timeout=0.001)
            self._ensure_active_locked(state, (key,))
            self._holders.setdefault(key, []).append((owner_id, mode))
            state.held_targets.add(key)
            retired = self._latest_retired_locked(key, exclude_owner=owner_id)
            if retired is not None:
                try:
                    self._add_dependency_locked(
                        dependent_id=owner_id,
                        dependency_id=retired.owner_id,
                        target=key,
                    )
                    self._ensure_active_locked(state, (key,))
                    return retired.snapshot, time.perf_counter() - started_at
                except BaseException:
                    self._release_access_locked(owner_id, key)
                    self._condition.notify_all()
                    raise
            return None, time.perf_counter() - started_at

    def finish_read(self, txn: Any, object_id: str) -> None:
        with self._condition:
            self._release_access_locked(id(txn), str(object_id))
            self._condition.notify_all()

    def retire_write(
        self,
        txn: Any,
        object_id: str,
        *,
        value: Any,
        version: int,
        exists: bool = True,
    ) -> None:
        key = str(object_id)
        owner_id = id(txn)
        with self._condition:
            state = self._state_locked(txn)
            try:
                self._ensure_active_locked(state, (key,))
                rows = self._retired.setdefault(key, [])
                rows[:] = [row for row in rows if row.owner_id != owner_id]
                rows.append(
                    _RetiredWrite(
                        owner_id=owner_id,
                        snapshot=BambooSnapshot(
                            value=value,
                            version=int(version),
                            exists=bool(exists),
                        ),
                    )
                )
                state.retired_targets.add(key)
            finally:
                self._release_access_locked(owner_id, key)
                self._condition.notify_all()

    def cancel_access(self, txn: Any, object_id: str) -> None:
        with self._condition:
            self._release_access_locked(id(txn), str(object_id))
            self._condition.notify_all()

    def wait_for_dependencies(
        self,
        txn: Any,
        *,
        timeout_s: float = 30.0,
    ) -> tuple[bool, str, tuple[str, ...], float]:
        owner_id = id(txn)
        started_at = time.perf_counter()
        deadline = started_at + max(0.001, float(timeout_s))
        with self._condition:
            state = self._state_locked(txn)
            while True:
                if state.status == "aborted":
                    return (
                        False,
                        "bamboo dependency aborted",
                        self._dependency_targets_locked(state),
                        time.perf_counter() - started_at,
                    )
                active = []
                for dependency_id in tuple(state.dependencies):
                    dependency = self._states.get(dependency_id)
                    if dependency is None or dependency.status == "committed":
                        continue
                    if dependency.status == "aborted":
                        self._cascade_abort_locked(owner_id)
                        return (
                            False,
                            "bamboo dependency aborted",
                            self._dependency_targets_locked(state),
                            time.perf_counter() - started_at,
                        )
                    active.append(dependency_id)
                if not active:
                    return True, "", (), time.perf_counter() - started_at
                if time.perf_counter() >= deadline:
                    self._cascade_abort_locked(owner_id)
                    return (
                        False,
                        "bamboo dependency timeout",
                        self._dependency_targets_locked(state),
                        time.perf_counter() - started_at,
                    )
                self._condition.wait(timeout=0.001)

    def mark_committed(self, txn: Any) -> None:
        owner_id = id(txn)
        with self._condition:
            state = self._states.get(owner_id)
            if state is None:
                return
            if state.status == "aborted":
                raise LockConflict(
                    "bamboo dependency aborted",
                    self._dependency_targets_locked(state),
                    kind="version-conflict",
                )
            state.status = "committed"
            state.finalized = True
            self._remove_retired_locked(owner_id)
            self._release_owner_locked(owner_id)
            self._detach_dependencies_locked(owner_id)
            self._cleanup_terminal_locked(owner_id)
            self._condition.notify_all()

    def mark_aborted(self, txn: Any) -> None:
        owner_id = id(txn)
        with self._condition:
            state = self._states.get(owner_id)
            if state is None:
                return
            self._cascade_abort_locked(owner_id)
            state = self._states.get(owner_id)
            if state is not None:
                state.finalized = True
                self._cleanup_terminal_locked(owner_id)
            self._condition.notify_all()

    def _state_locked(self, txn: Any) -> _TransactionState:
        owner_id = id(txn)
        state = self._states.get(owner_id)
        if state is None:
            state = _TransactionState(txn=txn)
            self._states[owner_id] = state
        return state

    @staticmethod
    def _ensure_active_locked(
        state: _TransactionState,
        targets: tuple[str, ...],
    ) -> None:
        if state.status != "active":
            raise LockConflict(
                "bamboo dependency aborted",
                targets,
                kind="version-conflict",
            )

    def _compatible_locked(self, key: str, *, owner_id: int, mode: str) -> bool:
        others = [
            holder_mode
            for holder_id, holder_mode in self._holders.get(key, [])
            if holder_id != owner_id
        ]
        if not others:
            return True
        return mode == "s" and all(holder_mode == "s" for holder_mode in others)

    def _latest_retired_locked(
        self,
        key: str,
        *,
        exclude_owner: int,
    ) -> Optional[_RetiredWrite]:
        for row in reversed(self._retired.get(key, [])):
            if row.owner_id == exclude_owner:
                continue
            state = self._states.get(row.owner_id)
            if state is not None and state.status == "active":
                return row
        return None

    def _add_dependency_locked(
        self,
        *,
        dependent_id: int,
        dependency_id: int,
        target: str,
    ) -> None:
        if dependent_id == dependency_id:
            return
        dependent = self._states[dependent_id]
        dependency = self._states.get(dependency_id)
        if dependency is None or dependency.status == "committed":
            return
        if dependency.status == "aborted" or self._depends_on_locked(
            dependency_id,
            dependent_id,
        ):
            self._cascade_abort_locked(dependent_id)
            raise LockConflict(
                "bamboo dependency cycle or aborted predecessor",
                (str(target),),
                kind="version-conflict",
            )
        dependent.dependencies.add(dependency_id)
        dependent.dependency_targets.setdefault(dependency_id, set()).add(str(target))
        dependency.dependents.add(dependent_id)

    def _depends_on_locked(self, start_id: int, target_id: int) -> bool:
        pending = [int(start_id)]
        seen: Set[int] = set()
        while pending:
            current = pending.pop()
            if current == int(target_id):
                return True
            if current in seen:
                continue
            seen.add(current)
            state = self._states.get(current)
            if state is not None:
                pending.extend(state.dependencies)
        return False

    def _cascade_abort_locked(self, owner_id: int) -> None:
        pending = [int(owner_id)]
        affected: Set[int] = set()
        while pending:
            current = pending.pop()
            if current in affected:
                continue
            affected.add(current)
            state = self._states.get(current)
            if state is not None:
                pending.extend(state.dependents)
        for current in affected:
            state = self._states.get(current)
            if state is None:
                continue
            state.status = "aborted"
            self._remove_retired_locked(current)
            self._release_owner_locked(current)
        for current in affected:
            self._detach_dependencies_locked(current)
            state = self._states.get(current)
            if state is not None:
                state.dependents.clear()

    def _detach_dependencies_locked(self, owner_id: int) -> None:
        state = self._states.get(owner_id)
        if state is None:
            return
        dependencies = tuple(state.dependencies)
        state.dependencies.clear()
        state.dependency_targets.clear()
        for dependency_id in dependencies:
            dependency = self._states.get(dependency_id)
            if dependency is not None:
                dependency.dependents.discard(owner_id)
                self._cleanup_terminal_locked(dependency_id)

    def _remove_retired_locked(self, owner_id: int) -> None:
        state = self._states.get(owner_id)
        targets = tuple(state.retired_targets) if state is not None else tuple(self._retired)
        for key in targets:
            rows = [row for row in self._retired.get(key, []) if row.owner_id != owner_id]
            if rows:
                self._retired[key] = rows
            else:
                self._retired.pop(key, None)
        if state is not None:
            state.retired_targets.clear()

    def _release_access_locked(self, owner_id: int, key: str) -> None:
        rows = [row for row in self._holders.get(key, []) if row[0] != owner_id]
        if rows:
            self._holders[key] = rows
        else:
            self._holders.pop(key, None)
        state = self._states.get(owner_id)
        if state is not None:
            state.held_targets.discard(key)

    def _release_owner_locked(self, owner_id: int) -> None:
        state = self._states.get(owner_id)
        targets = tuple(state.held_targets) if state is not None else tuple(self._holders)
        for key in targets:
            self._release_access_locked(owner_id, key)

    @staticmethod
    def _dependency_targets_locked(state: _TransactionState) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    target
                    for targets in state.dependency_targets.values()
                    for target in targets
                }
            )
        )

    def _cleanup_terminal_locked(self, owner_id: int) -> None:
        state = self._states.get(owner_id)
        if state is None:
            return
        if (
            state.finalized
            and state.status in {"committed", "aborted"}
            and not state.dependencies
            and not state.dependents
            and not state.retired_targets
            and not state.held_targets
        ):
            self._states.pop(owner_id, None)

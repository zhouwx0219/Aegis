"""Transaction-scoped private versions and atomic commit publication."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterable
from typing import Any

from .types import SnapshotValue


@dataclasses.dataclass(frozen=True)
class CommittedVersion:
    object_id: str
    value: str
    version: int
    exists: bool
    commit_epoch: int
    owner_tid: str
    background: bool = False


@dataclasses.dataclass(frozen=True)
class PrivateVersion:
    object_id: str
    value: str


class VersionManager:
    """Keep old committed versions visible to pinned transaction snapshots.

    Physical writes are installed by the native atomic batch primitive while
    this manager's publication latch is held. A single commit epoch then makes
    every object in the batch visible together.
    """

    def __init__(self, store: Any, *, max_commit_states: int = 4096):
        self.store = store
        self._lock = threading.RLock()
        self._publication_condition = threading.Condition(threading.RLock())
        self._active_publishers = 0
        self._snapshot_active = False
        self._snapshot_waiters = 0
        self._epoch = 0
        self._history: dict[str, list[CommittedVersion]] = {}
        self._private: dict[str, tuple[tuple[str, str], ...]] = {}
        self._transaction_states: dict[str, str] = {}
        self._pins: dict[str, int] = {}
        self._materialized_pins: set[str] = set()
        self._lazy_pins: set[str] = set()
        self._last_agent_commit_epoch: dict[str, int] = {}
        self._dirty_objects: set[str] = set()
        self._operations_since_gc = 0
        self._gc_interval = 256
        self._max_commit_states = max(64, int(max_commit_states))
        self._diagnostics: dict[str, int] = {
            "private_prepares": 0,
            "private_discards": 0,
            "atomic_publishes": 0,
            "published_objects": 0,
            "gc_versions": 0,
        }

    def register_object(self, object_id: str) -> None:
        key = str(object_id)
        with self._lock:
            current = self.store.get(key)
            self._epoch += 1
            self._history[key] = [
                CommittedVersion(
                    object_id=key,
                    value=str(current.value),
                    version=int(current.version),
                    exists=bool(current.exists),
                    commit_epoch=self._epoch,
                    owner_tid="bootstrap",
                    background=False,
                )
            ]

    def synchronize_object(self, object_id: str) -> None:
        """Import a native commit made before paper versioning was enabled."""
        key = str(object_id)
        with self._lock:
            current = self.store.get(key)
            versions = self._history.get(key, ())
            if versions and versions[-1].version == int(current.version):
                return
            self._epoch += 1
            self._history.setdefault(key, []).append(
                CommittedVersion(
                    object_id=key,
                    value=str(current.value),
                    version=int(current.version),
                    exists=bool(current.exists),
                    commit_epoch=self._epoch,
                    owner_tid="native-before-paper-runtime",
                    background=False,
                )
            )
            self._last_agent_commit_epoch[key] = self._epoch

    def current_epoch(self) -> int:
        with self._lock:
            return int(self._epoch)

    def has_lazy_pins(self) -> bool:
        with self._lock:
            return bool(self._lazy_pins)

    def snapshot_current(
        self,
        object_ids: Iterable[str],
    ) -> tuple[int, dict[str, SnapshotValue]]:
        """Materialize one multi-object snapshot at a publication boundary."""
        keys = tuple(dict.fromkeys(str(value) for value in object_ids))
        self._enter_snapshot()
        try:
            with self._lock:
                epoch = int(self._epoch)
                result = {
                    key: SnapshotValue(
                        value=str(current.value),
                        version=int(current.version),
                        exists=bool(current.exists),
                    )
                    for key in keys
                    for current in (self.store.get(key),)
                }
                return epoch, result
        finally:
            self._exit_snapshot()

    def pin(self, tid: str, epoch: int, *, materialized: bool = False) -> None:
        with self._lock:
            key = str(tid)
            self._pins[key] = min(int(epoch), self._epoch)
            if materialized:
                self._materialized_pins.add(key)
                self._lazy_pins.discard(key)
            else:
                self._lazy_pins.add(key)
                self._materialized_pins.discard(key)
            self._transaction_states.setdefault(key, "active")

    def pin_lazy(self, tid: str, object_ids: Iterable[str]) -> int:
        """Atomically establish a full baseline for an unplanned transaction."""
        key = str(tid)
        keys = tuple(dict.fromkeys(str(value) for value in object_ids))
        self._enter_snapshot()
        try:
            with self._lock:
                epoch = int(self._epoch)
                for object_id in keys:
                    current = self.store.get(object_id)
                    self._history[object_id] = [
                        CommittedVersion(
                            object_id=object_id,
                            value=str(current.value),
                            version=int(current.version),
                            exists=bool(current.exists),
                            commit_epoch=epoch,
                            owner_tid="lazy-pin-baseline",
                            background=False,
                        )
                    ]
                self._pins[key] = epoch
                self._lazy_pins.add(key)
                self._materialized_pins.discard(key)
                self._transaction_states.setdefault(key, "active")
                return epoch
        finally:
            self._exit_snapshot()

    def read_at(self, epoch: int, object_id: str) -> SnapshotValue:
        key = str(object_id)
        with self._lock:
            versions = self._history.get(key, ())
            for entry in reversed(versions):
                if entry.commit_epoch <= int(epoch):
                    return SnapshotValue(entry.value, entry.version, entry.exists)
            return SnapshotValue("", 0, False)

    def read_many_at(
        self,
        epoch: int,
        object_ids: Iterable[str],
    ) -> dict[str, SnapshotValue]:
        keys = tuple(dict.fromkeys(str(value) for value in object_ids))
        with self._lock:
            result = {}
            for key in keys:
                versions = self._history.get(key, ())
                entry = next(
                    (
                        candidate
                        for candidate in reversed(versions)
                        if candidate.commit_epoch <= int(epoch)
                    ),
                    None,
                )
                result[key] = (
                    SnapshotValue(entry.value, entry.version, entry.exists)
                    if entry is not None
                    else SnapshotValue("", 0, False)
                )
            return result

    def read_committed(self, object_id: str) -> SnapshotValue:
        key = str(object_id)
        with self._lock:
            current = self.store.get(key)
            return SnapshotValue(
                str(current.value), int(current.version), bool(current.exists)
            )

    def can_lock_pinned_version(
        self,
        epoch: int,
        object_id: str,
        observed_version: int,
        *,
        tid: str = "",
    ) -> bool:
        """Allow historical locking only across deferred background commits."""
        key = str(object_id)
        with self._lock:
            if str(tid) not in self._materialized_pins:
                known = next(
                    (
                        entry
                        for entry in reversed(self._history.get(key, ()))
                        if entry.commit_epoch <= int(epoch)
                    ),
                    None,
                )
                if known is not None and known.version != int(observed_version):
                    return False
            return int(self._last_agent_commit_epoch.get(key, 0)) <= int(epoch)

    def prepare(self, tid: str, writes: Iterable[tuple[str, str]]) -> None:
        private = tuple((str(object_id), str(value)) for object_id, value in writes)
        with self._lock:
            self._private[str(tid)] = private
            self._transaction_states[str(tid)] = "prepared"
            self._diagnostics["private_prepares"] += 1

    def atomic_publish(
        self,
        tid: str,
        writes: Iterable[tuple[str, str]],
        install: Callable[[], bool],
        *,
        background: bool = False,
        published_version: Callable[[str], int] | None = None,
    ) -> bool:
        key = str(tid)
        write_rows = tuple((str(object_id), str(value)) for object_id, value in writes)
        self.prepare(key, write_rows)
        self._enter_publication()
        try:
            try:
                installed = bool(install())
            except BaseException:
                with self._lock:
                    self._discard_locked(key)
                raise
            with self._lock:
                if not installed:
                    self._discard_locked(key)
                    return False
                if write_rows:
                    self._epoch += 1
                    commit_epoch = self._epoch
                    preserve_history = bool(self._lazy_pins)
                    published_rows = write_rows if preserve_history or not background else ()
                    for object_id, value in published_rows:
                        if published_version is None:
                            current = self.store.get(object_id)
                            current_value = str(current.value)
                            current_version = int(current.version)
                            current_exists = bool(current.exists)
                        else:
                            current_value = str(value)
                            current_version = int(published_version(object_id))
                            current_exists = True
                        committed = CommittedVersion(
                            object_id=object_id,
                            value=current_value,
                            version=current_version,
                            exists=current_exists,
                            commit_epoch=commit_epoch,
                            owner_tid=key,
                            background=bool(background),
                        )
                        if preserve_history:
                            self._history.setdefault(object_id, []).append(committed)
                            self._dirty_objects.add(object_id)
                        else:
                            previous = self._history.get(object_id, ())
                            if len(previous) > 1:
                                self._diagnostics["gc_versions"] += len(previous) - 1
                            self._history[object_id] = [committed]
                        if not background:
                            self._last_agent_commit_epoch[object_id] = commit_epoch
                    self._diagnostics["atomic_publishes"] += 1
                    self._diagnostics["published_objects"] += len(write_rows)
                self._private.pop(key, None)
                self._transaction_states[key] = "committed"
                self._trim_states_locked()
                self._maybe_garbage_collect_locked()
                return True
        finally:
            self._exit_publication()

    def finish(self, tid: str, *, committed: bool) -> None:
        key = str(tid)
        with self._lock:
            self._pins.pop(key, None)
            self._materialized_pins.discard(key)
            self._lazy_pins.discard(key)
            if committed:
                self._transaction_states[key] = "committed"
                self._private.pop(key, None)
            else:
                self._discard_locked(key)
            self._trim_states_locked()
            self._maybe_garbage_collect_locked()

    def snapshot_diagnostics(self) -> dict[str, int]:
        with self._lock:
            return {
                **self._diagnostics,
                "current_epoch": self._epoch,
                "private_transactions": len(self._private),
                "pinned_transactions": len(self._pins),
                "materialized_pins": len(self._materialized_pins),
                "lazy_pins": len(self._lazy_pins),
                "history_versions": sum(len(rows) for rows in self._history.values()),
                "commit_table_entries": len(self._transaction_states),
            }

    def _discard_locked(self, tid: str) -> None:
        if self._private.pop(tid, None) is not None:
            self._diagnostics["private_discards"] += 1
        self._transaction_states[tid] = "aborted"

    def _maybe_garbage_collect_locked(self) -> None:
        self._operations_since_gc += 1
        if self._operations_since_gc < self._gc_interval:
            return
        self._operations_since_gc = 0
        safe_epoch = min(self._pins.values(), default=self._epoch)
        dirty = tuple(self._dirty_objects)
        self._dirty_objects.clear()
        for object_id in dirty:
            versions = self._history.get(object_id, ())
            if len(versions) <= 1:
                continue
            predecessor = 0
            for index, entry in enumerate(versions):
                if entry.commit_epoch <= safe_epoch:
                    predecessor = index
                else:
                    break
            keep_from = max(0, predecessor)
            if keep_from:
                self._diagnostics["gc_versions"] += keep_from
                self._history[object_id] = versions[keep_from:]

    def _trim_states_locked(self) -> None:
        excess = len(self._transaction_states) - self._max_commit_states
        if excess <= 0:
            return
        for tid in tuple(self._transaction_states):
            if excess <= 0:
                break
            if tid in self._pins or tid in self._private:
                continue
            self._transaction_states.pop(tid, None)
            excess -= 1

    def _enter_publication(self) -> None:
        with self._publication_condition:
            while self._snapshot_active or self._snapshot_waiters:
                self._publication_condition.wait()
            self._active_publishers += 1

    def _exit_publication(self) -> None:
        with self._publication_condition:
            self._active_publishers -= 1
            if self._active_publishers <= 0:
                self._active_publishers = 0
                self._publication_condition.notify_all()

    def _enter_snapshot(self) -> None:
        with self._publication_condition:
            self._snapshot_waiters += 1
            try:
                while self._snapshot_active or self._active_publishers:
                    self._publication_condition.wait()
                self._snapshot_active = True
            finally:
                self._snapshot_waiters -= 1

    def _exit_snapshot(self) -> None:
        with self._publication_condition:
            self._snapshot_active = False
            self._publication_condition.notify_all()

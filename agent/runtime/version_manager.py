"""Transaction-scoped private versions and atomic commit publication."""

from __future__ import annotations

import dataclasses
import threading
import time
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
        self._boundary_condition = threading.Condition(threading.RLock())
        self._boundary_sequence = 0
        self._active_publications: dict[int, frozenset[str]] = {}
        self._active_snapshots: dict[int, frozenset[str]] = {}
        self._boundary_waiters = 0
        self._epoch = 0
        self._history: dict[str, list[CommittedVersion]] = {}
        self._private: dict[str, tuple[tuple[str, str], ...]] = {}
        self._transaction_states: dict[str, str] = {}
        self._pins: dict[str, int] = {}
        self._pin_coverage: dict[str, frozenset[str]] = {}
        self._pin_object_counts: dict[str, int] = {}
        self._lazy_pin_object_counts: dict[str, int] = {}
        self._wildcard_pins = 0
        self._wildcard_lazy_pins = 0
        self._materialized_pins: set[str] = set()
        self._lazy_pins: set[str] = set()
        self._last_agent_commit_epoch: dict[str, int] = {}
        self._background_version_changes: dict[str, int] = {}
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
            "native_publish_attempts": 0,
            "native_publishes": 0,
            "native_publish_pin_fallbacks": 0,
            "native_publish_disjoint_pin_bypasses": 0,
            "read_only_bypasses": 0,
            "background_version_change_events": 0,
            "version_risk_read_locks": 0,
            "object_boundary_acquires": 0,
            "object_boundary_waits": 0,
            "pinned_read_guard_acquires": 0,
            "pinned_read_guard_conflicts": 0,
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

    def has_lazy_pins(self, object_ids: Iterable[str] | None = None) -> bool:
        with self._lock:
            if object_ids is None:
                return bool(self._lazy_pins)
            keys = frozenset(str(value) for value in object_ids)
            return bool(
                self._wildcard_lazy_pins
                or any(self._lazy_pin_object_counts.get(key, 0) for key in keys)
            )

    def has_active_pins(self) -> bool:
        with self._lock:
            return bool(self._pins)

    def note_read_only_bypass(self) -> None:
        with self._lock:
            self._diagnostics["read_only_bypasses"] += 1

    def top_background_changed(
        self,
        object_ids: Iterable[str],
        *,
        limit: int = 2,
        min_changes: int = 2,
        min_share: float = 0.005,
        min_total_changes: int = 32,
    ) -> tuple[str, ...]:
        """Return the most frequently background-published candidate keys."""
        keys = set(str(value) for value in object_ids)
        with self._lock:
            # The frequency model is deliberately retained across the warmup
            # boundary while measurement-only diagnostics are reset. Derive
            # the denominator from the retained model, not the reset counter.
            total_changes = sum(self._background_version_changes.values())
            if total_changes < max(1, int(min_total_changes)):
                return ()
            ranked = sorted(
                (
                    (int(self._background_version_changes.get(key, 0)), key)
                    for key in keys
                    if int(self._background_version_changes.get(key, 0))
                    >= max(1, int(min_changes))
                    and (
                        int(self._background_version_changes.get(key, 0))
                        / total_changes
                    )
                    >= max(0.0, float(min_share))
                ),
                key=lambda item: (-item[0], item[1]),
            )
            return tuple(key for _count, key in ranked[: max(0, int(limit))])

    def background_change_family_is_risky(
        self,
        *,
        prefix: str,
        suffix: str,
        min_family_changes: int = 2,
        min_total_changes: int = 32,
    ) -> bool:
        """Return whether a key family has enough background-change evidence."""
        _exact, family_changes, total_changes = self.background_change_evidence(
            "",
            prefix=prefix,
            suffix=suffix,
        )
        return (
            total_changes >= max(1, int(min_total_changes))
            and family_changes >= max(1, int(min_family_changes))
        )

    def background_change_evidence(
        self,
        object_id: str,
        *,
        prefix: str,
        suffix: str,
    ) -> tuple[int, int, int]:
        """Return exact, matching-family, and global background change counts."""
        key = str(object_id)
        family_prefix = str(prefix)
        family_suffix = str(suffix)
        with self._lock:
            exact_changes = int(self._background_version_changes.get(key, 0))
            family_changes = sum(
                int(count)
                for candidate, count in self._background_version_changes.items()
                if candidate.startswith(family_prefix)
                and candidate.endswith(family_suffix)
            )
            total_changes = sum(self._background_version_changes.values())
            return exact_changes, family_changes, total_changes

    def note_version_risk_read_lock(self) -> None:
        with self._lock:
            self._diagnostics["version_risk_read_locks"] += 1

    def snapshot_current(
        self,
        object_ids: Iterable[str],
    ) -> tuple[int, dict[str, SnapshotValue]]:
        """Materialize one multi-object snapshot at a publication boundary."""
        keys = tuple(dict.fromkeys(str(value) for value in object_ids))
        boundaries = self._enter_snapshot(keys)
        try:
            with self._lock:
                epoch = int(self._epoch)
                result = {}
                for key in keys:
                    current = self.store.get(key)
                    result[key] = SnapshotValue(
                        value=str(current.value),
                        version=int(current.version),
                        exists=bool(current.exists),
                    )
                    # A native fast publish deliberately avoids maintaining
                    # per-object history while no transaction is pinned. The
                    # next snapshot boundary imports that current value as its
                    # baseline before the new pin becomes visible.
                    versions = self._history.get(key, ())
                    if not versions or versions[-1].version != int(current.version):
                        self._history[key] = [
                            CommittedVersion(
                                object_id=key,
                                value=str(current.value),
                                version=int(current.version),
                                exists=bool(current.exists),
                                commit_epoch=epoch,
                                owner_tid="native-fast-baseline",
                                background=True,
                            )
                        ]
                return epoch, result
        finally:
            self._exit_snapshot(boundaries)

    def snapshot_and_pin(
        self,
        tid: str,
        object_ids: Iterable[str],
        *,
        materialized: bool = True,
        coverage_object_ids: Iterable[str] | None = None,
    ) -> tuple[int, dict[str, SnapshotValue]]:
        """Capture a snapshot and expose its pin as one admission operation.

        A publisher that intersects this coverage must observe either the
        active snapshot reservation or the installed pin.  There is no gap in
        which it can native-publish without retaining the captured version.
        """
        pin_tid = str(tid)
        keys = tuple(dict.fromkeys(str(value) for value in object_ids))
        coverage = tuple(
            dict.fromkeys(
                str(value)
                for value in (
                    coverage_object_ids
                    if coverage_object_ids is not None
                    else keys
                )
            )
        )
        boundary = self._enter_snapshot(coverage)
        try:
            with self._lock:
                epoch = int(self._epoch)
                result: dict[str, SnapshotValue] = {}
                for key in keys:
                    current = self.store.get(key)
                    result[key] = SnapshotValue(
                        value=str(current.value),
                        version=int(current.version),
                        exists=bool(current.exists),
                    )
                    versions = self._history.get(key, ())
                    if not versions or versions[-1].version != int(current.version):
                        self._history[key] = [
                            CommittedVersion(
                                object_id=key,
                                value=str(current.value),
                                version=int(current.version),
                                exists=bool(current.exists),
                                commit_epoch=epoch,
                                owner_tid="native-fast-baseline",
                                background=True,
                            )
                        ]
                self._set_pin_locked(
                    pin_tid,
                    epoch,
                    frozenset(coverage),
                    materialized=materialized,
                )
                self._transaction_states.setdefault(pin_tid, "active")
                return epoch, result
        finally:
            self._exit_snapshot(boundary)

    def pin(
        self,
        tid: str,
        epoch: int,
        *,
        materialized: bool = False,
        object_ids: Iterable[str] = (),
    ) -> None:
        with self._lock:
            key = str(tid)
            self._set_pin_locked(
                key,
                min(int(epoch), self._epoch),
                frozenset(str(value) for value in object_ids),
                materialized=materialized,
            )
            self._transaction_states.setdefault(key, "active")

    def pin_lazy(self, tid: str, object_ids: Iterable[str]) -> int:
        """Atomically establish a full baseline for an unplanned transaction."""
        key = str(tid)
        keys = tuple(dict.fromkeys(str(value) for value in object_ids))
        boundaries = self._enter_snapshot(keys)
        try:
            with self._lock:
                epoch = int(self._epoch)
                for object_id in keys:
                    current = self.store.get(object_id)
                    versions = self._history.get(object_id, ())
                    if not versions or versions[-1].version != int(current.version):
                        self._history.setdefault(object_id, []).append(
                            CommittedVersion(
                                object_id=object_id,
                                value=str(current.value),
                                version=int(current.version),
                                exists=bool(current.exists),
                                commit_epoch=epoch,
                                owner_tid="lazy-pin-baseline",
                                background=False,
                            )
                        )
                self._set_pin_locked(
                    key,
                    epoch,
                    frozenset(keys),
                    materialized=False,
                )
                self._transaction_states.setdefault(key, "active")
                return epoch
        finally:
            self._exit_snapshot(boundaries)

    def read_at(self, epoch: int, object_id: str) -> SnapshotValue:
        key = str(object_id)
        with self._lock:
            versions = self._history.get(key, ())
            for entry in reversed(versions):
                if entry.commit_epoch <= int(epoch):
                    return SnapshotValue(entry.value, entry.version, entry.exists)
            return SnapshotValue("", 0, False)

    def history_debug(self, object_id: str, *, tid: str = "") -> dict[str, object]:
        key = str(object_id)
        pin_tid = str(tid)
        with self._lock:
            versions = tuple(self._history.get(key, ()))
            coverage = self._pin_coverage.get(pin_tid, frozenset())
            current = self.store.get(key)
            return {
                "object_id": key,
                "current_epoch": int(self._epoch),
                "current_version": int(current.version),
                "current_exists": bool(current.exists),
                "history": tuple(
                    (int(row.commit_epoch), int(row.version), bool(row.exists))
                    for row in versions
                ),
                "pin_epoch": self._pins.get(pin_tid),
                "pin_coverage_size": len(coverage),
                "pin_covers_object": key in coverage or not coverage,
                "materialized_pin": pin_tid in self._materialized_pins,
            }

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

    def enter_pinned_read_guard(
        self,
        tid: str,
        epoch: int,
        observed_versions: dict[str, int],
        write_object_ids: Iterable[str] = (),
    ) -> tuple[int, tuple[str, ...]]:
        """Reserve exact read coverage for the short commit interval.

        Background publications may have advanced the physical version while
        the pin retained the observed snapshot. Only an Agent publication
        after ``epoch`` invalidates that serialization point.
        """
        versions = {
            str(object_id): int(version)
            for object_id, version in observed_versions.items()
        }
        write_keys = {str(value) for value in write_object_ids}
        if not versions:
            return 0, ()
        boundary = self._enter_publication(
            set(versions) | write_keys
        )
        conflicts: list[str] = []
        with self._lock:
            pin_tid = str(tid)
            if pin_tid not in self._pins:
                conflicts.extend(versions)
            else:
                coverage = self._pin_coverage.get(pin_tid, frozenset())
                if coverage:
                    conflicts.extend(set(versions) - set(coverage))
                for object_id, observed_version in versions.items():
                    if object_id in conflicts:
                        continue
                    known = next(
                        (
                            entry
                            for entry in reversed(self._history.get(object_id, ()))
                            if entry.commit_epoch <= int(epoch)
                        ),
                        None,
                    )
                    if (
                        known is not None
                        and int(known.version) != int(observed_version)
                    ) or int(self._last_agent_commit_epoch.get(object_id, 0)) > int(
                        epoch
                    ):
                        conflicts.append(object_id)
                    elif write_keys and int(self.store.get_version(object_id)) != int(
                        observed_version
                    ):
                        # A read-write transaction cannot be serialized at its
                        # old snapshot without tracking incoming background
                        # anti-dependencies. Keep its serialization point at
                        # commit by validating every pinned read against the
                        # current committed version.
                        conflicts.append(object_id)
            if conflicts:
                self._diagnostics["pinned_read_guard_conflicts"] += len(conflicts)
            else:
                self._diagnostics["pinned_read_guard_acquires"] += 1
        if conflicts:
            self._exit_publication(boundary)
            return 0, tuple(sorted(set(conflicts)))
        return boundary, ()

    def exit_pinned_read_guard(self, boundary: int) -> None:
        if int(boundary) > 0:
            self._exit_publication(int(boundary))

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
        background_sample_rate: int = 1,
        publication_boundary_held: bool = False,
        published_version: Callable[[str], int] | None = None,
        timing_ms: dict[str, float] | None = None,
        private_prepared: bool = False,
    ) -> bool:
        key = str(tid)
        write_rows = tuple((str(object_id), str(value)) for object_id, value in writes)
        if not write_rows:
            started = time.perf_counter()
            installed = bool(install())
            if timing_ms is not None:
                timing_ms["publish"] = timing_ms.get("publish", 0.0) + (
                    time.perf_counter() - started
                ) * 1000.0
            with self._lock:
                self._diagnostics["read_only_bypasses"] += 1
            return installed
        if private_prepared:
            with self._lock:
                if (
                    self._transaction_states.get(key) != "prepared"
                    or self._private.get(key) != write_rows
                ):
                    self._diagnostics["missing_private_versions"] += 1
                    return False
        else:
            self.prepare(key, write_rows)
        boundaries = (
            0
            if publication_boundary_held
            else self._enter_publication(
                object_id for object_id, _value in write_rows
            )
        )
        try:
            try:
                installed = bool(install())
            except BaseException:
                with self._lock:
                    self._discard_locked(key)
                raise
            publish_started = time.perf_counter()
            with self._lock:
                if not installed:
                    self._discard_locked(key)
                    return False
                if write_rows:
                    self._epoch += 1
                    commit_epoch = self._epoch
                    if background and int(background_sample_rate) > 0:
                        self._note_background_version_changes_locked(
                            (object_id for object_id, _value in write_rows),
                            weight=max(1, int(background_sample_rate)),
                        )
                    for object_id, value in write_rows:
                        preserve_history = bool(
                            self._wildcard_lazy_pins
                            or self._lazy_pin_object_counts.get(object_id, 0)
                        )
                        if background and not preserve_history:
                            continue
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
                gc_started = time.perf_counter()
                self._maybe_garbage_collect_locked()
                gc_elapsed_ms = (time.perf_counter() - gc_started) * 1000.0
                if timing_ms is not None:
                    timing_ms["gc"] = timing_ms.get("gc", 0.0) + gc_elapsed_ms
                    timing_ms["publish"] = timing_ms.get("publish", 0.0) + (
                        time.perf_counter() - publish_started
                    ) * 1000.0 - gc_elapsed_ms
                return True
        finally:
            if boundaries:
                self._exit_publication(boundaries)

    def try_native_publish(
        self,
        object_ids: Iterable[str],
        install: Callable[[], Any],
        *,
        background: bool = False,
        background_sample_rate: int = 1,
        timing_ms: dict[str, float] | None = None,
    ) -> tuple[bool, Any]:
        """Use the native atomic batch when no old snapshot needs retention.

        The lightweight publication boundary is still required so a new
        multi-object snapshot cannot interleave with the native batch. No
        private version, commit record, history row, or GC work is created.
        """
        started = time.perf_counter()
        install_elapsed_ms = 0.0
        keys = frozenset(str(value) for value in object_ids)
        admitted, boundary = self._try_enter_native_publication(keys)
        if not admitted:
            if timing_ms is not None:
                timing_ms["publish"] = timing_ms.get("publish", 0.0) + (
                    time.perf_counter() - started
                ) * 1000.0
            return False, False
        committed = False
        try:
            install_started = time.perf_counter()
            result = install()
            install_elapsed_ms = (time.perf_counter() - install_started) * 1000.0
            committed = bool(getattr(result, "committed", result))
            return True, result
        finally:
            self._finish_native_publication(
                boundary,
                keys,
                committed=committed,
                background=background,
                background_sample_rate=background_sample_rate,
            )
            if timing_ms is not None:
                timing_ms["publish"] = timing_ms.get("publish", 0.0) + max(
                    0.0,
                    (time.perf_counter() - started) * 1000.0 - install_elapsed_ms,
                )

    def finish(self, tid: str, *, committed: bool) -> None:
        key = str(tid)
        with self._lock:
            if (
                key not in self._pins
                and key not in self._private
                and key not in self._transaction_states
            ):
                return
            self._remove_pin_locked(key)
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
                "pin_coverage_objects": sum(
                    len(keys) for keys in self._pin_coverage.values()
                ),
                "background_changed_objects": len(
                    self._background_version_changes
                ),
                "history_versions": sum(len(rows) for rows in self._history.values()),
                "commit_table_entries": len(self._transaction_states),
            }

    def reset_diagnostics(self) -> None:
        """Reset event counters while preserving versions and predictor state."""
        with self._lock:
            for key in self._diagnostics:
                self._diagnostics[key] = 0

    def _discard_locked(self, tid: str) -> None:
        if self._private.pop(tid, None) is not None:
            self._diagnostics["private_discards"] += 1
        self._transaction_states[tid] = "aborted"

    def _coverage_overlaps_locked(
        self,
        tid: str,
        object_ids: Iterable[str],
    ) -> bool:
        """Return whether a pin may observe any object in ``object_ids``.

        Empty coverage is retained as a conservative wildcard for callers that
        cannot supply a planned/observed set. Normal paper-runtime pins always
        provide their planned coverage.
        """
        coverage = self._pin_coverage.get(str(tid), frozenset())
        if not coverage:
            return True
        return not coverage.isdisjoint(str(value) for value in object_ids)

    def _set_pin_locked(
        self,
        tid: str,
        epoch: int,
        coverage: frozenset[str],
        *,
        materialized: bool,
    ) -> None:
        key = str(tid)
        self._remove_pin_locked(key)
        self._pins[key] = int(epoch)
        self._pin_coverage[key] = coverage
        if coverage:
            for object_id in coverage:
                self._pin_object_counts[object_id] = (
                    self._pin_object_counts.get(object_id, 0) + 1
                )
        else:
            self._wildcard_pins += 1
        if materialized:
            self._materialized_pins.add(key)
            return
        self._lazy_pins.add(key)
        if coverage:
            for object_id in coverage:
                self._lazy_pin_object_counts[object_id] = (
                    self._lazy_pin_object_counts.get(object_id, 0) + 1
                )
        else:
            self._wildcard_lazy_pins += 1

    def _remove_pin_locked(self, tid: str) -> None:
        key = str(tid)
        if key not in self._pins:
            return
        coverage = self._pin_coverage.get(key, frozenset())
        if coverage:
            for object_id in coverage:
                count = self._pin_object_counts.get(object_id, 0) - 1
                if count > 0:
                    self._pin_object_counts[object_id] = count
                else:
                    self._pin_object_counts.pop(object_id, None)
        else:
            self._wildcard_pins = max(0, self._wildcard_pins - 1)
        if key in self._lazy_pins:
            if coverage:
                for object_id in coverage:
                    count = self._lazy_pin_object_counts.get(object_id, 0) - 1
                    if count > 0:
                        self._lazy_pin_object_counts[object_id] = count
                    else:
                        self._lazy_pin_object_counts.pop(object_id, None)
            else:
                self._wildcard_lazy_pins = max(
                    0, self._wildcard_lazy_pins - 1
                )
        self._pins.pop(key, None)
        self._pin_coverage.pop(key, None)
        self._materialized_pins.discard(key)
        self._lazy_pins.discard(key)

    def _note_background_version_changes_locked(
        self,
        object_ids: Iterable[str],
        *,
        weight: int = 1,
    ) -> None:
        increment = max(1, int(weight))
        for object_id in object_ids:
            key = str(object_id)
            self._background_version_changes[key] = (
                int(self._background_version_changes.get(key, 0)) + increment
            )
            self._diagnostics["background_version_change_events"] += increment

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

    def _enter_publication(
        self,
        object_ids: Iterable[str],
    ) -> int:
        return self._enter_boundary(object_ids, publication=True)

    def _exit_publication(self, boundary: int) -> None:
        self._exit_boundary(boundary, publication=True)

    def _enter_snapshot(
        self,
        object_ids: Iterable[str],
    ) -> int:
        return self._enter_boundary(object_ids, publication=False)

    def _exit_snapshot(self, boundary: int) -> None:
        self._exit_boundary(boundary, publication=False)

    def _enter_boundary(
        self,
        object_ids: Iterable[str],
        *,
        publication: bool,
    ) -> int:
        """Reserve one exact coverage set without one Python lock per key."""
        keys = frozenset(str(value) for value in object_ids)
        waited = False
        with self._boundary_condition:
            waiting_registered = False
            try:
                while True:
                    publication_overlap = any(
                        not keys.isdisjoint(active)
                        for active in self._active_publications.values()
                    )
                    snapshot_overlap = publication and any(
                        not keys.isdisjoint(active)
                        for active in self._active_snapshots.values()
                    )
                    if not publication_overlap and not snapshot_overlap:
                        break
                    waited = True
                    if not waiting_registered:
                        self._boundary_waiters += 1
                        waiting_registered = True
                    self._boundary_condition.wait()
            finally:
                if waiting_registered:
                    self._boundary_waiters -= 1
            self._boundary_sequence += 1
            boundary = self._boundary_sequence
            target = (
                self._active_publications if publication else self._active_snapshots
            )
            target[boundary] = keys
            self._diagnostics["object_boundary_acquires"] += len(keys)
            if waited:
                self._diagnostics["object_boundary_waits"] += 1
            return boundary

    def _exit_boundary(self, boundary: int, *, publication: bool) -> None:
        with self._boundary_condition:
            target = (
                self._active_publications if publication else self._active_snapshots
            )
            target.pop(int(boundary), None)
            if self._boundary_waiters:
                self._boundary_condition.notify_all()

    def _try_enter_native_publication(
        self,
        object_ids: frozenset[str],
    ) -> tuple[bool, int]:
        """Check pin coverage and reserve native publication in one latch."""
        keys = frozenset(str(value) for value in object_ids)
        waited = False
        with self._boundary_condition:
            self._diagnostics["native_publish_attempts"] += 1
            waiting_registered = False
            try:
                while any(
                    not keys.isdisjoint(active)
                    for active in self._active_publications.values()
                ) or any(
                    not keys.isdisjoint(active)
                    for active in self._active_snapshots.values()
                ):
                    waited = True
                    if not waiting_registered:
                        self._boundary_waiters += 1
                        waiting_registered = True
                    self._boundary_condition.wait()
            finally:
                if waiting_registered:
                    self._boundary_waiters -= 1
            with self._lock:
                if self._wildcard_pins or any(
                    self._pin_object_counts.get(key, 0) for key in keys
                ):
                    self._diagnostics["native_publish_pin_fallbacks"] += 1
                    if waited:
                        self._diagnostics["object_boundary_waits"] += 1
                    return False, 0
                if self._pins:
                    self._diagnostics["native_publish_disjoint_pin_bypasses"] += 1
            self._boundary_sequence += 1
            boundary = self._boundary_sequence
            self._active_publications[boundary] = keys
            self._diagnostics["object_boundary_acquires"] += len(keys)
            if waited:
                self._diagnostics["object_boundary_waits"] += 1
            return True, boundary

    def _finish_native_publication(
        self,
        boundary: int,
        object_ids: frozenset[str],
        *,
        committed: bool,
        background: bool,
        background_sample_rate: int,
    ) -> None:
        with self._boundary_condition:
            self._active_publications.pop(int(boundary), None)
            if self._boundary_waiters:
                self._boundary_condition.notify_all()
        if not committed:
            return
        sample_rate = int(background_sample_rate)
        if background and sample_rate <= 0:
            return
        weight = max(1, sample_rate) if background else 1
        with self._lock:
            self._diagnostics["native_publishes"] += weight
            if background:
                self._note_background_version_changes_locked(
                    object_ids,
                    weight=weight,
                )

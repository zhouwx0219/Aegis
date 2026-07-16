"""Paper-aligned read/write Wound-Wait lock manager for ATCC."""

from __future__ import annotations

import dataclasses
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Set

from agent.cc.locks import LockConflict
from .context import TransactionContext, TransactionStatus


@dataclasses.dataclass
class _ObjectMetadata:
    readers: Set[str] = dataclasses.field(default_factory=set)
    writer: str = ""
    committing: bool = False
    publishers: Dict[str, TransactionContext] = dataclasses.field(default_factory=dict)
    waiters: List["_WaitRequest"] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _WaitRequest:
    object_id: str
    context: TransactionContext
    mode: str
    sequence: int
    cancelled: bool = False
    waited_ms: float = 0.0
    contention_noted: bool = False

    @property
    def tid(self) -> str:
        return self.context.tid

    @property
    def order(self) -> tuple[int, int, str, int]:
        return (-int(self.context.priority), int(self.context.start_ts_ns), self.tid, self.sequence)


@dataclasses.dataclass(frozen=True)
class CommitAdmissionBlock:
    reason: str
    object_ids: tuple[str, ...]


class PaperATCCLockManager:
    """Object metadata and dynamic-priority Wound-Wait in one critical region."""

    def __init__(
        self,
        *,
        wound_callback: Callable[[TransactionContext, str], None] | None = None,
        priority_callback: Callable[[TransactionContext], int] | None = None,
        contention_callback: Callable[[str, str, float], None] | None = None,
        priority_reorder_threshold: int = 2,
    ):
        self._condition = threading.Condition(threading.RLock())
        self._publisher_lock = threading.RLock()
        self._objects: Dict[str, _ObjectMetadata] = defaultdict(_ObjectMetadata)
        self._contexts: Dict[str, TransactionContext] = {}
        self._publisher_reservations: Dict[
            str, tuple[TransactionContext, frozenset[str]]
        ] = {}
        self._requests_by_tid: Dict[str, List[_WaitRequest]] = defaultdict(list)
        self._sequence = 0
        self._waiter_count = 0
        self._background_admission_waiters = 0
        self._wound_callback = wound_callback
        self._priority_callback = priority_callback
        self._contention_callback = contention_callback
        self._priority_reorder_threshold = max(1, int(priority_reorder_threshold))
        self._diagnostics: Dict[str, float] = defaultdict(float)
        self._native_diagnostics_lock = threading.Lock()
        self._native_diagnostics: Dict[str, float] = defaultdict(float)
        self._lock_acquires_by_phase: Dict[str, int] = defaultdict(int)
        self._wound_events: List[dict[str, object]] = []

    def register(self, context: TransactionContext) -> None:
        with self._condition:
            if context.status in {
                TransactionStatus.COMMITTED,
                TransactionStatus.ABORTING,
                TransactionStatus.ABORTED,
            }:
                raise LockConflict(
                    f"terminal transaction cannot acquire a lock: {context.tid}",
                    (),
                    kind="lock-preempted",
                )
            existing = self._contexts.get(context.tid)
            if existing is not None and existing is not context:
                raise RuntimeError(f"duplicate live transaction id: {context.tid}")
            self._contexts[context.tid] = context

    def validate_and_rlock(
        self,
        object_id: str,
        context: TransactionContext,
        observed_version: int,
        current_version: Callable[[], int],
        *,
        timeout_s: float = 5.0,
    ) -> float:
        key = str(object_id)
        started = time.perf_counter()
        deadline = started + max(0.0, float(timeout_s))
        with self._condition:
            self.register(context)
            if int(current_version()) != int(observed_version):
                self._note_contention(key, "validation-failure")
                raise LockConflict(
                    f"retroactive validation failed on {key}",
                    (key,),
                    kind="version-conflict",
                )
            if key in context.held_read_locks or key in context.held_write_locks:
                return time.perf_counter() - started
            request = self._enqueue(key, context, "read")
            try:
                while True:
                    self._ensure_live(request, key)
                    meta = self._objects[key]
                    writer = self._live_context(meta.writer)
                    publishers = self._publishers_for_key_locked(
                        key, exclude_tid=context.tid
                    )
                    publisher = (
                        min(publishers, key=self._priority_order)
                        if publishers
                        else None
                    )
                    blocker = writer if writer is not None and writer.tid != context.tid else publisher
                    if blocker is None:
                        if self._is_front(meta, request):
                            if int(current_version()) != int(observed_version):
                                self._note_contention(key, "validation-failure")
                                raise LockConflict(
                                    f"retroactive validation failed on {key}",
                                    (key,),
                                    kind="version-conflict",
                                )
                            meta.readers.add(context.tid)
                            context.held_read_locks.add(key)
                            self._note_acquired(context)
                            self._diagnostics["read_lock_acquires"] += 1
                            self._lock_acquires_by_phase[f"{context.phase.value}:read"] += 1
                            self._dequeue(meta, request)
                            return time.perf_counter() - started
                    elif self._can_wound(context, blocker, meta):
                        self._wound(
                            blocker,
                            f"wounded by {context.tid} while acquiring read lock on {key}",
                            object_id=key,
                            requester=context,
                        )
                        if meta.writer == blocker.tid:
                            meta.writer = ""
                            meta.committing = False
                        continue
                    self._wait(request, context, deadline, key, blockers=(blocker,))
            finally:
                self._dequeue(self._objects[key], request)

    def wlock(
        self,
        object_id: str,
        context: TransactionContext,
        *,
        timeout_s: float = 5.0,
    ) -> float:
        key = str(object_id)
        started = time.perf_counter()
        deadline = started + max(0.0, float(timeout_s))
        with self._condition:
            self.register(context)
            if key in context.held_write_locks:
                return time.perf_counter() - started
            request = self._enqueue(key, context, "write")
            try:
                while True:
                    self._ensure_live(request, key)
                    meta = self._objects[key]
                    blockers = self._write_blockers(key, meta, context)
                    non_woundable = [
                        blocker for blocker in blockers
                        if not self._can_wound(context, blocker, meta)
                    ]
                    if not blockers and self._is_front(meta, request):
                        meta.writer = context.tid
                        meta.readers.discard(context.tid)
                        context.held_read_locks.discard(key)
                        context.held_write_locks.add(key)
                        self._note_acquired(context)
                        self._diagnostics["write_lock_acquires"] += 1
                        self._lock_acquires_by_phase[f"{context.phase.value}:write"] += 1
                        self._dequeue(meta, request)
                        return time.perf_counter() - started
                    if blockers and not non_woundable:
                        for blocker in blockers:
                            self._wound(
                                blocker,
                                f"wounded by {context.tid} while acquiring write lock on {key}",
                                object_id=key,
                                requester=context,
                            )
                            meta.readers.discard(blocker.tid)
                            if meta.writer == blocker.tid:
                                meta.writer = ""
                                meta.committing = False
                        continue
                    self._wait(
                        request,
                        context,
                        deadline,
                        key,
                        blockers=non_woundable or blockers,
                    )
            finally:
                self._dequeue(self._objects[key], request)

    def validate_and_wlock(
        self,
        object_id: str,
        context: TransactionContext,
        observed_version: int,
        current_version: Callable[[], int],
        *,
        timeout_s: float = 5.0,
    ) -> float:
        """Atomically validate an observed version and acquire its write lock."""
        key = str(object_id)
        started = time.perf_counter()
        deadline = started + max(0.0, float(timeout_s))
        with self._condition:
            self.register(context)
            if int(current_version()) != int(observed_version):
                self._note_contention(key, "validation-failure")
                raise LockConflict(
                    f"retroactive validation failed on {key}",
                    (key,),
                    kind="version-conflict",
                )
            if key in context.held_write_locks:
                return time.perf_counter() - started
            request = self._enqueue(key, context, "write")
            try:
                while True:
                    self._ensure_live(request, key)
                    meta = self._objects[key]
                    blockers = self._write_blockers(key, meta, context)
                    non_woundable = [
                        blocker
                        for blocker in blockers
                        if not self._can_wound(context, blocker, meta)
                    ]
                    if not blockers and self._is_front(meta, request):
                        if int(current_version()) != int(observed_version):
                            self._note_contention(key, "validation-failure")
                            raise LockConflict(
                                f"retroactive validation failed on {key}",
                                (key,),
                                kind="version-conflict",
                            )
                        meta.writer = context.tid
                        meta.readers.discard(context.tid)
                        context.held_read_locks.discard(key)
                        context.held_write_locks.add(key)
                        self._note_acquired(context)
                        self._diagnostics["write_lock_acquires"] += 1
                        self._lock_acquires_by_phase[
                            f"{context.phase.value}:write"
                        ] += 1
                        self._dequeue(meta, request)
                        return time.perf_counter() - started
                    if blockers and not non_woundable:
                        for blocker in blockers:
                            self._wound(
                                blocker,
                                f"wounded by {context.tid} while acquiring write lock on {key}",
                                object_id=key,
                                requester=context,
                            )
                            meta.readers.discard(blocker.tid)
                            if meta.writer == blocker.tid:
                                meta.writer = ""
                                meta.committing = False
                        continue
                    self._wait(
                        request,
                        context,
                        deadline,
                        key,
                        blockers=non_woundable or blockers,
                    )
            finally:
                self._dequeue(self._objects[key], request)

    def acquire_write_set(
        self,
        object_ids: Iterable[str],
        context: TransactionContext,
        *,
        timeout_s: float = 5.0,
    ) -> float:
        keys = sorted(
            set(str(value) for value in object_ids) - context.held_write_locks
        )
        if not keys:
            return 0.0
        started = time.perf_counter()
        with self._condition:
            self.register(context)
            uncontended = all(
                not self._write_blockers(key, self._objects[key], context)
                and not any(self._request_live(request) for request in self._objects[key].waiters)
                for key in keys
            )
            if uncontended:
                for key in keys:
                    meta = self._objects[key]
                    meta.writer = context.tid
                    meta.readers.discard(context.tid)
                    context.held_read_locks.discard(key)
                    context.held_write_locks.add(key)
                    self._diagnostics["write_lock_acquires"] += 1
                    self._lock_acquires_by_phase[f"{context.phase.value}:write"] += 1
                self._note_acquired(context)
                return time.perf_counter() - started

            # Reserve the whole write set before waiting.  Acquiring keys one
            # at a time lets a transaction retain an arbitrary prefix while
            # it waits for the next hot object.  Those partial owners form a
            # convoy and make unrelated background publishers repeatedly
            # abort.  One request per key provides a stable, all-or-nothing
            # admission point: a writer either owns every key or none of the
            # newly requested keys.
            deadline = started + max(0.0, float(timeout_s))
            requests = [self._enqueue(key, context, "write") for key in keys]
            primary = requests[0]
            try:
                while True:
                    for request in requests:
                        self._ensure_live(request, request.object_id)

                    blockers_by_tid: dict[str, TransactionContext] = {}
                    blockers_by_key: dict[str, list[TransactionContext]] = {}
                    for request in requests:
                        meta = self._objects[request.object_id]
                        blockers = self._write_blockers(
                            request.object_id, meta, context
                        )
                        blockers_by_key[request.object_id] = blockers
                        for blocker in blockers:
                            blockers_by_tid[blocker.tid] = blocker

                    at_front = all(
                        self._is_front(self._objects[request.object_id], request)
                        for request in requests
                    )
                    if not blockers_by_tid and at_front:
                        for request in requests:
                            key = request.object_id
                            meta = self._objects[key]
                            meta.writer = context.tid
                            meta.readers.discard(context.tid)
                            context.held_read_locks.discard(key)
                            context.held_write_locks.add(key)
                            self._diagnostics["write_lock_acquires"] += 1
                            self._lock_acquires_by_phase[
                                f"{context.phase.value}:write"
                            ] += 1
                        self._note_acquired(context)
                        self._diagnostics["write_set_atomic_admissions"] += 1
                        return time.perf_counter() - started

                    woundable = bool(blockers_by_tid)
                    if woundable:
                        for object_id, blockers in blockers_by_key.items():
                            meta = self._objects[object_id]
                            if any(
                                not self._can_wound(context, blocker, meta)
                                for blocker in blockers
                            ):
                                woundable = False
                                break
                    if woundable:
                        wounded: set[str] = set()
                        for object_id, blockers in blockers_by_key.items():
                            meta = self._objects[object_id]
                            for blocker in blockers:
                                if blocker.tid in wounded:
                                    continue
                                wounded.add(blocker.tid)
                                self._wound(
                                    blocker,
                                    f"wounded by {context.tid} during write-set admission",
                                    object_id=object_id,
                                    requester=context,
                                )
                                meta.readers.discard(blocker.tid)
                                if meta.writer == blocker.tid:
                                    meta.writer = ""
                                    meta.committing = False
                        continue

                    self._wait(
                        primary,
                        context,
                        deadline,
                        primary.object_id,
                        blockers=tuple(blockers_by_tid.values()),
                    )
            finally:
                for request in requests:
                    self._dequeue(self._objects[request.object_id], request)

    def try_uncontended_background_publish(
        self,
        object_ids: Iterable[str],
        context: TransactionContext,
        publish: Callable[[], Any],
        *,
        allow_reader_bypass: bool = False,
        timing_callback: Callable[[float], None] | None = None,
    ) -> tuple[bool, Any]:
        """Publish buffered backend writes without materializing object locks.

        A per-object committing intent is registered before the metadata latch
        is released. Only the explicitly optimized runtime may bypass pinned
        readers; strict paper-atcc preserves ordinary RLock/WLock semantics.
        Writers, publishers, and queued write upgrades force the slow path.
        """
        if not context.is_background:
            raise ValueError("background publish fast path requires a background transaction")
        admission_started = time.perf_counter()
        keys = tuple(sorted(set(str(value) for value in object_ids)))
        key_coverage = frozenset(keys)
        with self._condition:
            if context.status != TransactionStatus.ACTIVE:
                if timing_callback is not None:
                    timing_callback((time.perf_counter() - admission_started) * 1000.0)
                return False, None
            publisher_conflicts = self._overlapping_publisher_keys_locked(
                key_coverage, exclude_tid=context.tid
            )
            if publisher_conflicts:
                self._note_background_publish_fallback_locked("active_writer")
                if timing_callback is not None:
                    timing_callback((time.perf_counter() - admission_started) * 1000.0)
                return False, CommitAdmissionBlock(
                    "active_publisher", publisher_conflicts
                )
            for key in keys:
                meta = self._objects.get(key)
                if meta is None:
                    continue
                writer = self._live_context(meta.writer)
                if writer is not None and writer.tid == context.tid:
                    writer = None
                readers = tuple(
                    reader
                    for tid in meta.readers
                    if tid != context.tid
                    and (reader := self._live_context(tid)) is not None
                    and (
                        not allow_reader_bypass
                        or reader.status != TransactionStatus.ACTIVE
                        or bool(reader.planned_write_targets)
                        or key in reader.retry_conflict_read_targets
                        or key in reader.retry_conflict_write_targets
                        or key in reader.hot_read_targets
                    )
                )
                if writer is not None and (
                    writer.status == TransactionStatus.COMMITTING
                    or meta.committing
                ):
                    self._note_background_publish_fallback_locked("commit_latch")
                    if timing_callback is not None:
                        timing_callback((time.perf_counter() - admission_started) * 1000.0)
                    return False, CommitAdmissionBlock("commit_latch", (key,))
                if writer is not None:
                    self._note_background_publish_fallback_locked("active_writer")
                    if timing_callback is not None:
                        timing_callback((time.perf_counter() - admission_started) * 1000.0)
                    return False, CommitAdmissionBlock("active_writer", (key,))
                if readers:
                    self._note_background_publish_fallback_locked("active_reader")
                    if timing_callback is not None:
                        timing_callback((time.perf_counter() - admission_started) * 1000.0)
                    return False, CommitAdmissionBlock("active_reader", (key,))
            context.transition(TransactionStatus.COMMITTING)
            with self._publisher_lock:
                self._publisher_reservations[context.tid] = (
                    context,
                    key_coverage,
                )
            self._diagnostics["background_fast_publishes"] += 1
        if timing_callback is not None:
            timing_callback((time.perf_counter() - admission_started) * 1000.0)
        succeeded = False
        try:
            result = publish()
            succeeded = bool(getattr(result, "committed", True))
            return True, result
        finally:
            with self._publisher_lock:
                self._publisher_reservations.pop(context.tid, None)
            if self._waiter_count or self._background_admission_waiters:
                with self._condition:
                    self._condition.notify_all()
            if not succeeded:
                with self._condition:
                    self._diagnostics["background_fast_publish_failures"] += 1

    def wait_for_background_admission(
        self,
        object_ids: Iterable[str],
        *,
        timeout_s: float = 0.002,
        allow_reader_bypass: bool = False,
    ) -> bool:
        """Wait briefly for covered keys to become writable."""
        keys = tuple(sorted(set(str(value) for value in object_ids)))
        if not keys:
            return True
        started = time.perf_counter()
        deadline = started + max(0.0, float(timeout_s))
        with self._condition:
            self._background_admission_waiters += 1
            self._diagnostics["background_publisher_queue_events"] += 1
            try:
                while True:
                    blocked = any(
                        self._background_admission_blocked_locked(
                            key,
                            allow_reader_bypass=allow_reader_bypass,
                        )
                        for key in keys
                    )
                    if not blocked:
                        waited_ms = (time.perf_counter() - started) * 1000.0
                        self._diagnostics[
                            "background_publisher_queue_wait_ms"
                        ] += waited_ms
                        return True
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        self._diagnostics[
                            "background_publisher_queue_timeouts"
                        ] += 1
                        waited_ms = (time.perf_counter() - started) * 1000.0
                        self._diagnostics[
                            "background_publisher_queue_wait_ms"
                        ] += waited_ms
                        return False
                    self._condition.wait(timeout=min(0.005, remaining))
            finally:
                self._background_admission_waiters -= 1

    def wait_for_background_publishers(
        self,
        object_ids: Iterable[str],
        *,
        timeout_s: float = 0.002,
    ) -> bool:
        """Compatibility alias for condition-aware publisher waiting."""
        return self.wait_for_background_admission(
            object_ids,
            timeout_s=timeout_s,
        )

    def background_pre_admission_block(
        self,
        object_ids: Iterable[str],
        *,
        allow_reader_bypass: bool = False,
    ) -> CommitAdmissionBlock | None:
        """Return an exact current blocker without reserving or waiting.

        This is a negative fast path for a background scheduler.  A clear
        result is only advisory—the normal atomic commit admission still
        closes races.  A blocked result lets the scheduler discard no state
        and move to another fresh transaction before doing avoidable work.
        """
        keys = tuple(sorted(set(str(value) for value in object_ids)))
        if not keys:
            return None
        with self._condition:
            conflicts = tuple(
                key
                for key in keys
                if self._background_admission_blocked_locked(
                    key,
                    allow_reader_bypass=allow_reader_bypass,
                )
            )
            if not conflicts:
                return None
            self._diagnostics["background_pre_admission_yields"] += 1
            self._diagnostics["background_pre_admission_objects"] += len(conflicts)
            return CommitAdmissionBlock("active_writer", conflicts)

    def try_uncontended_occ_publish(
        self,
        object_ids: Iterable[str],
        context: TransactionContext,
        publish: Callable[[], Any],
        *,
        timing_callback: Callable[[float], None] | None = None,
    ) -> tuple[bool, Any]:
        """Publish a pure-OCC write set when no object protection is active."""
        admission_started = time.perf_counter()
        keys = tuple(sorted(set(str(value) for value in object_ids)))
        key_coverage = frozenset(keys)
        with self._condition:
            if context.status != TransactionStatus.ACTIVE:
                if timing_callback is not None:
                    timing_callback((time.perf_counter() - admission_started) * 1000.0)
                return False, None
            conflicts = list(
                self._overlapping_publisher_keys_locked(
                    key_coverage, exclude_tid=context.tid
                )
            )
            for key in keys:
                meta = self._objects.get(key)
                if meta is None:
                    continue
                if meta.writer == context.tid:
                    continue
                readers = {
                    tid for tid in meta.readers if tid != context.tid
                }
                writer = self._live_context(meta.writer)
                if writer is not None and writer.tid == context.tid:
                    writer = None
                live_waiter = any(
                    self._request_live(request) for request in meta.waiters
                )
                if readers or writer is not None or live_waiter:
                    conflicts.append(key)
            if conflicts:
                conflicts = sorted(set(conflicts))
                self._diagnostics["commit_admission_conflicts"] += 1
                self._diagnostics["commit_admission_conflict_objects"] += len(conflicts)
                if timing_callback is not None:
                    timing_callback(
                        (time.perf_counter() - admission_started) * 1000.0
                    )
                return False, CommitAdmissionBlock(
                    "object_conflict",
                    tuple(conflicts),
                )
            context.transition(TransactionStatus.COMMITTING)
            with self._publisher_lock:
                self._publisher_reservations[context.tid] = (
                    context,
                    key_coverage,
                )
            self._diagnostics["occ_native_fast_publishes"] += 1
        if timing_callback is not None:
            timing_callback((time.perf_counter() - admission_started) * 1000.0)
        succeeded = False
        try:
            result = publish()
            succeeded = bool(getattr(result, "committed", True))
            return True, result
        finally:
            with self._publisher_lock:
                self._publisher_reservations.pop(context.tid, None)
            if self._waiter_count or self._background_admission_waiters:
                with self._condition:
                    self._condition.notify_all()
            if not succeeded:
                with self._condition:
                    self._diagnostics["occ_native_fast_publish_failures"] += 1

    def _background_admission_blocked_locked(
        self,
        object_id: str,
        *,
        allow_reader_bypass: bool = False,
    ) -> bool:
        key = str(object_id)
        if self._publishers_for_key_locked(key):
            return True
        meta = self._objects.get(key)
        if meta is None:
            return False
        writer = self._live_context(meta.writer)
        live_reader = any(
            reader is not None
            and (
                not allow_reader_bypass
                or reader.status != TransactionStatus.ACTIVE
                or bool(reader.planned_write_targets)
                or key in reader.retry_conflict_read_targets
                or key in reader.retry_conflict_write_targets
                or key in reader.hot_read_targets
            )
            for tid in meta.readers
            for reader in (self._live_context(tid),)
        )
        return bool(
            writer is not None
            or live_reader
        )

    def note_background_publish_fallback(
        self,
        reason: str,
        *,
        count_total: bool = True,
    ) -> None:
        with self._condition:
            self._note_background_publish_fallback_locked(
                reason,
                count_total=count_total,
            )

    def note_background_native_batch(self, outcome: str) -> None:
        """Record the outcome of the transaction-free background OCC path."""
        normalized = str(outcome).strip().lower().replace("-", "_")
        allowed = {
            "attempt",
            "commit",
            "read_only_commit",
            "validation_failure",
            "admission_fallback",
            "pin_fallback",
            "unsupported_fallback",
        }
        if normalized not in allowed:
            raise ValueError(f"unsupported native background batch outcome: {outcome}")
        # Native batches are the hottest background path. Their reporting
        # counters must not contend with Agent lock admission on _condition.
        with self._native_diagnostics_lock:
            self._native_diagnostics[f"background_native_batch_{normalized}s"] += 1

    def _note_background_publish_fallback_locked(
        self,
        reason: str,
        *,
        count_total: bool = True,
    ) -> None:
        normalized = str(reason).strip().lower().replace("-", "_")
        known = {
            "active_writer",
            "version_mismatch",
            "commit_latch",
            "missing_private_version",
            "multi_object_atomicity",
            "unsupported_operation",
        }
        if normalized not in known:
            normalized = "unsupported_operation"
        if count_total:
            self._diagnostics["background_publish_fallbacks"] += 1
        self._diagnostics[f"background_publish_fallback_{normalized}"] += 1

    def note_agent_blind_write_rebases(self, count: int) -> None:
        increment = max(0, int(count))
        if not increment:
            return
        with self._condition:
            self._diagnostics["agent_blind_write_rebases"] += increment

    def note_tpcc_exact_risk_wlock(self, *, family_fallback: bool = False) -> None:
        """Record one exact district dependency protected before its first read."""
        with self._condition:
            self._diagnostics["tpcc_exact_risk_wlocks"] += 1
            if family_fallback:
                self._diagnostics["tpcc_family_risk_wlocks"] += 1

    def note_tpcc_exact_guard_evidence(
        self,
        *,
        exact_changes: int,
        family_changes: int,
        total_changes: int,
        sufficient: bool,
    ) -> None:
        with self._condition:
            self._diagnostics["tpcc_exact_guard_checks"] += 1
            if not sufficient:
                self._diagnostics["tpcc_exact_guard_insufficient_evidence"] += 1
            for name, value in (
                ("exact", exact_changes),
                ("family", family_changes),
                ("total", total_changes),
            ):
                key = f"tpcc_exact_guard_max_{name}_changes"
                self._diagnostics[key] = max(
                    self._diagnostics.get(key, 0),
                    max(0, int(value)),
                )

    def enter_committing(self, context: TransactionContext) -> None:
        with self._condition:
            for object_id in context.held_write_locks:
                meta = self._objects[object_id]
                if meta.writer != context.tid:
                    raise RuntimeError(f"transaction does not own write lock: {object_id}")
                meta.committing = True
            self._condition.notify_all()

    def begin_committing(self, context: TransactionContext) -> bool:
        """Atomically make a lock owner non-preemptible before validation/install."""
        with self._condition:
            if context.status != TransactionStatus.ACTIVE:
                return False
            for object_id in context.held_write_locks:
                if self._objects[object_id].writer != context.tid:
                    return False
            context.transition(TransactionStatus.COMMITTING)
            for object_id in context.held_write_locks:
                self._objects[object_id].committing = True
            self._condition.notify_all()
            return True

    def update_priority(self, context: TransactionContext, priority: int) -> None:
        normalized = max(0, int(priority))
        if normalized == context.priority:
            return
        if not context.pending_request:
            context.priority = normalized
            context.priority_epoch += 1
            return
        if abs(normalized - int(context.priority)) < self._priority_reorder_threshold:
            return
        with self._condition:
            context.priority = normalized
            context.priority_epoch += 1
            self._diagnostics["priority_reorders"] += 1
            self._condition.notify_all()

    def release_all(self, context: TransactionContext) -> None:
        with self._condition:
            for object_id in tuple(context.held_read_locks):
                self._objects[object_id].readers.discard(context.tid)
            for object_id in tuple(context.held_write_locks):
                meta = self._objects[object_id]
                if meta.writer == context.tid:
                    meta.writer = ""
                    meta.committing = False
            context.held_read_locks.clear()
            context.held_write_locks.clear()
            if context.lock_acquired_ns:
                context.lock_hold_time_ms += max(
                    0.0, (time.monotonic_ns() - context.lock_acquired_ns) / 1_000_000.0
                )
                context.lock_acquired_ns = 0
            context.pending_request = ""
            with self._publisher_lock:
                self._publisher_reservations.pop(context.tid, None)
            for request in self._requests_by_tid.pop(context.tid, ()):
                request.cancelled = True
                meta = self._objects.get(request.object_id)
                if meta is not None and request in meta.waiters:
                    meta.waiters.remove(request)
                    self._waiter_count -= 1
            self._contexts.pop(context.tid, None)
            self._condition.notify_all()

    @staticmethod
    def _note_acquired(context: TransactionContext) -> None:
        if not context.lock_acquired_ns:
            context.lock_acquired_ns = time.monotonic_ns()

    def snapshot(self, object_id: str) -> dict[str, object]:
        with self._condition:
            key = str(object_id)
            meta = self._objects[key]
            publishers = tuple(
                sorted(
                    publisher.tid
                    for publisher in self._publishers_for_key_locked(key)
                )
            )
            return {
                "reader_count": len(meta.readers),
                "readers": tuple(sorted(meta.readers)),
                "writer": meta.writer,
                "committing": bool(meta.committing),
                "publisher": (
                    min(publishers) if publishers else ""
                ),
                "publishers": publishers,
                "waiters": tuple(request.tid for request in sorted(meta.waiters, key=lambda item: item.order)),
            }

    def global_waiter_count(self) -> int:
        return self._waiter_count

    def snapshot_diagnostics(self) -> dict[str, object]:
        with self._native_diagnostics_lock:
            native_diagnostics = dict(self._native_diagnostics)
        with self._condition:
            live_by_status: Dict[str, int] = defaultdict(int)
            for context in self._contexts.values():
                live_by_status[context.status.value] += 1
            return {
                "background_publish_fallbacks": self._diagnostics.get(
                    "background_publish_fallbacks", 0
                ),
                **{
                    f"background_publish_fallback_{reason}": self._diagnostics.get(
                        f"background_publish_fallback_{reason}", 0
                    )
                    for reason in (
                        "active_reader",
                        "active_writer",
                        "version_mismatch",
                        "commit_latch",
                        "missing_private_version",
                        "multi_object_atomicity",
                        "unsupported_operation",
                    )
                },
                **dict(self._diagnostics),
                **native_diagnostics,
                "lock_acquires_by_phase": dict(sorted(self._lock_acquires_by_phase.items())),
                "wound_events": tuple(dict(event) for event in self._wound_events),
                "current_waiters": self._waiter_count,
                "live_contexts": len(self._contexts),
                "live_contexts_by_status": dict(sorted(live_by_status.items())),
                "live_context_ids": tuple(sorted(self._contexts)),
            }

    def reset_diagnostics(self) -> None:
        """Start a new reporting interval without changing lock state."""
        with self._native_diagnostics_lock:
            self._native_diagnostics.clear()
        with self._condition:
            self._diagnostics.clear()
            self._lock_acquires_by_phase.clear()
            self._wound_events.clear()

    def has_foreign_committing_writer(
        self,
        object_id: str,
        context: TransactionContext,
    ) -> bool:
        return bool(
            self.foreign_committing_writer_targets((object_id,), context)
        )

    def foreign_committing_writer_targets(
        self,
        object_ids: Iterable[str],
        context: TransactionContext,
    ) -> tuple[str, ...]:
        """Snapshot conflicting committers for a read set under one latch."""
        keys = tuple(dict.fromkeys(str(value) for value in object_ids))
        conflicts: list[str] = []
        with self._condition:
            for key in keys:
                meta = self._objects[key]
                if (
                    (
                        meta.committing
                        and meta.writer
                        and meta.writer != context.tid
                    )
                    or self._publishers_for_key_locked(
                        key, exclude_tid=context.tid
                    )
                ):
                    conflicts.append(key)
        return tuple(conflicts)

    def has_object_protection(
        self,
        object_ids: Iterable[str],
        context: TransactionContext,
    ) -> bool:
        """Return whether a native OCC fast path would bypass active locks."""
        with self._condition:
            for object_id in set(str(value) for value in object_ids):
                meta = self._objects[str(object_id)]
                if any(tid != context.tid for tid in meta.readers):
                    return True
                if meta.writer and meta.writer != context.tid:
                    return True
                if any(
                    request.context is not context and self._request_live(request)
                    for request in meta.waiters
                ):
                    return True
            return False

    def _enqueue(self, key: str, context: TransactionContext, mode: str) -> _WaitRequest:
        request = _WaitRequest(
            object_id=key,
            context=context,
            mode=mode,
            sequence=self._sequence,
        )
        self._sequence += 1
        self._objects[key].waiters.append(request)
        self._requests_by_tid[context.tid].append(request)
        self._waiter_count += 1
        context.pending_request = key
        return request

    def _dequeue(self, meta: _ObjectMetadata, request: _WaitRequest) -> None:
        if request.waited_ms > 0.0 and not request.contention_noted:
            self._note_contention(request.object_id, "lock-wait", request.waited_ms)
            request.contention_noted = True
        if request in meta.waiters:
            meta.waiters.remove(request)
            self._waiter_count -= 1
        requests = self._requests_by_tid.get(request.tid)
        if requests is not None and request in requests:
            requests.remove(request)
            if not requests:
                self._requests_by_tid.pop(request.tid, None)
        if request.context.pending_request:
            request.context.pending_request = ""

    def _is_front(self, meta: _ObjectMetadata, request: _WaitRequest) -> bool:
        self._prune_waiters_locked(meta)
        live = [item for item in meta.waiters if self._request_live(item)]
        return not live or min(live, key=lambda item: item.order) is request

    def _prune_waiters_locked(self, meta: _ObjectMetadata) -> None:
        for request in tuple(meta.waiters):
            if not self._request_live(request):
                self._dequeue(meta, request)

    def _request_live(self, request: _WaitRequest) -> bool:
        return not request.cancelled and request.context.status not in {
            TransactionStatus.ABORTING,
            TransactionStatus.ABORTED,
            TransactionStatus.COMMITTED,
        }

    def _ensure_live(self, request: _WaitRequest, key: str) -> None:
        if not self._request_live(request):
            raise LockConflict(
                f"transaction aborted while waiting on {key}",
                (key,),
                kind="lock-preempted",
            )

    def _live_context(self, tid: str) -> TransactionContext | None:
        if not tid:
            return None
        # A committed/aborted context remains a real blocker until release_all
        # removes its lock metadata and unregisters it. Filtering by status here
        # loses wait attribution during commit publication and cleanup.
        return self._contexts.get(tid)

    def _write_blockers(
        self,
        object_id: str,
        meta: _ObjectMetadata,
        context: TransactionContext,
    ) -> List[TransactionContext]:
        tids = set(meta.readers)
        if meta.writer:
            tids.add(meta.writer)
        tids.discard(context.tid)
        blockers = [
            candidate
            for tid in sorted(tids)
            if (candidate := self._live_context(tid)) is not None
        ]
        blockers.extend(
            self._publishers_for_key_locked(
                object_id, exclude_tid=context.tid
            )
        )
        return blockers

    def _publishers_for_key_locked(
        self,
        object_id: str,
        *,
        exclude_tid: str = "",
    ) -> List[TransactionContext]:
        key = str(object_id)
        with self._publisher_lock:
            return [
                context
                for tid, (context, coverage) in self._publisher_reservations.items()
                if tid != exclude_tid and key in coverage
            ]

    def _overlapping_publisher_keys_locked(
        self,
        object_ids: frozenset[str],
        *,
        exclude_tid: str = "",
    ) -> tuple[str, ...]:
        conflicts: set[str] = set()
        with self._publisher_lock:
            for tid, (_context, coverage) in self._publisher_reservations.items():
                if tid != exclude_tid:
                    conflicts.update(object_ids & coverage)
        return tuple(sorted(conflicts))

    def _can_wound(
        self,
        requester: TransactionContext,
        holder: TransactionContext,
        meta: _ObjectMetadata,
    ) -> bool:
        if holder.status in {
            TransactionStatus.COMMITTING,
            TransactionStatus.COMMITTED,
            TransactionStatus.ABORTING,
            TransactionStatus.ABORTED,
        }:
            return False
        if meta.committing and meta.writer == holder.tid:
            return False
        return self._priority_order(requester) < self._priority_order(holder)

    @staticmethod
    def _priority_order(context: TransactionContext) -> tuple[int, int, str]:
        return (-int(context.priority), int(context.start_ts_ns), context.tid)

    def _wound(
        self,
        context: TransactionContext,
        reason: str,
        *,
        object_id: str,
        requester: TransactionContext | None = None,
    ) -> None:
        victim_status = context.status.value
        if not context.try_transition(
            TransactionStatus.ABORTING,
            from_statuses=(TransactionStatus.ACTIVE, TransactionStatus.WAITING),
        ):
            return
        requester_role = (
            "unknown"
            if requester is None
            else "background" if requester.is_background else "agent"
        )
        victim_role = "background" if context.is_background else "agent"
        relation = f"{requester_role}_to_{victim_role}"
        self._diagnostics[f"wounds_{relation}"] += 1
        event = {
            "requester_role": requester_role,
            "victim_role": victim_role,
            "object_id": str(object_id),
            "requester_phase": (
                requester.phase.value if requester is not None else "unknown"
            ),
            "victim_phase": context.phase.value,
            "requester_retry": (
                int(requester.retry_count) if requester is not None else -1
            ),
            "victim_retry": int(context.retry_count),
            "requester_priority": (
                int(requester.priority) if requester is not None else -1
            ),
            "victim_priority": int(context.priority),
            "victim_status": victim_status,
        }
        self._wound_events.append(event)
        if len(self._wound_events) > 1024:
            del self._wound_events[: len(self._wound_events) - 1024]
        self._note_contention(object_id, "wound")
        self._diagnostics["wounds"] += 1
        if requester is not None and not requester.is_background and context.is_background:
            requester.background_aborts_caused += 1
        elif requester is not None and not requester.is_background and not context.is_background:
            requester.agent_aborts_caused += 1
        if self._wound_callback is not None:
            self._wound_callback(context, reason)
        self.release_all(context)
        self._condition.notify_all()

    def _wait(
        self,
        request: _WaitRequest,
        context: TransactionContext,
        deadline: float,
        key: str,
        *,
        blockers: Iterable[TransactionContext] = (),
    ) -> None:
        if context.status == TransactionStatus.ACTIVE:
            context.transition(TransactionStatus.WAITING)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if context.status == TransactionStatus.WAITING:
                context.transition(TransactionStatus.ACTIVE)
            self._note_contention(key, "lock-timeout")
            self._diagnostics["lock_timeouts"] += 1
            raise LockConflict(f"lock timeout on {key}", (key,), kind="lock-timeout")
        started = time.perf_counter()
        first_wait = request.waited_ms <= 0.0
        self._condition.wait(timeout=min(0.005, remaining))
        waited_ms = (time.perf_counter() - started) * 1000.0
        request.waited_ms += waited_ms
        context.blocked_time_ms += waited_ms
        if first_wait:
            self._diagnostics["lock_wait_events"] += 1
            role = "background" if context.is_background else "agent"
            self._diagnostics[f"{role}_lock_wait_events"] += 1
        self._diagnostics["lock_wait_ms"] += waited_ms
        role = "background" if context.is_background else "agent"
        self._diagnostics[f"{role}_lock_wait_ms"] += waited_ms
        live_blockers = [blocker for blocker in blockers if blocker is not None]
        if live_blockers:
            share = waited_ms / len(live_blockers)
            for blocker in live_blockers:
                if not blocker.is_background:
                    if context.is_background:
                        blocker.background_blocked_ms_caused += share
                    else:
                        blocker.agent_blocked_ms_caused += share
        if self._priority_callback is not None:
            self.update_priority(context, self._priority_callback(context))
        if context.status == TransactionStatus.WAITING:
            context.transition(TransactionStatus.ACTIVE)

    def _note_contention(self, object_id: str, event: str, wait_ms: float = 0.0) -> None:
        if self._contention_callback is not None:
            self._contention_callback(str(object_id), str(event), max(0.0, float(wait_ms)))

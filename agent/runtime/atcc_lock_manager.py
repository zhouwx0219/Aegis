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
        self._objects: Dict[str, _ObjectMetadata] = defaultdict(_ObjectMetadata)
        self._contexts: Dict[str, TransactionContext] = {}
        self._requests_by_tid: Dict[str, List[_WaitRequest]] = defaultdict(list)
        self._sequence = 0
        self._waiter_count = 0
        self._wound_callback = wound_callback
        self._priority_callback = priority_callback
        self._contention_callback = contention_callback
        self._priority_reorder_threshold = max(1, int(priority_reorder_threshold))
        self._diagnostics: Dict[str, float] = defaultdict(float)
        self._lock_acquires_by_phase: Dict[str, int] = defaultdict(int)

    def register(self, context: TransactionContext) -> None:
        with self._condition:
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
                    publishers = [
                        candidate
                        for tid, candidate in meta.publishers.items()
                        if tid != context.tid
                    ]
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
                    blockers = self._write_blockers(meta, context)
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
                    blockers = self._write_blockers(meta, context)
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
                not self._write_blockers(self._objects[key], context)
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
        total = 0.0
        for object_id in keys:
            total += self.wlock(object_id, context, timeout_s=timeout_s)
        return total

    def try_uncontended_background_publish(
        self,
        object_ids: Iterable[str],
        context: TransactionContext,
        publish: Callable[[], Any],
    ) -> tuple[bool, Any]:
        """Publish buffered backend writes without materializing object locks.

        A per-object committing intent is registered before the metadata latch
        is released. Pinned readers do not block publication: their transaction
        snapshot retains the old committed version. Writers, publishers, and
        queued write upgrades still force the Wound-Wait path.
        """
        if not context.is_background:
            raise ValueError("background publish fast path requires a background transaction")
        keys = sorted(set(str(value) for value in object_ids))
        with self._condition:
            if context.status != TransactionStatus.ACTIVE:
                return False, None
            for key in keys:
                meta = self._objects[key]
                writer = self._live_context(meta.writer)
                if writer is not None and writer.tid == context.tid:
                    writer = None
                live_write_waiter = any(
                    request.mode == "write" and self._request_live(request)
                    for request in meta.waiters
                )
                if writer is not None and (
                    writer.status == TransactionStatus.COMMITTING
                    or meta.committing
                ):
                    self._note_background_publish_fallback_locked("commit_latch")
                    return False, None
                if writer is not None or live_write_waiter:
                    self._note_background_publish_fallback_locked("active_writer")
                    return False, None
            context.transition(TransactionStatus.COMMITTING)
            for key in keys:
                self._objects[key].publishers[context.tid] = context
            self._diagnostics["background_fast_publishes"] += 1
        try:
            result = publish()
            with self._condition:
                if not bool(getattr(result, "committed", True)):
                    self._diagnostics["background_fast_publish_failures"] += 1
            return True, result
        except BaseException:
            with self._condition:
                self._diagnostics["background_fast_publish_failures"] += 1
            raise
        finally:
            with self._condition:
                for key in keys:
                    meta = self._objects[key]
                    meta.publishers.pop(context.tid, None)
                self._condition.notify_all()

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
            meta = self._objects[str(object_id)]
            return {
                "reader_count": len(meta.readers),
                "readers": tuple(sorted(meta.readers)),
                "writer": meta.writer,
                "committing": bool(meta.committing),
                "publisher": (
                    min(meta.publishers) if meta.publishers else ""
                ),
                "publishers": tuple(sorted(meta.publishers)),
                "waiters": tuple(request.tid for request in sorted(meta.waiters, key=lambda item: item.order)),
            }

    def global_waiter_count(self) -> int:
        return self._waiter_count

    def snapshot_diagnostics(self) -> dict[str, object]:
        with self._condition:
            return {
                "background_publish_fallbacks": self._diagnostics.get(
                    "background_publish_fallbacks", 0
                ),
                **{
                    f"background_publish_fallback_{reason}": self._diagnostics.get(
                        f"background_publish_fallback_{reason}", 0
                    )
                    for reason in (
                        "active_writer",
                        "version_mismatch",
                        "commit_latch",
                        "missing_private_version",
                        "multi_object_atomicity",
                        "unsupported_operation",
                    )
                },
                **dict(self._diagnostics),
                "lock_acquires_by_phase": dict(sorted(self._lock_acquires_by_phase.items())),
                "current_waiters": self._waiter_count,
            }

    def has_foreign_committing_writer(
        self,
        object_id: str,
        context: TransactionContext,
    ) -> bool:
        with self._condition:
            meta = self._objects[str(object_id)]
            return bool(
                (
                    meta.committing
                    and meta.writer
                    and meta.writer != context.tid
                )
                or (
                    any(tid != context.tid for tid in meta.publishers)
                )
            )

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

    def _write_blockers(self, meta: _ObjectMetadata, context: TransactionContext) -> List[TransactionContext]:
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
            publisher
            for tid, publisher in sorted(meta.publishers.items())
            if tid != context.tid
        )
        return blockers

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
        self._note_contention(object_id, "wound")
        self._diagnostics["wounds"] += 1
        if requester is not None and not requester.is_background and context.is_background:
            requester.background_aborts_caused += 1
        elif requester is not None and not requester.is_background and not context.is_background:
            requester.agent_aborts_caused += 1
        if context.status in {TransactionStatus.ACTIVE, TransactionStatus.WAITING}:
            context.transition(TransactionStatus.ABORTING)
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

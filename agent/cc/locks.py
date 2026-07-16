"""Small lock tables used by agent-side CC strategies."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple


class LockConflict(RuntimeError):
    def __init__(
        self,
        reason: str,
        targets: Iterable[str],
        *,
        kind: str = "",
    ):
        super().__init__(reason)
        self.reason = str(reason)
        self.targets = tuple(str(target) for target in targets)
        self.kind = normalize_conflict_kind(kind or reason)


def normalize_conflict_kind(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized or normalized == "none":
        return "none"
    if "retroactive" in normalized or "version" in normalized or "atomic version" in normalized:
        return "version-conflict"
    if "wound" in normalized or "preempt" in normalized or "aborted while waiting" in normalized:
        return "lock-preempted"
    if "timeout" in normalized:
        return "lock-timeout"
    if "lock" in normalized or "reservation" in normalized:
        return "lock-conflict"
    return normalized


class ExclusiveLockTable:
    """Exclusive lock table with optional priority ordering."""

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._owners: Dict[str, int] = {}
        self._waiters: Dict[str, List[Tuple[int, float, int, int]]] = {}
        self._next_sequence = 0

    @contextlib.contextmanager
    def acquire(
        self,
        targets: Iterable[str],
        *,
        owner: Any,
        wait: bool = True,
        priority: int = 0,
    ):
        target_tuple = tuple(sorted(set(str(target) for target in targets)))
        owner_id = id(owner)
        owner_age = float(getattr(owner, "started_at", time.perf_counter()))
        acquired: List[str] = []
        try:
            for target in target_tuple:
                self._acquire_one(
                    target,
                    owner_id=owner_id,
                    owner_age=owner_age,
                    wait=wait,
                    priority=int(priority),
                )
                acquired.append(target)
            yield
        finally:
            self.release(acquired, owner_id=owner_id)

    def release(self, targets: Iterable[str], *, owner_id: int) -> None:
        with self._condition:
            for target in targets:
                if self._owners.get(str(target)) == int(owner_id):
                    self._owners.pop(str(target), None)
            self._condition.notify_all()

    def _acquire_one(
        self,
        target: str,
        *,
        owner_id: int,
        owner_age: float,
        wait: bool,
        priority: int,
    ) -> None:
        waiter = None
        try:
            while True:
                with self._condition:
                    if waiter is None and wait:
                        waiter = (-int(priority), float(owner_age), self._next_sequence, int(owner_id))
                        self._next_sequence += 1
                        self._waiters.setdefault(str(target), []).append(waiter)
                    owner = self._owners.get(str(target))
                    queue = self._waiters.get(str(target), [])
                    first = min(queue, default=waiter)
                    can_enter = waiter is None or waiter == first
                    if (owner is None or owner == int(owner_id)) and can_enter:
                        self._owners[str(target)] = int(owner_id)
                        if waiter in queue:
                            queue.remove(waiter)
                        if not queue:
                            self._waiters.pop(str(target), None)
                        self._condition.notify_all()
                        return
                    if not wait:
                        raise LockConflict(
                            f"no-wait lock conflict on {target}",
                            (target,),
                        )
                    self._condition.wait(timeout=0.001)
        except BaseException:
            if waiter is not None:
                with self._condition:
                    queue = self._waiters.get(str(target), [])
                    if waiter in queue:
                        queue.remove(waiter)
                    if not queue:
                        self._waiters.pop(str(target), None)
                    self._condition.notify_all()
            raise


class TwoPhaseLockTable:
    """S/X table with no-wait and wait-die policies."""

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._holders: Dict[str, List[Tuple[int, float, str]]] = {}

    @contextlib.contextmanager
    def acquire(
        self,
        targets: Iterable[str],
        *,
        owner: Any,
        mode: str,
        policy: str,
    ):
        target_tuple = tuple(sorted(set(str(target) for target in targets)))
        owner_id = id(owner)
        owner_age = float(getattr(owner, "started_at", time.perf_counter()))
        mode = self._normalize_mode(mode)
        policy = str(policy).strip().lower()
        acquired: List[Tuple[str, str]] = []
        try:
            for target in target_tuple:
                self._acquire_one(
                    target,
                    owner_id=owner_id,
                    owner_age=owner_age,
                    mode=mode,
                    policy=policy,
                )
                acquired.append((target, mode))
            yield
        finally:
            self.release(acquired, owner_id=owner_id)

    def release(self, acquired: Iterable[Tuple[str, str]], *, owner_id: int) -> None:
        with self._condition:
            for target, _mode in acquired:
                rows = [
                    row
                    for row in self._holders.get(str(target), [])
                    if int(row[0]) != int(owner_id)
                ]
                if rows:
                    self._holders[str(target)] = rows
                else:
                    self._holders.pop(str(target), None)
            self._condition.notify_all()

    def _acquire_one(
        self,
        target: str,
        *,
        owner_id: int,
        owner_age: float,
        mode: str,
        policy: str,
    ) -> None:
        while True:
            with self._condition:
                holders = self._holders.setdefault(str(target), [])
                if self._compatible(holders, owner_id=owner_id, mode=mode):
                    holders.append((int(owner_id), float(owner_age), str(mode)))
                    return
                if policy == "nowait":
                    raise LockConflict(
                        f"2pl-nowait conflict on {target}",
                        (target,),
                    )
                if policy != "wait-die":
                    raise ValueError(f"unsupported 2PL policy: {policy}")
                conflicting = [row for row in holders if int(row[0]) != int(owner_id)]
                if any(owner_age > float(holder_age) for _hid, holder_age, _mode in conflicting):
                    raise LockConflict(
                        f"2pl-wait-die aborted younger transaction on {target}",
                        (target,),
                    )
                self._condition.wait(timeout=0.001)

    @staticmethod
    def _compatible(
        holders: List[Tuple[int, float, str]],
        *,
        owner_id: int,
        mode: str,
    ) -> bool:
        others = [row for row in holders if int(row[0]) != int(owner_id)]
        if not others:
            return True
        if mode == "s":
            return all(holder_mode == "s" for _hid, _age, holder_mode in others)
        return False

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized in {"s", "shared", "read"}:
            return "s"
        if normalized in {"x", "exclusive", "write"}:
            return "x"
        raise ValueError(f"unsupported lock mode: {mode}")


class ReservationTable:
    """Soft hot-object reservations used by agent-side ATCC."""

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._owners: Dict[str, Tuple[int, float]] = {}
        self._reservation_owner_objects: Dict[int, Any] = {}
        self._writers: Dict[str, Dict[int, int]] = {}
        self._waiters: Dict[str, List[Tuple[int, float, int, int]]] = {}
        self._next_sequence = 0
        self._reservation_waiter_count = 0
        self._reservation_waiter_target_sizes: List[int] = []
        self._reservation_unique_targets: set[str] = set()
        self._reservation_all_or_nothing_failed_grant_checks = 0
        self._reservation_all_or_nothing_not_front_wait_s = 0.0
        self._reservation_front_queue_wait_s = 0.0
        self._reservation_owner_blocked_checks = 0
        self._reservation_writer_blocked_checks = 0
        self._reservation_blocked_target_checks = 0
        self._background_writer_waiter_blocked_checks = 0
        self._background_writer_waiter_blocked_targets = 0
        self._background_writer_reservation_blocked_checks = 0

    @contextlib.contextmanager
    def reserve(
        self,
        targets: Iterable[str],
        *,
        owner: Any,
        ttl_s: float = 5.0,
        wait: bool = True,
        timeout_s: float = 1.0,
        priority: int = 0,
    ):
        target_tuple = tuple(sorted(set(str(target) for target in targets if str(target))))
        owner_id = id(owner)
        owner_age = float(getattr(owner, "started_at", time.perf_counter()))
        deadline = time.perf_counter() + max(0.001, float(ttl_s))
        wait_deadline = time.perf_counter() + max(0.001, float(timeout_s))
        started_at = time.perf_counter()
        waiter = None
        state_started_at = started_at
        state_front_for_all: bool | None = None
        front_queue_wait_s = 0.0
        not_front_wait_s = 0.0

        def record_wait_state(front_for_all: bool) -> None:
            nonlocal state_started_at, state_front_for_all, front_queue_wait_s, not_front_wait_s
            now = time.perf_counter()
            if state_front_for_all is not None:
                elapsed_s = max(0.0, now - state_started_at)
                if state_front_for_all:
                    front_queue_wait_s += elapsed_s
                else:
                    not_front_wait_s += elapsed_s
            state_started_at = now
            state_front_for_all = bool(front_for_all)

        with self._condition:
            try:
                while True:
                    self._purge_expired_locked()
                    if waiter is None and wait and target_tuple:
                        waiter = (-int(priority), float(owner_age), self._next_sequence, int(owner_id))
                        self._next_sequence += 1
                        for target in target_tuple:
                            self._waiters.setdefault(target, []).append(waiter)
                        self._reservation_waiter_count += 1
                        self._reservation_waiter_target_sizes.append(len(target_tuple))
                        self._reservation_unique_targets.update(target_tuple)
                    reservation_blocked = self._blocked_reservations_locked(target_tuple, owner_id)
                    writer_blocked = self._blocked_writers_locked(target_tuple, owner_id)
                    blocked = list(reservation_blocked)
                    blocked.extend(writer_blocked)
                    first_for_all = self._first_waiter_for_all_locked(target_tuple, waiter)
                    if waiter is not None:
                        record_wait_state(first_for_all)
                    if reservation_blocked:
                        self._reservation_owner_blocked_checks += 1
                    if blocked:
                        self._reservation_blocked_target_checks += len(blocked)
                    if writer_blocked:
                        self._reservation_writer_blocked_checks += 1
                    if waiter is not None and len(target_tuple) > 1 and not first_for_all:
                        self._reservation_all_or_nothing_failed_grant_checks += 1
                    if not blocked and first_for_all:
                        break
                    if not wait or time.perf_counter() >= wait_deadline:
                        raise LockConflict("reservation conflict", tuple(blocked or target_tuple))
                    self._condition.wait(timeout=0.001)
                self._remove_waiter_locked(target_tuple, waiter)
                for target in target_tuple:
                    self._owners[target] = (owner_id, deadline)
                self._reservation_owner_objects[owner_id] = owner
                self._condition.notify_all()
                self._reservation_front_queue_wait_s += front_queue_wait_s
                self._reservation_all_or_nothing_not_front_wait_s += not_front_wait_s
                wait_s = time.perf_counter() - started_at
            except BaseException:
                self._reservation_front_queue_wait_s += front_queue_wait_s
                self._reservation_all_or_nothing_not_front_wait_s += not_front_wait_s
                self._remove_waiter_locked(target_tuple, waiter)
                self._condition.notify_all()
                raise
        try:
            yield wait_s
        finally:
            self.release(target_tuple, owner_id=owner_id)

    def release(self, targets: Iterable[str], *, owner_id: int) -> None:
        with self._condition:
            for target in targets:
                current = self._owners.get(str(target))
                if current is not None and current[0] == int(owner_id):
                    self._owners.pop(str(target), None)
            if not any(current[0] == int(owner_id) for current in self._owners.values()):
                self._reservation_owner_objects.pop(int(owner_id), None)
            self._condition.notify_all()

    def wait_for_write(
        self,
        targets: Iterable[str],
        *,
        owner: Any,
        wait: bool = True,
        timeout_s: float = 0.050,
    ) -> float:
        target_tuple = tuple(sorted(set(str(target) for target in targets if str(target))))
        owner_id = id(owner)
        started_at = time.perf_counter()
        with self._condition:
            while True:
                self._purge_expired_locked()
                reservation_blocked = self._blocked_reservations_locked(target_tuple, owner_id)
                waiter_blocked = self._blocked_waiters_locked(target_tuple, owner_id)
                blocked = list(reservation_blocked)
                blocked.extend(waiter_blocked)
                if waiter_blocked:
                    self._background_writer_waiter_blocked_checks += 1
                    self._background_writer_waiter_blocked_targets += len(waiter_blocked)
                if reservation_blocked:
                    self._background_writer_reservation_blocked_checks += 1
                    self._record_background_blocked_owners_locked(reservation_blocked)
                if not blocked:
                    return time.perf_counter() - started_at
                if not wait or time.perf_counter() - started_at >= float(timeout_s):
                    raise LockConflict("reservation blocked write", tuple(blocked))
                self._condition.wait(timeout=0.001)

    @contextlib.contextmanager
    def write_guard(
        self,
        targets: Iterable[str],
        *,
        owner: Any,
        wait: bool = True,
        timeout_s: float = 0.050,
        respect_waiters: bool = True,
    ):
        target_tuple = tuple(sorted(set(str(target) for target in targets if str(target))))
        owner_id = id(owner)
        wait_s = self._acquire_write_guard(
            target_tuple,
            owner_id=owner_id,
            wait=wait,
            timeout_s=timeout_s,
            respect_waiters=respect_waiters,
        )
        try:
            yield wait_s
        finally:
            self._release_write_guard(target_tuple, owner_id=owner_id)

    def _acquire_write_guard(
        self,
        targets: Iterable[str],
        *,
        owner_id: int,
        wait: bool,
        timeout_s: float,
        respect_waiters: bool,
    ) -> float:
        target_tuple = tuple(targets)
        started_at = time.perf_counter()
        with self._condition:
            while True:
                self._purge_expired_locked()
                reservation_blocked = self._blocked_reservations_locked(target_tuple, owner_id)
                waiter_blocked = (
                    self._blocked_waiters_locked(target_tuple, owner_id)
                    if bool(respect_waiters)
                    else []
                )
                blocked = list(reservation_blocked)
                blocked.extend(waiter_blocked)
                if waiter_blocked:
                    self._background_writer_waiter_blocked_checks += 1
                    self._background_writer_waiter_blocked_targets += len(waiter_blocked)
                if reservation_blocked:
                    self._background_writer_reservation_blocked_checks += 1
                    self._record_background_blocked_owners_locked(reservation_blocked)
                if not blocked:
                    for target in target_tuple:
                        owners = self._writers.setdefault(target, {})
                        owners[int(owner_id)] = owners.get(int(owner_id), 0) + 1
                    self._condition.notify_all()
                    return time.perf_counter() - started_at
                if not wait or time.perf_counter() - started_at >= float(timeout_s):
                    raise LockConflict("reservation blocked write", tuple(blocked))
                self._condition.wait(timeout=0.001)

    def _release_write_guard(self, targets: Iterable[str], *, owner_id: int) -> None:
        with self._condition:
            for target in targets:
                owners = self._writers.get(str(target))
                if not owners:
                    continue
                count = owners.get(int(owner_id), 0)
                if count <= 1:
                    owners.pop(int(owner_id), None)
                else:
                    owners[int(owner_id)] = count - 1
                if not owners:
                    self._writers.pop(str(target), None)
            self._condition.notify_all()

    def _blocked_reservations_locked(
        self,
        targets: Iterable[str],
        owner_id: int,
    ) -> List[str]:
        blocked = []
        for target in targets:
            current = self._owners.get(str(target))
            if current is not None and current[0] != int(owner_id):
                blocked.append(str(target))
        return blocked

    def _blocked_writers_locked(
        self,
        targets: Iterable[str],
        owner_id: int,
    ) -> List[str]:
        blocked = []
        for target in targets:
            owners = self._writers.get(str(target), {})
            if any(writer_id != int(owner_id) for writer_id in owners):
                blocked.append(str(target))
        return blocked

    def _blocked_waiters_locked(
        self,
        targets: Iterable[str],
        owner_id: int,
    ) -> List[str]:
        blocked = []
        for target in targets:
            queue = self._waiters.get(str(target), [])
            if any(int(waiter[3]) != int(owner_id) for waiter in queue):
                blocked.append(str(target))
        return blocked

    def _record_background_blocked_owners_locked(self, targets: Iterable[str]) -> None:
        owner_ids = {
            int(current[0])
            for target in targets
            for current in (self._owners.get(str(target)),)
            if current is not None
        }
        for owner_id in owner_ids:
            owner = self._reservation_owner_objects.get(owner_id)
            if owner is None:
                continue
            try:
                current = int(getattr(owner, "background_blocked_checks", 0) or 0)
                setattr(owner, "background_blocked_checks", current + 1)
            except (AttributeError, TypeError, ValueError):
                continue

    def _purge_expired_locked(self) -> None:
        now = time.perf_counter()
        expired = [
            target
            for target, (_owner_id, deadline) in self._owners.items()
            if deadline <= now
        ]
        for target in expired:
            self._owners.pop(target, None)

    def _first_waiter_for_all_locked(
        self,
        targets: Iterable[str],
        waiter: Tuple[int, float, int, int] | None,
    ) -> bool:
        if waiter is None:
            return True
        for target in targets:
            queue = self._waiters.get(str(target), [])
            if queue and min(queue) != waiter:
                return False
        return True

    def _remove_waiter_locked(
        self,
        targets: Iterable[str],
        waiter: Tuple[int, float, int, int] | None,
    ) -> None:
        if waiter is None:
            return
        for target in targets:
            queue = self._waiters.get(str(target), [])
            if waiter in queue:
                queue.remove(waiter)
            if not queue:
                self._waiters.pop(str(target), None)

    def snapshot_diagnostics(self) -> Dict[str, Any]:
        with self._condition:
            return {
                "reservation_waiter_count": int(self._reservation_waiter_count),
                "reservation_waiter_target_sizes": list(self._reservation_waiter_target_sizes),
                "reservation_unique_target_count": len(self._reservation_unique_targets),
                "reservation_all_or_nothing_failed_grant_checks": int(
                    self._reservation_all_or_nothing_failed_grant_checks
                ),
                "reservation_all_or_nothing_not_front_wait_ms": (
                    self._reservation_all_or_nothing_not_front_wait_s * 1000.0
                ),
                "reservation_front_queue_wait_ms": self._reservation_front_queue_wait_s * 1000.0,
                "reservation_owner_blocked_checks": int(self._reservation_owner_blocked_checks),
                "reservation_writer_blocked_checks": int(self._reservation_writer_blocked_checks),
                "reservation_blocked_target_checks": int(self._reservation_blocked_target_checks),
                "background_writer_waiter_blocked_checks": int(
                    self._background_writer_waiter_blocked_checks
                ),
                "background_writer_waiter_blocked_targets": int(
                    self._background_writer_waiter_blocked_targets
                ),
                "background_writer_reservation_blocked_checks": int(
                    self._background_writer_reservation_blocked_checks
                ),
            }

    def snapshot_pressure(self, targets: Iterable[str] = ()) -> Dict[str, Any]:
        target_tuple = tuple(sorted(set(str(target) for target in targets if str(target))))
        with self._condition:
            self._purge_expired_locked()
            if not target_tuple:
                target_tuple = tuple(
                    sorted(
                        set(self._owners)
                        | set(self._writers)
                        | set(self._waiters)
                    )
                )
            queue_lengths = {
                target: len(self._waiters.get(target, ()))
                for target in target_tuple
            }
            front_waiters = {
                min(queue)
                for target in target_tuple
                for queue in (self._waiters.get(target, ()),)
                if queue
            }
            queued_target_count = sum(1 for target in target_tuple if self._waiters.get(target))
            front_waiter_count = len(front_waiters)
            convoy_active = (
                len(target_tuple) > 1
                and queued_target_count > 1
                and front_waiter_count > 1
            )
            waiters = {
                waiter
                for target in target_tuple
                for waiter in self._waiters.get(target, ())
            }
            return {
                "reservation_queue_lengths": queue_lengths,
                "reservation_owner_targets": tuple(
                    target for target in target_tuple if target in self._owners
                ),
                "reservation_writer_targets": tuple(
                    target for target in target_tuple if self._writers.get(target)
                ),
                "reservation_waiter_count_current": len(waiters),
                "reservation_convoy_active": bool(convoy_active),
                "reservation_convoy_queue_target_count": int(queued_target_count),
                "reservation_convoy_front_waiter_count": int(front_waiter_count),
                "reservation_convoy_pressure": (
                    int(queued_target_count) * max(0, int(front_waiter_count) - 1)
                ),
            }

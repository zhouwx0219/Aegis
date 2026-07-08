"""Small lock tables used by agent-side CC strategies."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple


class LockConflict(RuntimeError):
    def __init__(self, reason: str, targets: Iterable[str]):
        super().__init__(reason)
        self.reason = str(reason)
        self.targets = tuple(str(target) for target in targets)


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
        self._writers: Dict[str, Dict[int, int]] = {}
        self._waiters: Dict[str, List[Tuple[int, float, int, int]]] = {}
        self._next_sequence = 0

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
        with self._condition:
            try:
                while True:
                    self._purge_expired_locked()
                    if waiter is None and wait and target_tuple:
                        waiter = (-int(priority), float(owner_age), self._next_sequence, int(owner_id))
                        self._next_sequence += 1
                        for target in target_tuple:
                            self._waiters.setdefault(target, []).append(waiter)
                    blocked = self._blocked_reservations_locked(target_tuple, owner_id)
                    blocked.extend(self._blocked_writers_locked(target_tuple, owner_id))
                    first_for_all = self._first_waiter_for_all_locked(target_tuple, waiter)
                    if not blocked and first_for_all:
                        break
                    if not wait or time.perf_counter() >= wait_deadline:
                        raise LockConflict("reservation conflict", tuple(blocked or target_tuple))
                    self._condition.wait(timeout=0.001)
                self._remove_waiter_locked(target_tuple, waiter)
                for target in target_tuple:
                    self._owners[target] = (owner_id, deadline)
                self._condition.notify_all()
                wait_s = time.perf_counter() - started_at
            except BaseException:
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
                blocked = [
                    target
                    for target in target_tuple
                    if target in self._owners and self._owners[target][0] != owner_id
                ]
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
    ):
        target_tuple = tuple(sorted(set(str(target) for target in targets if str(target))))
        owner_id = id(owner)
        wait_s = self._acquire_write_guard(
            target_tuple,
            owner_id=owner_id,
            wait=wait,
            timeout_s=timeout_s,
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
    ) -> float:
        target_tuple = tuple(targets)
        started_at = time.perf_counter()
        with self._condition:
            while True:
                self._purge_expired_locked()
                blocked = self._blocked_reservations_locked(target_tuple, owner_id)
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

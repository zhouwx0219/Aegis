"""Transaction-level traditional CC protocols for agent transactions.

These strategies keep the Data Agent System layering intact: the agent runtime
owns transaction semantics and concurrency-control policy, while the native
store remains a versioned KV substrate.
"""

from __future__ import annotations

import dataclasses
import contextlib
import threading
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class TraditionalCommitOutcome:
    committed: bool = False
    rejected: bool = False
    needs_regeneration: bool = False
    winner_branch_id: str = ""
    action: str = "abort"
    reason: str = ""
    conflict_object_ids: Tuple[str, ...] = ()


class TraditionalLockAbort(RuntimeError):
    def __init__(self, reason: str, targets: Iterable[str]):
        super().__init__(reason)
        self.reason = str(reason)
        self.targets = tuple(str(target) for target in targets)


class TwoPhaseLockTable:
    """S/X lock table with no-wait and wait-die deadlock prevention."""

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
        normalized_mode = self._normalize_mode(mode)
        normalized_policy = str(policy).strip().lower()
        acquired: List[Tuple[str, str]] = []
        try:
            for target in target_tuple:
                self._acquire_one(
                    target,
                    owner_id=owner_id,
                    owner_age=owner_age,
                    mode=normalized_mode,
                    policy=normalized_policy,
                )
                acquired.append((target, normalized_mode))
            yield
        finally:
            self.release(acquired, owner_id=owner_id)

    def release(self, acquired: Iterable[Tuple[str, str]], *, owner_id: int) -> None:
        with self._condition:
            for target, _mode in acquired:
                rows = self._holders.get(target, [])
                rows = [row for row in rows if int(row[0]) != int(owner_id)]
                if rows:
                    self._holders[target] = rows
                else:
                    self._holders.pop(target, None)
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
                    raise TraditionalLockAbort(
                        f"2pl-nowait: no-wait lock conflict on {target}",
                        (target,),
                    )
                if policy != "wait-die":
                    raise ValueError(f"unsupported 2PL policy: {policy}")
                conflicting = [
                    row for row in holders if int(row[0]) != int(owner_id)
                ]
                # Wait-die: old transactions wait for young holders; young
                # transactions die when they encounter an older holder.
                if any(owner_age > float(holder_age) for _hid, holder_age, _mode in conflicting):
                    raise TraditionalLockAbort(
                        f"2pl-wait-die: younger transaction aborted on {target}",
                        (target,),
                    )
                self._condition.wait(timeout=0.001)

    @staticmethod
    def _compatible(
        holders: Sequence[Tuple[int, float, str]],
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


class TraditionalCCExecutor:
    """Execute full traditional CC protocols over core branch read/write sets."""

    def __init__(self, store: Any, *, lock_table: Any):
        self.store = store
        self.lock_table = lock_table
        self.two_phase_locks = TwoPhaseLockTable()
        self._logical_ts = 0

    def commit_task(
        self,
        strategy: str,
        txn: Any,
        branches: Sequence[Any],
        stats: Any,
    ) -> TraditionalCommitOutcome:
        normalized = str(strategy).strip().lower()
        if normalized in {"2pl-nowait", "2pl-wait-die"}:
            return self._commit_2pl(normalized, txn, branches, stats)
        if normalized == "mvcc-full":
            return self._commit_mvcc(txn, branches, stats)
        if normalized == "silo-full":
            return self._commit_with_write_locks(
                normalized, txn, branches, stats, validate_reads=True
            )
        if normalized == "tictoc-full":
            return self._commit_with_write_locks(
                normalized, txn, branches, stats, validate_reads=False
            )
        raise ValueError(f"unsupported full traditional CC: {strategy}")

    def _commit_2pl(
        self,
        strategy: str,
        txn: Any,
        branches: Sequence[Any],
        stats: Any,
    ) -> TraditionalCommitOutcome:
        modes = self._branch_lock_modes(branches)
        targets = sorted(modes)
        policy = "nowait" if strategy == "2pl-nowait" else "wait-die"
        try:
            # The table supports one mode per acquire call.  Taking X for the
            # union is conservative and preserves strict 2PL correctness for
            # the multi-candidate agent task.
            mode = "x" if any(value == "x" for value in modes.values()) else "s"
            with self.two_phase_locks.acquire(
                targets,
                owner=txn,
                mode=mode,
                policy=policy,
            ):
                return self._commit_ranked(
                    txn,
                    branches,
                    stats,
                    validate_reads=True,
                    validate_writes=True,
                    reason_prefix=strategy,
                )
        except TraditionalLockAbort as exc:
            return TraditionalCommitOutcome(
                committed=False,
                needs_regeneration=True,
                winner_branch_id=branches[0].branch_id if branches else "",
                action="regenerate_required",
                reason=exc.reason,
                conflict_object_ids=exc.targets or tuple(targets),
            )

    def _commit_mvcc(
        self,
        txn: Any,
        branches: Sequence[Any],
        stats: Any,
    ) -> TraditionalCommitOutcome:
        # Snapshot isolation: reads are satisfied by the transaction snapshot;
        # commit only rejects write-write conflicts after the snapshot.
        return self._commit_ranked(
            txn,
            branches,
            stats,
            validate_reads=False,
            validate_writes=True,
            reason_prefix="mvcc-full",
        )

    def _commit_with_write_locks(
        self,
        strategy: str,
        txn: Any,
        branches: Sequence[Any],
        stats: Any,
        *,
        validate_reads: bool,
    ) -> TraditionalCommitOutcome:
        targets = self._write_targets(branches)
        with self.lock_table.acquire(targets):
            return self._commit_ranked(
                txn,
                branches,
                stats,
                validate_reads=validate_reads,
                validate_writes=True,
                reason_prefix=strategy,
            )

    def _commit_ranked(
        self,
        txn: Any,
        branches: Sequence[Any],
        stats: Any,
        *,
        validate_reads: bool,
        validate_writes: bool,
        reason_prefix: str,
    ) -> TraditionalCommitOutcome:
        if not branches:
            return TraditionalCommitOutcome(action="abort", reason="no candidates")
        stats.candidates_generated += len(branches)
        stats.n_tasks += 1
        ordered = sorted(enumerate(branches), key=lambda row: row[1].quality, reverse=True)
        first_conflict: Tuple[str, ...] = ()
        first_branch_id = ordered[0][1].branch_id
        for position, (_index, branch) in enumerate(ordered):
            conflict_targets = self._try_install_branch(
                branch,
                txn,
                validate_reads=validate_reads,
                validate_writes=validate_writes,
            )
            if conflict_targets:
                if not first_conflict:
                    first_conflict = conflict_targets
                    first_branch_id = branch.branch_id
                continue
            if position > 0:
                stats.n_reselect += 1
            return TraditionalCommitOutcome(
                committed=True,
                winner_branch_id=branch.branch_id,
                action="reselect" if position > 0 else "direct",
            )
        return TraditionalCommitOutcome(
            committed=False,
            needs_regeneration=True,
            winner_branch_id=first_branch_id,
            action="regenerate_required",
            reason=f"{reason_prefix}: validation conflict",
            conflict_object_ids=first_conflict,
        )

    def _conflict_targets(
        self,
        branch: Any,
        *,
        validate_reads: bool,
        validate_writes: bool,
    ) -> Tuple[str, ...]:
        conflicts = set()
        if validate_reads:
            for read in getattr(branch, "read_set", ()):
                if self._version_of(read.object_id) != int(read.version):
                    conflicts.add(str(read.object_id))
        if validate_writes:
            for write in getattr(branch, "writes", ()):
                if self._version_of(write.object_id) != int(write.base_version):
                    conflicts.add(str(write.object_id))
        return tuple(sorted(conflicts))

    def _try_install_branch(
        self,
        branch: Any,
        txn: Any,
        *,
        validate_reads: bool,
        validate_writes: bool,
    ) -> Tuple[str, ...]:
        manager = getattr(txn, "manager", None)
        lock = getattr(manager, "_lock", None)
        if lock is None:
            conflicts = self._conflict_targets(
                branch,
                validate_reads=validate_reads,
                validate_writes=validate_writes,
            )
            if conflicts:
                return conflicts
            for write in getattr(branch, "writes", ()):
                ok = self.store.put_if_version(
                    write.object_id,
                    int(write.base_version),
                    str(write.branch_value),
                )
                if not ok:
                    return (str(write.object_id),)
            return ()
        with lock:
            conflicts = self._conflict_targets(
                branch,
                validate_reads=validate_reads,
                validate_writes=validate_writes,
            )
            if conflicts:
                return conflicts
            for write in getattr(branch, "writes", ()):
                ok = self.store.put_if_version(
                    write.object_id,
                    int(write.base_version),
                    str(write.branch_value),
                )
                if not ok:
                    return (str(write.object_id),)
            self._logical_ts = max(self._logical_ts + 1, int(time.time_ns()))
            return ()

    def _version_of(self, object_id: str) -> int:
        return int(self.store.get_version(str(object_id)))

    @staticmethod
    def _branch_targets(branches: Iterable[Any]) -> List[str]:
        targets = set()
        for branch in branches:
            for read in getattr(branch, "read_set", ()):
                targets.add(str(read.object_id))
            for write in getattr(branch, "writes", ()):
                targets.add(str(write.object_id))
        return sorted(targets)

    @staticmethod
    def _branch_lock_modes(branches: Iterable[Any]) -> Dict[str, str]:
        modes: Dict[str, str] = {}
        for branch in branches:
            for read in getattr(branch, "read_set", ()):
                modes.setdefault(str(read.object_id), "s")
            for write in getattr(branch, "writes", ()):
                modes[str(write.object_id)] = "x"
        return modes

    @staticmethod
    def _write_targets(branches: Iterable[Any]) -> List[str]:
        targets = set()
        for branch in branches:
            for write in getattr(branch, "writes", ()):
                targets.add(str(write.object_id))
        return sorted(targets)

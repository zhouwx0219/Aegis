"""Workload-independent online hotspot metadata for paper ATCC."""

from __future__ import annotations

import dataclasses
import contextlib
import threading
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class HotnessConfig:
    min_accesses: int = 3
    absolute_hot_accesses: int = 8
    access_skew_multiple: float = 2.0
    min_access_share: float = 0.002
    min_conflict_events: int = 2
    conflict_ratio: float = 0.05
    min_wait_events: int = 2
    min_wait_ms: float = 0.25


@dataclasses.dataclass
class _ObjectHeat:
    accesses: int = 0
    conflicts: int = 0
    validation_failures: int = 0
    lock_wait_events: int = 0
    lock_wait_ms: float = 0.0
    wounds: int = 0


class HotnessTracker:
    """Classify objects from observed access and contention signals only."""

    def __init__(self, config: HotnessConfig | None = None, *, shards: int = 64):
        self.config = config or HotnessConfig()
        shard_count = max(1, int(shards))
        self._locks = tuple(threading.RLock() for _ in range(shard_count))
        self._objects = tuple({} for _ in range(shard_count))
        self._profiled = tuple(set() for _ in range(shard_count))
        self._profiled_transactions = tuple({} for _ in range(shard_count))
        self._accesses = [0] * shard_count

    def observe_access(self, object_id: str) -> bool:
        key = str(object_id)
        index = self._shard_index(key)
        with self._locks[index]:
            heat = self._objects[index].setdefault(key, _ObjectHeat())
            heat.accesses += 1
            self._accesses[index] += 1
            return self._is_hot(
                heat,
                self._accesses[index],
                len(self._objects[index]),
                access_share_scale=len(self._objects),
            )

    def prime_accesses(self, object_ids: Iterable[str]) -> None:
        """Load a read-only workload-frequency prior before execution starts."""
        grouped: dict[int, dict[str, int]] = {}
        for object_id in object_ids:
            key = str(object_id)
            index = self._shard_index(key)
            counts = grouped.setdefault(index, {})
            counts[key] = counts.get(key, 0) + 1
        for index, counts in grouped.items():
            with self._locks[index]:
                for key, count in counts.items():
                    heat = self._objects[index].setdefault(key, _ObjectHeat())
                    heat.accesses += int(count)
                    self._accesses[index] += int(count)
                    self._profiled[index].add(key)

    def is_profiled_hot(self, object_id: str) -> bool:
        key = str(object_id)
        index = self._shard_index(key)
        with self._locks[index]:
            if key not in self._profiled[index]:
                return False
            heat = self._objects[index].get(key)
            return bool(
                heat is not None
                and self._is_hot(
                    heat,
                    self._accesses[index],
                    len(self._objects[index]),
                    access_share_scale=len(self._objects),
                )
            )

    def prime_transaction(self, object_ids: Iterable[str]) -> None:
        """Record how many distinct planned transactions touch each object."""
        for key in {str(object_id) for object_id in object_ids}:
            index = self._shard_index(key)
            with self._locks[index]:
                counts = self._profiled_transactions[index]
                counts[key] = counts.get(key, 0) + 1

    def is_profiled_shared(self, object_id: str, *, min_transactions: int = 2) -> bool:
        key = str(object_id)
        index = self._shard_index(key)
        with self._locks[index]:
            return int(self._profiled_transactions[index].get(key, 0)) >= max(
                1, int(min_transactions)
            )

    def observe_contention(
        self,
        object_id: str,
        event: str,
        wait_ms: float = 0.0,
    ) -> None:
        key = str(object_id)
        normalized = str(event).strip().lower()
        index = self._shard_index(key)
        with self._locks[index]:
            heat = self._objects[index].setdefault(key, _ObjectHeat())
            if normalized in {"validation-failure", "version-conflict"}:
                heat.validation_failures += 1
                heat.conflicts += 1
            elif normalized == "lock-wait":
                heat.lock_wait_events += 1
                heat.lock_wait_ms += max(0.0, float(wait_ms))
            elif normalized in {"wound", "lock-preempted"}:
                heat.wounds += 1
                heat.conflicts += 1
            elif normalized in {"lock-conflict", "lock-timeout"}:
                heat.conflicts += 1

    def is_hot(self, object_id: str) -> bool:
        key = str(object_id)
        index = self._shard_index(key)
        with self._locks[index]:
            heat = self._objects[index].get(key)
            return bool(
                heat is not None
                and self._is_hot(
                    heat,
                    self._accesses[index],
                    len(self._objects[index]),
                    access_share_scale=len(self._objects),
                )
            )

    def hot_targets(self, object_ids: Iterable[str]) -> set[str]:
        targets = {str(object_id) for object_id in object_ids}
        grouped: dict[int, list[str]] = {}
        for target in targets:
            grouped.setdefault(self._shard_index(target), []).append(target)
        hot = set()
        for index, keys in grouped.items():
            with self._locks[index]:
                for key in keys:
                    heat = self._objects[index].get(key)
                    if heat is not None and self._is_hot(
                        heat,
                        self._accesses[index],
                        len(self._objects[index]),
                        access_share_scale=len(self._objects),
                    ):
                        hot.add(key)
        return hot

    def snapshot(self) -> dict[str, object]:
        with contextlib.ExitStack() as stack:
            for lock in self._locks:
                stack.enter_context(lock)
            total_accesses, object_count = self._population()
            rows = []
            hot = 0
            for index, shard in enumerate(self._objects):
                shard_rows = list(shard.values())
                rows.extend(shard_rows)
                hot += sum(
                    1
                    for heat in shard_rows
                    if self._is_hot(
                        heat,
                        self._accesses[index],
                        len(shard),
                        access_share_scale=len(self._objects),
                    )
                )
            return {
                "observed_objects": object_count,
                "total_accesses": total_accesses,
                "hot_objects": hot,
                "validation_failures": sum(row.validation_failures for row in rows),
                "lock_wait_events": sum(row.lock_wait_events for row in rows),
                "lock_wait_ms": sum(row.lock_wait_ms for row in rows),
                "wounds": sum(row.wounds for row in rows),
            }

    def object_snapshot(self, object_id: str) -> dict[str, object]:
        key = str(object_id)
        index = self._shard_index(key)
        with self._locks[index]:
            heat = self._objects[index].get(key, _ObjectHeat())
            return {
                **dataclasses.asdict(heat),
                "hot": self._is_hot(
                    heat,
                    self._accesses[index],
                    len(self._objects[index]),
                    access_share_scale=len(self._objects),
                ),
            }

    def _shard_index(self, object_id: str) -> int:
        return hash(str(object_id)) % len(self._objects)

    def _population(self) -> tuple[int, int]:
        return sum(self._accesses), sum(len(shard) for shard in self._objects)

    def _is_hot(
        self,
        heat: _ObjectHeat,
        total_accesses: int,
        object_count: int,
        *,
        access_share_scale: int = 1,
    ) -> bool:
        cfg = self.config
        accesses = max(0, int(heat.accesses))
        conflict_hot = (
            accesses >= cfg.min_accesses
            and heat.conflicts >= cfg.min_conflict_events
            and heat.conflicts / accesses >= cfg.conflict_ratio
        )
        wait_hot = (
            accesses >= cfg.min_accesses
            and heat.lock_wait_events >= cfg.min_wait_events
            and heat.lock_wait_ms >= cfg.min_wait_ms
        )
        if conflict_hot or wait_hot:
            return True
        if accesses < cfg.min_accesses or total_accesses <= 0:
            return False
        # A repeatedly accessed object is hot even when hash sharding places it
        # in an otherwise empty shard. This keeps classification independent of
        # Python's per-process hash seed.
        if accesses >= max(cfg.min_accesses, cfg.absolute_hot_accesses):
            return True
        average = total_accesses / max(1, object_count)
        return (
            accesses >= cfg.access_skew_multiple * average
            and accesses / total_accesses
            >= cfg.min_access_share * max(1, int(access_share_scale))
        )

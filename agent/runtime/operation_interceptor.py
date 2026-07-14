"""Unified hooks for transaction operations and phase/action changes."""

from __future__ import annotations

import time
from typing import Any, Protocol

from .context import LockAction, TransactionPhase


class TransactionHooks(Protocol):
    def on_begin(self, txn: Any) -> None: ...
    def before_read(self, txn: Any, object_id: str) -> None: ...
    def after_read(self, txn: Any, object_id: str, version: int) -> None: ...
    def before_write(self, txn: Any, object_id: str) -> None: ...
    def after_write(self, txn: Any, object_id: str) -> None: ...
    def on_phase_change(self, txn: Any, phase: TransactionPhase) -> None: ...
    def on_action_change(self, txn: Any, old_action: LockAction, new_action: LockAction) -> None: ...
    def before_commit(self, txn: Any) -> None: ...
    def on_abort(self, txn: Any, reason: str) -> None: ...
    def on_finish(self, txn: Any) -> None: ...


class NoopTransactionHooks:
    def on_begin(self, txn: Any) -> None: pass
    def before_read(self, txn: Any, object_id: str) -> None: pass
    def after_read(self, txn: Any, object_id: str, version: int) -> None: pass
    def before_write(self, txn: Any, object_id: str) -> None: pass
    def after_write(self, txn: Any, object_id: str) -> None: pass
    def on_phase_change(self, txn: Any, phase: TransactionPhase) -> None: pass
    def on_action_change(self, txn: Any, old_action: LockAction, new_action: LockAction) -> None: pass
    def before_commit(self, txn: Any) -> None: pass
    def on_abort(self, txn: Any, reason: str) -> None: pass
    def on_finish(self, txn: Any) -> None: pass


class OperationInterceptor:
    def __init__(self, hooks: TransactionHooks | None = None, *, state_collector: Any = None):
        self.hooks = hooks or NoopTransactionHooks()
        self.state_collector = state_collector

    def _account_agent_interval(self, txn: Any) -> float:
        if self._bypass_background_state(txn):
            return 0.0
        now = time.monotonic_ns()
        interval_ms = max(
            0.0,
            (now - txn.context.last_agent_accounted_ns) / 1_000_000.0,
        )
        txn.context.agent_cost_ms += interval_ms
        txn.context.last_agent_accounted_ns = now
        if self.state_collector is not None:
            self.state_collector.record_agent_interval(txn.context, interval_ms)
        return interval_ms

    def _record(self, txn: Any, object_id: str, *, write: bool) -> None:
        if self._bypass_background_state(txn):
            return
        interval_ms = self._account_agent_interval(txn)
        if self.state_collector is not None:
            self.state_collector.record_operation(
                txn.context,
                object_id,
                write=write,
                interval_ms=interval_ms,
                hot=False,
            )

    def account_agent_interval(self, txn: Any) -> float:
        return self._account_agent_interval(txn)

    @staticmethod
    def reset_agent_interval(txn: Any) -> None:
        txn.context.last_agent_accounted_ns = time.monotonic_ns()

    def operation_finished(
        self,
        txn: Any,
        *,
        elapsed_ms: float,
        blocked_before_ms: float,
    ) -> None:
        if self._bypass_background_state(txn):
            return
        blocked_delta = max(0.0, txn.context.blocked_time_ms - blocked_before_ms)
        txn.context.operation_cost_ms += max(0.0, float(elapsed_ms) - blocked_delta)
        txn.context.completed_operations += 1
        now = time.monotonic_ns()
        txn.context.last_operation_end_ns = now
        txn.context.last_agent_accounted_ns = now

    def begin(self, txn: Any) -> None:
        self.hooks.on_begin(txn)

    def before_read(self, txn: Any, object_id: str) -> None:
        self._record(txn, object_id, write=False)
        self.hooks.before_read(txn, object_id)

    def after_read(self, txn: Any, object_id: str, version: int) -> None:
        txn.context.read_versions.setdefault(str(object_id), int(version))
        self.hooks.after_read(txn, object_id, int(version))

    def before_write(self, txn: Any, object_id: str) -> None:
        self._record(txn, object_id, write=True)
        txn.context.write_targets.add(str(object_id))
        self.hooks.before_write(txn, object_id)

    def after_write(self, txn: Any, object_id: str) -> None:
        self.hooks.after_write(txn, object_id)

    def phase_change(self, txn: Any, phase: TransactionPhase) -> None:
        if self.state_collector is not None and phase != txn.context.phase:
            self.state_collector.finish_round(txn.context)
        txn.context.change_phase(phase)
        self.hooks.on_phase_change(txn, phase)

    def action_change(self, txn: Any, action: LockAction) -> None:
        previous = txn.context.action
        txn.context.change_action(action)
        self.hooks.on_action_change(txn, previous, action)

    def before_commit(self, txn: Any) -> None:
        self.hooks.before_commit(txn)

    def abort(self, txn: Any, reason: str) -> None:
        self.hooks.on_abort(txn, reason)

    def finish(self, txn: Any) -> None:
        self.hooks.on_finish(txn)
        if self.state_collector is not None:
            self.state_collector.discard(txn.context)

    @staticmethod
    def _bypass_background_state(txn: Any) -> bool:
        metadata = getattr(txn, "metadata", {}) or {}
        return bool(
            getattr(txn.context, "is_background", False)
            or metadata.get("runtime_background", False)
            or metadata.get("_cold_occ_fast_task", False)
        )

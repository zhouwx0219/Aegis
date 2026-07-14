"""Agent-side single-plan transaction runtime."""

from .transaction import AgentTransaction, AgentTransactionManager
from .context import LockAction, LockClass, TransactionContext, TransactionPhase, TransactionStatus
from .operation_interceptor import OperationInterceptor, TransactionHooks
from .atcc_lock_manager import PaperATCCLockManager
from .priority import PriorityConfig, PriorityManager
from .hotness import HotnessConfig, HotnessTracker
from .state_collector import PhaseAwareState, StateCollector, phase_aware_state_from_dict
from .undo_log import UndoLog, UndoRecord
from .version_manager import CommittedVersion, PrivateVersion, VersionManager
from .paper_policy import AtomicPolicyManager, CompiledPhasePolicy, CompiledPolicyEntry
from .paper_hooks import PaperATCCHooks
from .trajectory import (
    PaperRewardConfig,
    PolicyTransition,
    TrajectoryCollector,
    apply_paper_rewards,
    policy_transition_from_dict,
    system_performance_delta,
)
from .types import (
    ConflictDetail,
    ReadRecord,
    SnapshotValue,
    TransactionEvent,
    TransactionResult,
    TransactionState,
    WriteRecord,
)

__all__ = [
    "AgentTransaction",
    "AgentTransactionManager",
    "ConflictDetail",
    "ReadRecord",
    "SnapshotValue",
    "TransactionEvent",
    "TransactionResult",
    "TransactionState",
    "VersionManager",
    "WriteRecord",
]

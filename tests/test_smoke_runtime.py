import dataclasses
import math
import threading
import unittest
import tempfile
from pathlib import Path

from agent.cc import LockConflict
from agent.cli.smoke import run_smoke_checks
from agent.runtime import (
    AgentTransactionManager,
    LockAction,
    LockClass,
    TransactionPhase,
    TransactionStatus,
    TransactionContext,
    PriorityConfig,
    PriorityManager,
    CompiledPhasePolicy,
    CompiledPolicyEntry,
    AtomicPolicyManager,
    HotnessTracker,
    apply_paper_rewards,
)
from agent.workloads import build_workload, execute_task, register_workload
from agent.cc.atcc.ppo import (
    DiscretePPOPolicy,
    DiscretePPOTrainer,
    EpsilonGreedyPolicy,
    PPOConfig,
    normalize_returns_by_source,
    policy_group_key,
    reconstruct_behavior_distribution,
    state_key,
)
from agent.runtime import PhaseAwareState, PolicyTransition
from agent.runtime.paper_policy import StaticThresholdPhasePolicy
from agent.runtime.trajectory import system_performance_delta


class SmokeRuntimeTests(unittest.TestCase):
    def test_static_threshold_policy_uses_only_fixed_conflict_signal(self):
        base = PhaseAwareState(
            phase="refine",
            inter_round_interval_ms=10_000,
            read_set_size=20,
            write_set_size=10,
            read_set_growth=10,
            write_set_growth=10,
            access_overlap_ratio=1.0,
            completed_rounds=5,
            completed_operations=100,
            recent_write_ratio=1.0,
            hotspot_access_ratio=1.0,
            blocked_time_ms=10_000,
            retry_count=5,
            current_action=0,
            priority=999,
            recent_conflict_kind="write-write",
            global_abort_rate=1.0,
            global_conflict_abort_rate=0.19,
        )
        policy = StaticThresholdPhasePolicy()

        self.assertEqual(LockClass.NONE, policy.select(base).protected)
        crossed = dataclasses.replace(base, global_conflict_abort_rate=0.20)
        self.assertEqual(LockClass.HOT_WRITE, policy.select(crossed).protected)
        demo_policy = StaticThresholdPhasePolicy(conflict_abort_threshold=0.10)
        self.assertEqual(LockClass.HOT_WRITE, demo_policy.select(base).protected)
        broad_demo = StaticThresholdPhasePolicy(
            conflict_abort_threshold=0.10,
            protection_mask=int(LockClass.HOT_READ | LockClass.HOT_WRITE),
        )
        self.assertEqual(
            LockClass.HOT_READ | LockClass.HOT_WRITE,
            broad_demo.select(base).protected,
        )

    def test_conditional_status_transition_has_one_concurrent_winner(self):
        context = TransactionContext("conditional-transition", 0, 0)
        barrier = threading.Barrier(17)
        results = []

        def begin_abort():
            barrier.wait()
            results.append(
                context.try_transition(
                    TransactionStatus.ABORTING,
                    from_statuses=(TransactionStatus.ACTIVE, TransactionStatus.WAITING),
                )
            )

        threads = [threading.Thread(target=begin_abort) for _ in range(16)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(1, sum(results))
        self.assertEqual(TransactionStatus.ABORTING, context.status)
        context.transition(TransactionStatus.ABORTED)

    def test_hotness_tracker_uses_runtime_access_and_contention_signals(self):
        tracker = HotnessTracker()
        for index in range(200):
            tracker.observe_access(f"cold-{index}")
        for _ in range(20):
            tracker.observe_access("skewed")

        self.assertTrue(tracker.is_hot("skewed"))
        self.assertFalse(tracker.is_hot("cold-0"))

        tracker.observe_contention("background-only", "validation-failure")
        self.assertFalse(tracker.is_hot("background-only"))
        for _ in range(3):
            tracker.observe_access("validation-hot")
            tracker.observe_access("wait-hot")
        for _ in range(2):
            tracker.observe_contention("validation-hot", "validation-failure")
            tracker.observe_contention("wait-hot", "lock-wait", 0.5)
        self.assertTrue(tracker.is_hot("validation-hot"))
        self.assertTrue(tracker.is_hot("wait-hot"))

    def test_action_change_reclassifies_earlier_optimistic_access(self):
        manager = AgentTransactionManager()
        manager.register_object("target", "0", kind="row")
        txn = manager.begin("reclassify", {"paper_atcc": True})
        txn.read("target")
        self.assertNotIn("target", txn.context.hot_read_targets)

        for index in range(200):
            manager.hotness_tracker.observe_access(f"cold-{index}")
        for _ in range(20):
            manager.hotness_tracker.observe_access("target")
        manager.transition_atcc_action(txn, LockAction(LockClass.HOT_READ))

        self.assertIn("target", txn.context.hot_read_targets)
        self.assertIn("target", txn.context.held_read_locks)
        manager.atcc_locks.release_all(txn.context)

    def test_retry_state_preserves_structured_conflict_history(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin(
            "retry",
            {
                "paper_atcc": True,
                "retry_count": 1,
                "previous_failure_reason": "retroactive validation failed on row",
            },
        )

        state = manager.state_collector.snapshot(txn.context)
        self.assertEqual("version-conflict", state.recent_conflict_kind)

    def test_version_conflict_returns_class_and_inherits_monotonic_retry_mask(self):
        policy = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="commit", action=13)], generation=1
        )
        manager = AgentTransactionManager(paper_policy=policy)
        manager.register_object("row", "0", kind="row")
        agent = manager.begin(
            "retry-mask",
            {
                "paper_atcc": True,
                "retry_count": 0,
                "retry_protection_mask": 13,
            },
        )
        agent.read("row")
        writer = manager.begin("writer")
        writer.write("row", "1")
        self.assertTrue(writer.commit("occ").committed)

        result = agent.commit("paper-atcc")

        self.assertFalse(result.committed)
        self.assertEqual(("row",), result.conflict_object_ids)
        self.assertEqual("cold-read", result.conflict_details[0].protection_class)
        self.assertEqual(15, result.retry_protection_mask)
        retry = manager.begin(
            "retry-mask",
            {"paper_atcc": True, "retry_count": 1},
        )
        self.assertEqual(15, int(retry.context.action.protected))
        self.assertEqual(1, retry.context.retry_validation_conflicts)
        self.assertEqual({"row"}, retry.context.retry_conflict_read_targets)
        retry.abort("test cleanup")

    def test_second_version_conflict_escalates_to_full_observed_classes(self):
        manager = AgentTransactionManager()
        manager.register_object("cold-read", "0", kind="row")
        manager.register_object("hot-write", "0", kind="row")
        txn = manager.begin("double-conflict", {"paper_atcc": True})
        txn.read("cold-read")
        _, first_mask = manager._prepare_retry_feedback(
            txn,
            reason="atomic version check failed",
            conflict_object_ids=("cold-read",),
        )
        for _ in range(8):
            manager.hotness_tracker.observe_access("hot-write")
        txn.write("hot-write", "1")

        details, second_mask = manager._prepare_retry_feedback(
            txn,
            reason="atomic version check failed",
            conflict_object_ids=("hot-write",),
        )

        self.assertEqual(int(LockClass.COLD_READ), first_mask)
        self.assertEqual("hot-write", details[0].protection_class)
        self.assertTrue(LockClass(second_mask) & LockClass.COLD_READ)
        self.assertTrue(LockClass(second_mask) & LockClass.HOT_WRITE)
        retry = manager.begin(
            "double-conflict",
            {"paper_atcc": True, "retry_count": 2},
        )
        self.assertIn("cold-read", retry.context.retry_conflict_read_targets)
        self.assertIn("hot-write", retry.context.retry_conflict_write_targets)
        retry.abort("test cleanup")
        diagnostics = manager.retry_protection_diagnostics()
        self.assertEqual(1, diagnostics["full_observed_escalations"])
        txn.abort("test cleanup")

    def test_refine_guard_adds_missing_hot_read_protection_to_action14(self):
        policy = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="refine", action=14)], generation=1
        )
        manager = AgentTransactionManager(paper_policy=policy)
        manager.register_object("hot", "0", kind="row")
        for _ in range(8):
            manager.hotness_tracker.observe_access("hot")
        txn = manager.begin(
            "hot-read-guard",
            {
                "paper_atcc": True,
                "retry_protection_mask": 14,
            },
        )
        txn.read("hot")
        self.assertNotIn("hot", txn.context.held_read_locks)

        txn.enter_phase("refine")

        self.assertEqual(15, int(txn.context.action.protected))
        self.assertIn("hot", txn.context.held_read_locks)
        txn.abort("test cleanup")

    def test_transaction_context_uses_stable_generation_and_full_lifecycle(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("task", {"retry_count": 2, "generation": 7})

        self.assertEqual("task:2:7", txn.context.tid)
        txn.enter_phase(TransactionPhase.REFINE)
        txn.write("row", "1")
        result = txn.commit("occ")

        self.assertTrue(result.committed)
        self.assertEqual(TransactionStatus.COMMITTED, txn.context.status)
        self.assertEqual(TransactionPhase.REFINE, txn.context.phase)

    def test_paper_policy_escalates_at_completed_batch_boundary(self):
        policy = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="refine", action=2)], generation=1
        )
        manager = AgentTransactionManager(paper_policy=policy)
        manager.register_object("a", "0", kind="row")
        manager.register_object("b", "0", kind="row")
        txn = manager.begin(
            "operation-boundary",
            {"paper_atcc": True},
            snapshot_object_ids=("a", "b"),
        )

        txn.read("a")
        self.assertEqual(0, int(txn.context.action.protected))
        txn.read("b")
        self.assertEqual(0, int(txn.context.action.protected))
        txn.enter_phase("refine")

        self.assertEqual(2, int(txn.context.action.protected))
        self.assertEqual({"a", "b"}, txn.context.held_read_locks)
        txn.abort("test cleanup")
        manager.atcc_locks.release_all(txn.context)

    def test_low_and_medium_first_attempt_wait_for_retry_evidence(self):
        policy = CompiledPhasePolicy(
            [
                CompiledPolicyEntry(phase="explore", action=15),
                CompiledPolicyEntry(phase="refine", action=15),
            ],
            generation=1,
            occ_cold_start_guard=False,
        )
        for level in ("low", "medium"):
            with self.subTest(level=level):
                manager = AgentTransactionManager(
                    paper_policy=policy,
                    low_conflict_occ_guard=True,
                )
                manager.register_object("hot", "0", kind="row")
                for _ in range(8):
                    manager.hotness_tracker.observe_access("hot")
                first = manager.begin(
                    f"{level}-first",
                    {
                        "paper_atcc": True,
                        "workload": "ycsb",
                        "context": {"level": level},
                    },
                )

                first.enter_phase("explore")
                first.read("hot")
                first.enter_phase("refine")

                self.assertEqual(LockClass.NONE, first.context.action.protected)
                self.assertEqual(set(), first.context.held_read_locks)
                first.abort("test cleanup")

                retry = manager.begin(
                    f"{level}-retry",
                    {
                        "paper_atcc": True,
                        "workload": "ycsb",
                        "context": {"level": level},
                        "retry_count": 1,
                    },
                )
                retry.enter_phase("explore")
                self.assertEqual(LockClass.NONE, retry.context.action.protected)
                retry.read("hot")
                retry.enter_phase("refine")
                self.assertEqual(LockClass(15), retry.context.action.protected)
                retry.abort("test cleanup")

    def test_lock_action_only_expands_protection(self):
        initial = LockAction()
        reads = LockAction(LockClass.HOT_READ)
        reads_and_writes = LockAction(LockClass.HOT_READ | LockClass.HOT_WRITE)

        self.assertEqual(LockClass.HOT_READ, reads.added_since(initial))
        self.assertEqual(LockClass.HOT_WRITE, reads_and_writes.added_since(reads))
        with self.assertRaises(ValueError):
            reads.added_since(reads_and_writes)

    def test_phase_cannot_move_backwards(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("phase-order")
        txn.enter_phase("commit")

        with self.assertRaises(ValueError):
            txn.enter_phase("refine")

    def test_first_access_reads_latest_committed_value_without_plan_oracle(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        observer = manager.begin("observer", strategy="paper-atcc")
        writer = manager.begin("writer")
        writer.write("row", "1")
        self.assertTrue(writer.commit("occ").committed)

        self.assertEqual("1", observer.read("row").value)
        observer.abort("test cleanup")

    def test_paper_strategy_at_begin_activates_runtime_hooks(self):
        policy = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="refine", action=int(LockClass.COLD_READ))],
            generation=1,
        )
        manager = AgentTransactionManager(paper_policy=policy)
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("integrated", strategy="paper-atcc")
        txn.read("row")
        txn.enter_phase("refine")

        self.assertTrue(txn.metadata["paper_atcc"])
        self.assertIn("row", txn.context.held_read_locks)
        txn.abort("test cleanup")

    def test_action_change_retroactively_validates_prior_reads(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        agent = manager.begin("agent")
        agent.read("row")
        writer = manager.begin("writer")
        writer.write("row", "1")
        self.assertTrue(writer.commit("occ").committed)

        with self.assertRaises(Exception):
            manager.transition_atcc_action(
                agent,
                LockAction(LockClass.COLD_READ),
                timeout_s=0.01,
            )

        self.assertEqual(TransactionStatus.ABORTED, agent.context.status)

    def test_refine_write_action_upgrades_observed_planned_write_to_wlock(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        agent = manager.begin(
            "planned-write",
            {
                "paper_atcc": True,
                "planned_write_targets": ["row"],
            },
        )
        agent.read("row")

        manager.transition_atcc_action(
            agent,
            LockAction(LockClass.COLD_WRITE),
        )

        self.assertIn("row", agent.context.held_write_locks)
        self.assertNotIn("row", agent.context.held_read_locks)
        agent.abort("test cleanup")

    def test_conflict_on_observed_planned_write_is_classified_as_write(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        agent = manager.begin(
            "planned-conflict",
            {
                "paper_atcc": True,
                "planned_write_targets": ["row"],
            },
        )
        agent.read("row")

        details, _mask = manager._prepare_retry_feedback(
            agent,
            reason="atomic version check failed",
            conflict_object_ids=("row",),
        )

        self.assertEqual("write", details[0].access_kind)
        self.assertEqual("cold-write", details[0].protection_class)
        agent.abort("test cleanup")

    def test_write_action_change_rejects_historical_background_version(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        agent = manager.begin("historical-write", {"paper_atcc": True})
        agent.read("row")
        agent.write("row", "agent")
        background = manager.begin("background", {"paper_atcc_backend": True})
        background.write("row", "background")
        self.assertTrue(background.commit("occ").committed)

        with self.assertRaises(Exception):
            manager.transition_atcc_action(
                agent,
                LockAction(LockClass.COLD_WRITE),
                timeout_s=0.01,
            )

        self.assertEqual(TransactionStatus.ABORTED, agent.context.status)

    def test_younger_dynamic_priority_does_not_wound_older_writer(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        lower = manager.begin("lower")
        higher = manager.begin("higher")
        manager.atcc_locks.update_priority(lower.context, 1)
        manager.atcc_locks.update_priority(higher.context, 9)
        manager.atcc_locks.wlock("row", lower.context, timeout_s=0.01)

        with self.assertRaises(LockConflict):
            manager.atcc_locks.wlock("row", higher.context, timeout_s=0.002)

        self.assertEqual(TransactionStatus.ACTIVE, lower.context.status)
        self.assertEqual(0, higher.context.agent_aborts_caused)
        diagnostics = manager.atcc_locks.snapshot_diagnostics()
        self.assertEqual(0, diagnostics.get("wounds_agent_to_agent", 0))
        self.assertEqual(lower.context.tid, manager.atcc_locks.snapshot("row")["writer"])
        manager.atcc_locks.release_all(lower.context)
        manager.atcc_locks.release_all(higher.context)
        self.assertEqual(
            0, manager.atcc_locks.snapshot_diagnostics()["live_contexts"]
        )

    def test_older_transaction_wounds_younger_writer_independent_of_priority(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        older = manager.begin("older")
        younger = manager.begin("younger")
        manager.atcc_locks.update_priority(older.context, 1)
        manager.atcc_locks.update_priority(younger.context, 9)
        manager.atcc_locks.wlock("row", younger.context, timeout_s=0.01)

        manager.atcc_locks.wlock("row", older.context, timeout_s=0.01)

        self.assertEqual(TransactionStatus.ABORTED, younger.context.status)
        self.assertEqual(older.context.tid, manager.atcc_locks.snapshot("row")["writer"])
        manager.atcc_locks.release_all(older.context)

    def test_aborted_context_cannot_reregister_for_a_lock(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("aborted-reregister", {"paper_atcc": True})
        txn.abort("test")

        with self.assertRaises(LockConflict):
            manager.atcc_locks.wlock("row", txn.context, timeout_s=0.01)

        self.assertEqual(
            0, manager.atcc_locks.snapshot_diagnostics()["live_contexts"]
        )

    def test_policy_lock_footprint_survives_release_for_reward_accounting(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("policy-footprint", {"paper_atcc": True})
        txn.read("row")

        manager.transition_atcc_action(txn, LockAction(LockClass.COLD_READ))
        manager.atcc_locks.release_all(txn.context)

        self.assertFalse(txn.context.held_read_locks)
        self.assertEqual({"row"}, txn.context.policy_read_lock_targets)

    def test_committing_writer_cannot_be_wounded(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        writer = manager.begin("writer")
        challenger = manager.begin("challenger")
        manager.atcc_locks.update_priority(writer.context, 1)
        manager.atcc_locks.update_priority(challenger.context, 9)
        manager.atcc_locks.wlock("row", writer.context, timeout_s=0.01)
        self.assertTrue(manager.atcc_locks.begin_committing(writer.context))

        with self.assertRaises(Exception):
            manager.atcc_locks.wlock("row", challenger.context, timeout_s=0.002)

        self.assertEqual(writer.context.tid, manager.atcc_locks.snapshot("row")["writer"])
        manager.atcc_locks.release_all(writer.context)

    def test_commit_wait_protection_is_disabled_with_priority_ablation(self):
        manager = AgentTransactionManager(priority_enabled=False)
        manager.register_object("row", "0", kind="row")
        older = manager.begin("older")
        younger = manager.begin("younger")
        older.context.phase = TransactionPhase.COMMIT
        younger.context.phase = TransactionPhase.COMMIT
        manager.atcc_locks.wlock("row", younger.context, timeout_s=0.01)

        manager.atcc_locks.wlock("row", older.context, timeout_s=0.01)

        self.assertEqual(TransactionStatus.ABORTED, younger.context.status)
        self.assertEqual(older.context.tid, manager.atcc_locks.snapshot("row")["writer"])
        manager.atcc_locks.release_all(older.context)

    def test_priority_enabled_commit_wait_protects_paid_work(self):
        manager = AgentTransactionManager(priority_enabled=True)
        manager.register_object("row", "0", kind="row")
        older = manager.begin("older")
        younger = manager.begin("younger")
        older.context.phase = TransactionPhase.COMMIT
        younger.context.phase = TransactionPhase.COMMIT
        manager.atcc_locks.wlock("row", younger.context, timeout_s=0.01)

        with self.assertRaises(LockConflict):
            manager.atcc_locks.wlock("row", older.context, timeout_s=0.002)

        self.assertEqual(TransactionStatus.ACTIVE, younger.context.status)
        self.assertEqual(younger.context.tid, manager.atcc_locks.snapshot("row")["writer"])
        manager.atcc_locks.release_all(younger.context)
        manager.atcc_locks.release_all(older.context)

    def test_priority_matches_paper_runtime_cost_formula(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("priority", {"retry_count": 2})
        txn.context.completed_operations = 25
        # Runtime overhead is not Op(T) in the paper formula.
        txn.context.operation_cost_ms = 999
        txn.context.blocked_time_ms = 11
        txn.context.agent_cost_ms = 39
        priority = PriorityManager(
            PriorityConfig(
                sql_weight=1,
                blocked_weight=1,
                retry_weight=1,
                interval_weight=1,
                sql_quantum_ms=10,
                blocked_quantum_ms=10,
                interval_quantum_ms=10,
            )
        ).compute(txn.context)

        self.assertEqual(8, priority)
        txn.context.prior_retry_cost_ms = 20
        self.assertEqual(10, PriorityManager(
            PriorityConfig(
                sql_weight=1,
                blocked_weight=1,
                retry_weight=1,
                interval_weight=1,
                sql_quantum_ms=10,
                blocked_quantum_ms=10,
                interval_quantum_ms=10,
            )
        ).compute(txn.context))

    def test_agent_write_protection_is_deferred_until_unified_commit(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin(
            "deferred-agent-write",
            {
                "paper_atcc": True,
                "retry_protection_mask": int(LockClass.COLD_WRITE),
                "commit_admission_write_protection": True,
            },
        )

        txn.write("row", "1")

        self.assertFalse(txn.context.held_write_locks)
        self.assertEqual({"row"}, txn.context.policy_write_lock_targets)
        self.assertTrue(txn.commit("paper-atcc").committed)
        self.assertEqual("1", manager.store.get("row").value)

    def test_delayed_write_apply_is_independent_of_optimized_profile(self):
        manager = AgentTransactionManager(delayed_write_apply_enabled=True)
        manager.register_object("row", "0", kind="row")
        txn = manager.begin(
            "delayed-write-ablation",
            {
                "paper_atcc": True,
                "retry_protection_mask": int(LockClass.COLD_WRITE),
            },
            strategy="paper-atcc",
        )

        txn.write("row", "1")

        self.assertTrue(txn.metadata.get("paper_atcc", False))
        self.assertTrue(txn.metadata["commit_admission_write_protection"])
        self.assertFalse(txn.context.held_write_locks)
        self.assertEqual({"row"}, txn.context.policy_write_lock_targets)
        self.assertTrue(txn.commit("paper-atcc").committed)
        self.assertEqual("1", manager.store.get("row").value)

    def test_late_phase_hot_write_is_protected_on_first_attempt(self):
        manager = AgentTransactionManager()
        manager.register_object("hot-row", "0", kind="row")
        for _ in range(8):
            manager.hotness_tracker.observe_access("hot-row")
        txn = manager.begin(
            "hot-write",
            {
                "paper_atcc": True,
                "retry_protection_mask": int(LockClass.HOT_WRITE),
            },
        )
        txn.enter_phase("commit")

        txn.write("hot-row", "1")

        self.assertIn("hot-row", txn.context.held_write_locks)
        self.assertTrue(txn.commit("paper-atcc").committed)

    def test_late_phase_cold_write_is_protected_when_action_selects_it(self):
        manager = AgentTransactionManager()
        manager.register_object("cold-row", "0", kind="row")
        txn = manager.begin(
            "cold-write",
            {
                "paper_atcc": True,
                "retry_protection_mask": int(LockClass.COLD_WRITE),
            },
        )
        txn.enter_phase("commit")

        txn.write("cold-row", "1")

        self.assertIn("cold-row", txn.context.held_write_locks)
        self.assertTrue(txn.commit("paper-atcc").committed)

    def test_late_phase_blind_write_rebases_after_pinned_snapshot(self):
        manager = AgentTransactionManager()
        manager.register_object("anchor", "0", kind="row")
        manager.register_object("target", "0", kind="row")
        agent = manager.begin(
            "blind-rebase",
            {
                "paper_atcc": True,
                "retry_protection_mask": int(LockClass.COLD_WRITE),
            },
        )
        agent.read("anchor")
        background = manager.begin("background", {"paper_atcc_backend": True})
        background.write("target", "background")
        self.assertTrue(background.commit("occ").committed)
        background_version = manager.store.get("target").version
        agent.enter_phase("commit")

        agent.write("target", "agent")

        self.assertEqual(background_version, agent.write_set["target"].base_version)
        self.assertTrue(agent.commit("paper-atcc").committed)
        self.assertEqual("agent", manager.store.get("target").value)

    def test_profiled_hot_planned_write_does_not_bypass_explore_policy(self):
        manager = AgentTransactionManager()
        manager.register_object("hot-target", "0", kind="row")
        manager.hotness_tracker.prime_accesses(["hot-target"] * 8)
        agent = manager.begin(
            "profiled-write",
            {
                "paper_atcc": True,
                "planned_write_targets": ["hot-target"],
            },
        )

        agent.read("hot-target")

        self.assertNotIn("hot-target", agent.context.held_write_locks)
        self.assertEqual(LockClass.NONE, agent.context.action.protected)
        agent.abort("test cleanup")

    def test_first_planned_write_read_refreshes_after_unrelated_snapshot_pin(self):
        manager = AgentTransactionManager()
        manager.register_object("anchor", "0", kind="row")
        manager.register_object("target", "0", kind="row")
        manager.hotness_tracker.prime_accesses(["target"] * 8)
        agent = manager.begin(
            "planned-read-refresh",
            {
                "paper_atcc": True,
                "planned_write_targets": ["target"],
            },
        )
        agent.read("anchor")
        background = manager.begin("background", {"paper_atcc_backend": True})
        background.write("target", "background")
        self.assertTrue(background.commit("occ").committed)
        current = manager.store.get("target")

        observed = agent.read("target")

        self.assertEqual(current.version, observed.version)
        self.assertEqual("background", observed.value)
        self.assertNotIn("target", agent.context.held_write_locks)
        agent.abort("test cleanup")

    def test_profiled_hot_read_does_not_bypass_explore_policy(self):
        manager = AgentTransactionManager()
        manager.register_object("hot-read", "0", kind="row")
        manager.hotness_tracker.prime_accesses(["hot-read"] * 8)
        agent = manager.begin("profiled-read", {"paper_atcc": True})

        agent.read("hot-read")

        self.assertNotIn("hot-read", agent.context.held_read_locks)
        self.assertEqual(LockClass.NONE, agent.context.action.protected)
        agent.abort("test cleanup")

    def test_profiled_shared_cold_write_does_not_bypass_explore_policy(self):
        manager = AgentTransactionManager()
        manager.register_object("shared-cold", "0", kind="row")
        manager.hotness_tracker.prime_transaction(["shared-cold"])
        manager.hotness_tracker.prime_transaction(["shared-cold"])
        agent = manager.begin(
            "shared-cold-write",
            {
                "paper_atcc": True,
                "planned_write_targets": ["shared-cold"],
            },
        )

        agent.read("shared-cold")

        self.assertNotIn("shared-cold", agent.context.held_write_locks)
        self.assertEqual(LockClass.NONE, agent.context.action.protected)
        agent.abort("test cleanup")

    def test_pure_policy_retry_does_not_promote_profiled_shared_write(self):
        manager = AgentTransactionManager(performance_guards_enabled=False)
        manager.register_object("shared-cold", "0", kind="row")
        manager.hotness_tracker.prime_transaction(["shared-cold"])
        manager.hotness_tracker.prime_transaction(["shared-cold"])
        agent = manager.begin(
            "pure-policy-retry-write",
            {
                "paper_atcc": True,
                "retry_count": 1,
            },
        )
        agent.context.phase = TransactionPhase.REFINE

        agent.write("shared-cold", "agent")

        self.assertNotIn("shared-cold", agent.context.held_write_locks)
        self.assertEqual(LockClass.NONE, agent.context.action.protected)
        agent.abort("test cleanup")

    def test_write_conflict_retry_eagerly_protects_only_failed_write_class(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        first = manager.begin(
            "write-retry",
            {
                "paper_atcc": True,
                "retry_count": 0,
            },
        )
        first.read("row")
        first.write("row", "agent-1")
        background = manager.begin(
            "background",
            {"paper_atcc_backend": True},
        )
        background.write("row", "background")
        self.assertTrue(background.commit("occ").committed)
        failed = first.commit("paper-atcc")
        self.assertFalse(failed.committed)
        self.assertEqual("cold-write", failed.conflict_details[0].protection_class)

        retry = manager.begin(
            "write-retry",
            {"paper_atcc": True, "retry_count": 1},
        )
        retry.write("row", "agent-2")

        self.assertIn("row", retry.context.held_write_locks)
        self.assertTrue(retry.commit("paper-atcc").committed)
        self.assertEqual("agent-2", manager.store.get("row").value)

    def test_write_conflict_retry_protects_exact_object_before_retry_read(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        first = manager.begin("read-before-write", {"paper_atcc": True})
        first.read("row")
        first.write("row", "agent-1")
        background = manager.begin("background", {"paper_atcc_backend": True})
        background.write("row", "background")
        self.assertTrue(background.commit("occ").committed)
        self.assertFalse(first.commit("paper-atcc").committed)

        # Exact object identity, rather than its possibly changing hot/cold
        # class, makes the retry acquire WLock at its first read.
        for _ in range(16):
            manager.hotness_tracker.observe_access("row")
        retry = manager.begin(
            "read-before-write",
            {"paper_atcc": True, "retry_count": 1},
        )
        retry.read("row")

        self.assertIn("row", retry.context.retry_conflict_write_targets)
        self.assertIn("row", retry.context.held_write_locks)
        retry.write("row", "agent-2")
        self.assertTrue(retry.commit("paper-atcc").committed)

    def test_pending_priority_reorders_only_after_threshold(self):
        manager = AgentTransactionManager()
        context = TransactionContext("priority-threshold", 0, 0)
        context.pending_request = "row"

        manager.atcc_locks.update_priority(context, 1)
        self.assertEqual(0, context.priority)
        self.assertEqual(
            0,
            manager.atcc_locks.snapshot_diagnostics().get("priority_reorders", 0),
        )

        manager.atcc_locks.update_priority(context, 2)
        self.assertEqual(2, context.priority)
        self.assertEqual(
            1,
            manager.atcc_locks.snapshot_diagnostics().get("priority_reorders", 0),
        )

    def test_traditional_manager_bypasses_paper_version_publication(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("traditional")
        txn.write("row", "1")

        self.assertTrue(txn.commit("occ").committed)

        self.assertFalse(manager.paper_versioning_enabled)
        diagnostics = manager.version_manager.snapshot_diagnostics()
        self.assertEqual(0, diagnostics["atomic_publishes"])

    def test_operation_cost_excludes_measured_lock_wait(self):
        manager = AgentTransactionManager()
        txn = manager.begin("cost-accounting")
        txn.context.blocked_time_ms = 7.0

        manager.interceptor.operation_finished(
            txn,
            elapsed_ms=10.0,
            blocked_before_ms=0.0,
        )

        self.assertAlmostEqual(3.0, txn.context.operation_cost_ms)
        self.assertEqual(1, txn.context.completed_operations)

    def test_blocked_backend_eventually_outranks_reasoning_interval(self):
        manager = PriorityManager()
        agent = TransactionContext("agent", 0, 1)
        agent.agent_cost_ms = 100.0
        backend = TransactionContext("backend", 0, 1, is_background=True)
        backend.blocked_time_ms = 1100.0

        self.assertGreater(manager.compute(backend), manager.compute(agent))

    def test_state_collector_uses_real_operation_and_phase_progress(self):
        manager = AgentTransactionManager()
        manager.register_object("a", "0", kind="row")
        manager.register_object("b", "0", kind="row")
        txn = manager.begin("state")
        txn.read("a")
        txn.enter_phase("refine")
        txn.read("a")
        txn.write("b", "1")

        state = manager.state_collector.snapshot(txn.context)

        self.assertEqual("refine", state.phase)
        self.assertEqual(1, state.completed_rounds)
        self.assertEqual(1.0 / 2.0, state.access_overlap_ratio)
        self.assertEqual(0.5, state.recent_write_ratio)

    def test_undo_log_recovers_install_without_commit_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "undo.jsonl"
            manager = AgentTransactionManager(undo_log_path=str(path))
            manager.register_object("a", "old-a", kind="row")
            manager.register_object("b", "old-b", kind="row")
            a = manager.store.get("a")
            b = manager.store.get("b")
            manager.undo_log.begin("crashed:0:0")
            manager.undo_log.update("crashed:0:0", object_id="a", old_value=a.value, old_version=a.version)
            manager.undo_log.update("crashed:0:0", object_id="b", old_value=b.value, old_version=b.version)
            self.assertTrue(
                manager.store.batch_put_if_version(
                    [("a", a.version), ("b", b.version)],
                    [("a", "new-a"), ("b", "new-b")],
                )
            )

            restarted = AgentTransactionManager(store=manager.store, undo_log_path=str(path))
            recovered = restarted.recover()

            self.assertEqual(["crashed:0:0"], recovered)
            self.assertEqual("old-a", restarted.store.get("a").value)
            self.assertEqual("old-b", restarted.store.get("b").value)

    def test_undo_log_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "undo.jsonl"
            manager = AgentTransactionManager(undo_log_path=str(path))
            manager.undo_log.begin("txn")
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("BEGIN", "BROKEN"), encoding="utf-8")

            with self.assertRaises(RuntimeError):
                AgentTransactionManager(undo_log_path=str(path))

    def test_paper_policy_changes_action_inside_transaction_phase(self):
        policy = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="refine", action=int(LockClass.COLD_READ))],
            generation=1,
        )
        manager = AgentTransactionManager(paper_policy=policy)
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("paper", {"paper_atcc": True})
        txn.read("row")

        txn.enter_phase("refine")

        self.assertEqual(LockClass.COLD_READ, txn.context.action.protected)
        self.assertIn("row", txn.context.held_read_locks)
        self.assertTrue(txn.commit("occ").committed)
        self.assertEqual(0, manager.atcc_locks.snapshot("row")["reader_count"])

    def test_policy_install_is_atomic_and_generation_monotonic(self):
        manager = AtomicPolicyManager(CompiledPhasePolicy(generation=1))
        updated = CompiledPhasePolicy(generation=2)
        manager.install(updated)
        self.assertIs(updated, manager.snapshot())
        with self.assertRaises(ValueError):
            manager.install(CompiledPhasePolicy(generation=2))

    def test_paper_policy_never_releases_protection_on_later_phase(self):
        policy = CompiledPhasePolicy(
            [
                CompiledPolicyEntry(phase="refine", action=int(LockClass.COLD_READ)),
                CompiledPolicyEntry(phase="commit", action=int(LockClass.NONE)),
            ],
            generation=1,
        )
        manager = AgentTransactionManager(paper_policy=policy)
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("monotonic", {"paper_atcc": True})
        txn.read("row")
        txn.enter_phase("refine")
        txn.enter_phase("commit")

        self.assertEqual(LockClass.COLD_READ, txn.context.action.protected)
        self.assertIn("row", txn.context.held_read_locks)

    def test_paper_runtime_records_transaction_internal_trajectory(self):
        policy = CompiledPhasePolicy(
            [
                CompiledPolicyEntry(phase="refine", action=int(LockClass.COLD_READ)),
                CompiledPolicyEntry(
                    phase="commit",
                    action=int(LockClass.COLD_READ | LockClass.COLD_WRITE),
                ),
            ],
            generation=1,
        )
        manager = AgentTransactionManager(paper_policy=policy)
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("trajectory", {"paper_atcc": True})
        txn.read("row")
        txn.enter_phase("refine")
        txn.enter_phase("commit")
        txn.write("row", "1")
        self.assertTrue(txn.commit("paper-atcc").committed)

        transitions = manager.trajectory_collector.snapshot()
        self.assertGreaterEqual(len(transitions), 3)
        self.assertFalse(transitions[0].done)
        self.assertTrue(transitions[-1].done)
        terminal = [transition for transition in transitions if transition.done]
        self.assertEqual(1, len(terminal))
        self.assertGreater(terminal[0].reward, 0.0)
        self.assertTrue(math.isfinite(terminal[0].reward))

    def test_system_performance_delta_rewards_throughput_and_tail_improvement(self):
        previous = PhaseAwareState(
            phase="refine",
            inter_round_interval_ms=10,
            read_set_size=2,
            write_set_size=1,
            read_set_growth=1,
            write_set_growth=1,
            access_overlap_ratio=0.5,
            completed_rounds=1,
            completed_operations=3,
            recent_write_ratio=0.3,
            hotspot_access_ratio=0.5,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=1,
            global_agent_task_throughput=50,
            global_agent_task_tail_latency_ms=100,
            global_conflict_abort_rate=0.5,
        )
        improved = dataclasses.replace(
            previous,
            global_agent_task_throughput=100,
            global_agent_task_tail_latency_ms=50,
            global_conflict_abort_rate=0.1,
        )
        regressed = dataclasses.replace(
            previous,
            global_agent_task_throughput=25,
            global_agent_task_tail_latency_ms=200,
            global_conflict_abort_rate=0.9,
        )
        self.assertGreater(system_performance_delta(previous, improved), 0.0)
        self.assertLess(system_performance_delta(previous, regressed), 0.0)

        background_improved = dataclasses.replace(
            previous,
            global_background_throughput=20,
            global_background_abort_rate=0.0,
        )
        background_regressed = dataclasses.replace(
            previous,
            global_background_throughput=0,
            global_background_abort_rate=1.0,
        )
        background_baseline = dataclasses.replace(
            previous,
            global_background_throughput=10,
            global_background_abort_rate=0.5,
        )
        self.assertGreater(
            system_performance_delta(background_baseline, background_improved), 0.0
        )
        self.assertLess(
            system_performance_delta(background_baseline, background_regressed), 0.0
        )

    def test_paper_reward_uses_normalized_restart_retry_lock_and_system_terms(self):
        state = PhaseAwareState(
            phase="commit",
            inter_round_interval_ms=10,
            read_set_size=1,
            write_set_size=1,
            read_set_growth=0,
            write_set_growth=1,
            access_overlap_ratio=0.5,
            completed_rounds=2,
            completed_operations=2,
            recent_write_ratio=0.5,
            hotspot_access_ratio=1.0,
            blocked_time_ms=5,
            retry_count=1,
            current_action=4,
            priority=3,
        )
        raw = PolicyTransition(
            "reward",
            state,
            4,
            0.0,
            state,
            True,
            commit_indicator=0,
            abort_indicator=1,
            operation_cost_ms=10,
            agent_cost_ms=20,
            retry_cost_ms=30,
            lock_wait_ms=5,
            new_lock_count=1,
            system_delta=-1,
        )
        externality = dataclasses.replace(
            raw,
            txn_id="reward-externality",
            background_blocked_ms_caused=50,
            background_aborts_caused=1,
            agent_blocked_ms_caused=25,
            agent_aborts_caused=1,
        )

        rows, report = apply_paper_rewards([raw, externality])

        self.assertLess(rows[0].reward, -80.0)
        self.assertLess(rows[1].reward, rows[0].reward)
        self.assertIn("C_restart", report["formula"])
        self.assertEqual(10.0, report["normalization"]["operation_cost_ms"])
        self.assertEqual(
            50.0, report["normalization"]["background_blocked_ms_caused"]
        )

    def test_discrete_ppo_compiles_learned_monotonic_action(self):
        state = PhaseAwareState(
            phase="refine",
            inter_round_interval_ms=20,
            read_set_size=4,
            write_set_size=1,
            read_set_growth=2,
            write_set_growth=1,
            access_overlap_ratio=0.5,
            completed_rounds=1,
            completed_operations=5,
            recent_write_ratio=0.2,
            hotspot_access_ratio=0.5,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=2,
        )
        transitions = [
            PolicyTransition("good", state, 3, 100.0, state, True, 0.5),
            PolicyTransition("bad", state, 0, -100.0, state, True, 0.5),
        ] * 8
        policy = DiscretePPOPolicy(seed=1)
        trainer = DiscretePPOTrainer(PPOConfig(learning_rate=0.1, epochs=20))

        report = trainer.train(policy, transitions)
        compiled = policy.compile(generation=1)

        self.assertEqual(16.0, report["transitions"])
        self.assertEqual(3, int(compiled.select(state).protected))
        nearby = dataclasses.replace(
            state,
            global_throughput=128.0,
            global_tail_latency_ms=512.0,
        )
        nearby_probabilities = policy.probabilities(nearby)
        self.assertGreater(nearby_probabilities[3], nearby_probabilities[0])

    def test_ppo_compilation_stays_within_observed_group_action_support(self):
        state = PhaseAwareState(
            phase="refine",
            inter_round_interval_ms=20,
            read_set_size=4,
            write_set_size=1,
            read_set_growth=2,
            write_set_growth=1,
            access_overlap_ratio=0.5,
            completed_rounds=1,
            completed_operations=5,
            recent_write_ratio=0.2,
            hotspot_access_ratio=0.5,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=2,
        )
        transitions = [
            PolicyTransition("occ", state, 0, 10.0, state, True, 0.5),
            PolicyTransition("hot-write", state, 4, 20.0, state, True, 0.5),
        ] * 8
        policy = DiscretePPOPolicy(seed=17)
        DiscretePPOTrainer(PPOConfig(epochs=4)).train(policy, transitions)

        compiled = policy.compile(generation=2)

        self.assertTrue(compiled.entries)
        self.assertTrue({entry.action for entry in compiled.entries} <= {0, 4})

    def test_ppo_compilation_projects_factorized_bits_to_best_supported_action(self):
        state = PhaseAwareState(
            phase="commit",
            inter_round_interval_ms=20,
            read_set_size=8,
            write_set_size=4,
            read_set_growth=0,
            write_set_growth=0,
            access_overlap_ratio=0.5,
            completed_rounds=2,
            completed_operations=12,
            recent_write_ratio=0.5,
            hotspot_access_ratio=0.75,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
        )
        transitions = []
        for index in range(8):
            transitions.extend(
                (
                    PolicyTransition(f"best-{index}", state, 1, 100.0, state, True, 0.25),
                    PolicyTransition(f"bit-2-{index}", state, 2, 40.0, state, True, 0.25),
                    PolicyTransition(f"bit-4-{index}", state, 4, 30.0, state, True, 0.25),
                    PolicyTransition(f"wide-{index}", state, 15, -100.0, state, True, 0.25),
                )
            )
        policy = DiscretePPOPolicy(seed=23)
        report = DiscretePPOTrainer(PPOConfig(epochs=8)).train(policy, transitions)

        compiled = policy.compile(generation=3)

        self.assertEqual(1, int(compiled.select(state).protected))
        self.assertEqual(
            "highest_normalized_return_supported_complete_action",
            report["supported_action_calibration"]["projection"],
        )

    def test_discrete_ppo_updates_each_state_group_independently(self):
        base = PhaseAwareState(
            phase="commit",
            inter_round_interval_ms=0,
            read_set_size=2,
            write_set_size=1,
            read_set_growth=0,
            write_set_growth=0,
            access_overlap_ratio=0,
            completed_rounds=2,
            completed_operations=3,
            recent_write_ratio=0.5,
            hotspot_access_ratio=0,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
        )
        low_pressure = dataclasses.replace(base, global_abort_rate=0.0)
        high_pressure = dataclasses.replace(base, global_abort_rate=0.9)
        transitions = []
        for index in range(24):
            transitions.extend(
                (
                    PolicyTransition(f"low-good-{index}", low_pressure, 0, 100.0, low_pressure, True, 0.5),
                    PolicyTransition(f"low-bad-{index}", low_pressure, 1, -100.0, low_pressure, True, 0.5),
                    PolicyTransition(f"high-good-{index}", high_pressure, 1, 100.0, high_pressure, True, 0.5),
                    PolicyTransition(f"high-bad-{index}", high_pressure, 0, -100.0, high_pressure, True, 0.5),
                )
            )
        policy = DiscretePPOPolicy(seed=7)
        DiscretePPOTrainer(
            PPOConfig(learning_rate=0.0, group_learning_rate=0.1, epochs=12)
        ).train(policy, transitions)

        self.assertGreater(
            policy.probabilities(low_pressure)[0],
            policy.probabilities(low_pressure)[1],
        )
        self.assertGreater(
            policy.probabilities(high_pressure)[1],
            policy.probabilities(high_pressure)[0],
        )

    def test_exploration_policy_keeps_stay_path_well_sampled(self):
        state = PhaseAwareState(
            phase="refine",
            inter_round_interval_ms=0,
            read_set_size=1,
            write_set_size=1,
            read_set_growth=1,
            write_set_growth=1,
            access_overlap_ratio=0,
            completed_rounds=1,
            completed_operations=2,
            recent_write_ratio=0.5,
            hotspot_access_ratio=0,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
        )
        probabilities = DiscretePPOPolicy(
            seed=1, stay_probability=0.5
        ).probabilities(state)
        self.assertAlmostEqual(0.53125, probabilities[0])
        self.assertAlmostEqual(0.03125, probabilities[15])
        self.assertAlmostEqual(1.0, sum(probabilities))

        transition = PolicyTransition(
            "behavior",
            state,
            15,
            0.0,
            state,
            True,
            probabilities[15],
        )
        reconstructed = reconstruct_behavior_distribution(transition)
        self.assertIsNotNone(reconstructed)
        self.assertAlmostEqual(probabilities[0], reconstructed[0])
        self.assertAlmostEqual(probabilities[15], reconstructed[15])

    def test_policy_iteration_exploration_records_exact_distribution(self):
        state = PhaseAwareState(
            phase="commit",
            inter_round_interval_ms=0,
            read_set_size=1,
            write_set_size=1,
            read_set_growth=0,
            write_set_growth=1,
            access_overlap_ratio=1,
            completed_rounds=2,
            completed_operations=2,
            recent_write_ratio=0.5,
            hotspot_access_ratio=1,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
        )
        base = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="commit", current_action=0, action=5)]
        )
        exploring = EpsilonGreedyPolicy(base, seed=7, epsilon=0.2)

        _selected, probabilities = exploring.select_with_distribution(state)
        transition = PolicyTransition(
            "iteration",
            state,
            5,
            0.0,
            state,
            True,
            probabilities[5],
            behavior_action_probabilities=probabilities,
        )
        reconstructed = reconstruct_behavior_distribution(transition)

        self.assertAlmostEqual(0.8125, probabilities[5])
        self.assertAlmostEqual(0.0125, probabilities[0])
        self.assertAlmostEqual(1.0, sum(probabilities))
        self.assertEqual(
            tuple(probabilities[action] for action in range(16)),
            tuple(reconstructed[action] for action in range(16)),
        )

    def test_runtime_metrics_publish_logical_agent_task_latency(self):
        manager = AgentTransactionManager()
        manager.note_agent_task_outcome(committed=True, latency_ms=123.0)

        metrics = manager.paper_runtime_metrics()

        self.assertGreater(metrics["agent_task_throughput"], 0.0)
        self.assertEqual(123.0, metrics["agent_task_average_latency_ms"])
        self.assertEqual(123.0, metrics["agent_task_tail_latency_ms"])

    def test_runtime_metrics_publish_background_throughput_and_abort_rate(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        committed = manager.begin(
            "background-commit", {"paper_atcc_backend": True}
        )
        committed.write("row", "1")
        self.assertTrue(committed.commit("occ").committed)
        aborted = manager.begin("background-abort", {"paper_atcc_backend": True})
        aborted.abort("lock-timeout", strategy="occ")

        metrics = manager.paper_runtime_metrics()

        self.assertGreater(metrics["background_throughput"], 0.0)
        self.assertEqual(0.5, metrics["background_abort_rate"])

    def test_measurement_reset_preserves_versions_but_clears_event_counters(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        background = manager.begin(
            "warmup-background",
            {"paper_atcc_backend": True, "planned_write_targets": ["row"]},
        )
        background.write("row", "1")
        self.assertTrue(background.commit("occ").committed)
        self.assertGreater(
            manager.version_manager.snapshot_diagnostics()["atomic_publishes"],
            0,
        )

        manager.reset_measurement_diagnostics()

        self.assertEqual("1", manager.store.get("row").value)
        self.assertEqual(
            0,
            manager.version_manager.snapshot_diagnostics()["atomic_publishes"],
        )
        self.assertEqual(
            0,
            manager.atcc_locks.snapshot_diagnostics().get(
                "background_fast_publishes", 0
            ),
        )

    def test_compiled_policy_occ_guard_yields_after_conflict_pressure(self):
        policy = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="refine", current_action=0, action=13)],
            generation=1,
            occ_cold_start_guard=True,
        )
        state = PhaseAwareState(
            phase="refine",
            inter_round_interval_ms=10,
            read_set_size=2,
            write_set_size=1,
            read_set_growth=1,
            write_set_growth=1,
            access_overlap_ratio=0.5,
            completed_rounds=1,
            completed_operations=3,
            recent_write_ratio=0.3,
            hotspot_access_ratio=1.0,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=1,
        )

        self.assertEqual(0, int(policy.select(state).protected))
        sparse_pressure = dataclasses.replace(
            state,
            global_abort_rate=0.04,
            global_conflict_abort_rate=0.04,
            global_background_abort_rate=0.04,
        )
        self.assertEqual(0, int(policy.select(sparse_pressure).protected))
        sustained_pressure = dataclasses.replace(
            state,
            global_abort_rate=0.06,
            global_conflict_abort_rate=0.06,
        )
        self.assertEqual(13, int(policy.select(sustained_pressure).protected))
        pressured = dataclasses.replace(state, global_waiter_count=1)
        self.assertEqual(13, int(policy.select(pressured).protected))
        restored = CompiledPhasePolicy.from_dict(policy.to_dict())
        self.assertTrue(restored.occ_cold_start_guard)

    def test_compiled_policy_uses_nearest_action_group_medoid(self):
        entries = [
            CompiledPolicyEntry(phase="commit", action=3, throughput_bucket=1),
            CompiledPolicyEntry(phase="commit", action=3, throughput_bucket=2),
            CompiledPolicyEntry(phase="commit", action=1, throughput_bucket=3),
        ]
        policy = CompiledPhasePolicy(entries, generation=1)
        state = PhaseAwareState(
            phase="commit",
            inter_round_interval_ms=0,
            read_set_size=0,
            write_set_size=0,
            read_set_growth=0,
            write_set_growth=0,
            access_overlap_ratio=0,
            completed_rounds=0,
            completed_operations=0,
            recent_write_ratio=0,
            hotspot_access_ratio=0,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
            global_throughput=64,
        )

        self.assertEqual(1, int(policy.select(state).protected))

    def test_compiled_policy_does_not_fallback_across_phases(self):
        policy = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="refine", current_action=0, action=3)],
            generation=1,
        )
        state = PhaseAwareState(
            phase="explore",
            inter_round_interval_ms=0,
            read_set_size=1,
            write_set_size=0,
            read_set_growth=1,
            write_set_growth=0,
            access_overlap_ratio=0,
            completed_rounds=0,
            completed_operations=1,
            recent_write_ratio=0,
            hotspot_access_ratio=0,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
        )

        self.assertEqual(0, int(policy.select(state).protected))

    def test_compiled_policy_fallback_keeps_conflict_states_separate(self):
        entries = [
            CompiledPolicyEntry(
                phase="commit",
                action=0,
                read_set_bucket=1,
                write_set_bucket=1,
                hotspot_bucket=0,
                abort_bucket=0,
                throughput_bucket=1,
            ),
            CompiledPolicyEntry(
                phase="commit",
                action=3,
                read_set_bucket=1,
                write_set_bucket=1,
                hotspot_bucket=3,
                abort_bucket=3,
                throughput_bucket=1,
            ),
        ]
        policy = CompiledPhasePolicy(entries, generation=1)
        base = PhaseAwareState(
            phase="commit",
            inter_round_interval_ms=0,
            read_set_size=1,
            write_set_size=1,
            read_set_growth=0,
            write_set_growth=0,
            access_overlap_ratio=0,
            completed_rounds=2,
            completed_operations=2,
            recent_write_ratio=0.5,
            hotspot_access_ratio=0,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
            global_abort_rate=0,
            global_throughput=64,
        )
        self.assertEqual(0, int(policy.select(base).protected))
        pressured = dataclasses.replace(
            base,
            hotspot_access_ratio=1.0,
            global_abort_rate=1.0,
        )
        self.assertEqual(3, int(policy.select(pressured).protected))

    def test_compiled_policy_declares_tm_owned_priority(self):
        payload = CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="commit", action=3)], generation=7
        ).to_dict()
        self.assertEqual("lock_protection_mask_4bit", payload["action_space"])
        self.assertEqual("transaction_manager_formula", payload["priority_control"])
        self.assertFalse(payload["priority_is_policy_action"])
        self.assertFalse(payload["future_access_plan_features"])
        self.assertNotIn("priority", payload["entries"][0])

    def test_switching_policy_is_invariant_to_tm_priority(self):
        base = PhaseAwareState(
            phase="commit",
            inter_round_interval_ms=10,
            read_set_size=2,
            write_set_size=1,
            read_set_growth=1,
            write_set_growth=1,
            access_overlap_ratio=0.5,
            completed_rounds=2,
            completed_operations=4,
            recent_write_ratio=0.5,
            hotspot_access_ratio=0.5,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
            global_abort_rate=0.2,
            global_conflict_abort_rate=0.2,
        )
        prioritized = dataclasses.replace(base, priority=10_000)
        policy = CompiledPhasePolicy(
            [
                CompiledPolicyEntry(
                    phase="commit",
                    action=5,
                    priority_bucket=15,
                )
            ],
            generation=1,
        )

        self.assertEqual(state_key(base), state_key(prioritized))
        self.assertEqual(policy_group_key(base), policy_group_key(prioritized))
        self.assertEqual(policy.select(base), policy.select(prioritized))

    def test_compiled_ppo_policy_refines_nonexact_table_lookup(self):
        state = PhaseAwareState(
            phase="refine",
            inter_round_interval_ms=0,
            read_set_size=1,
            write_set_size=0,
            read_set_growth=1,
            write_set_growth=0,
            access_overlap_ratio=0,
            completed_rounds=1,
            completed_operations=1,
            recent_write_ratio=0,
            hotspot_access_ratio=0,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
        )
        policy = DiscretePPOPolicy(seed=3)
        policy.prototypes["refine"] = state
        policy.prototype_counts["refine"] = 1
        policy.actor_weights[0][0] = 10.0
        policy.actor_weights[2][0] = 10.0
        compiled = policy.compile(generation=9, refinement_distance_threshold=0.0)
        nearby = dataclasses.replace(state, global_throughput=64.0)
        restored = CompiledPhasePolicy.from_dict(compiled.to_dict())

        self.assertTrue(restored.to_dict()["selective_refinement"])
        self.assertEqual(5, int(restored.select(nearby).protected))

    def test_ppo_normalizes_each_exploration_run_without_workload_labels(self):
        state = PhaseAwareState(
            phase="refine",
            inter_round_interval_ms=0,
            read_set_size=0,
            write_set_size=0,
            read_set_growth=0,
            write_set_growth=0,
            access_overlap_ratio=0,
            completed_rounds=0,
            completed_operations=0,
            recent_write_ratio=0,
            hotspot_access_ratio=0,
            blocked_time_ms=0,
            retry_count=0,
            current_action=0,
            priority=0,
        )
        samples = [
            (PolicyTransition("a", state, 0, 0, state, True, 1, "run-a"), -1000.0),
            (PolicyTransition("b", state, 0, 0, state, True, 1, "run-a"), 1000.0),
            (PolicyTransition("c", state, 0, 0, state, True, 1, "run-b"), -1.0),
            (PolicyTransition("d", state, 0, 0, state, True, 1, "run-b"), 1.0),
        ]

        normalized, statistics = normalize_returns_by_source(samples)

        self.assertEqual({"run-a", "run-b"}, set(statistics))
        self.assertAlmostEqual(-1.0, normalized[0])
        self.assertAlmostEqual(1.0, normalized[1])
        self.assertAlmostEqual(-1.0, normalized[2])
        self.assertAlmostEqual(1.0, normalized[3])

    def test_core_runtime_and_workloads_are_runnable(self):
        report = run_smoke_checks()

        self.assertTrue(report["ok"])
        self.assertEqual("dbx1000", report["native_backend"])
        self.assertEqual("1", report["runtime_counter"])
        self.assertTrue(report["ycsb_task_committed"])
        self.assertTrue(report["tpcc_task_committed"])

    def test_conflicting_transactions_abort_on_stale_version(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", "0", kind="counter")

        first = manager.begin("first")
        second = manager.begin("second")
        first.read("counter")
        first.write("counter", "1")
        second.read("counter")
        second.write("counter", "1")

        first_result = first.commit("occ")
        second_result = second.commit("occ")

        self.assertTrue(first_result.committed)
        self.assertFalse(second_result.committed)
        self.assertEqual("aborted", second_result.state.value)
        self.assertIn("counter", second_result.conflict_object_ids)

    def test_workload_task_can_commit(self):
        manager = AgentTransactionManager()
        workload = build_workload("ycsb", "low")
        register_workload(manager, workload)
        task = workload.generate_tasks(1, seed=1)[0]

        result = execute_task(manager, task, cc="dynamic-atcc")

        self.assertTrue(result.committed)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from agent.runtime import AgentTransactionManager
from agent.workloads.credit_review import CreditReviewConfig, CreditReviewWorkload
from scripts.unified_trace.run_credit_review_experiment import (
    acquire_policy_commit_batch,
    credit_retry_delay_ms,
    execute_attempt,
    summarize,
    transaction_metadata,
)


class CreditReviewWorkloadTests(unittest.TestCase):
    def test_credit_retry_delay_matches_paper_retry_semantics(self):
        self.assertEqual(
            0,
            credit_retry_delay_ms(task_id="credit-retry", attempt=0),
        )
        retry_delay = credit_retry_delay_ms(
            task_id="credit-retry",
            attempt=1,
        )
        self.assertGreaterEqual(retry_delay, 500)
        self.assertLessEqual(retry_delay, 5000)
        self.assertEqual(
            retry_delay,
            credit_retry_delay_ms(task_id="credit-retry", attempt=1),
        )
        self.assertEqual(
            0,
            credit_retry_delay_ms(
                task_id="credit-retry",
                attempt=1,
                retry_delay_scale=0.0,
            ),
        )

    def test_retry_replanning_delay_precedes_attempt(self):
        workload = CreditReviewWorkload(
            CreditReviewConfig(company_count=16, reasoning_scale=0.0, commit_apply_ms=0)
        )
        manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
        workload.register(manager)
        task = workload.task_for(seed=19, worker_id=0, sequence=0)

        with mock.patch(
            "scripts.unified_trace.run_credit_review_experiment.sleep_for_reasoning"
        ) as sleep:
            result, _execution = execute_attempt(
                manager,
                workload,
                task,
                task_id="credit-retry-silo",
                system="silo",
                attempt=1,
            )

        self.assertTrue(result.committed)
        sleep.assert_called_once_with(
            credit_retry_delay_ms(task_id="credit-retry-silo", attempt=1)
        )

    def test_cursor_reveals_targets_only_as_operations_execute(self):
        workload = CreditReviewWorkload(
            CreditReviewConfig(company_count=16, reasoning_scale=0.0, commit_apply_ms=0)
        )
        manager = AgentTransactionManager(record_traces=True, collect_trajectories=False)
        workload.register(manager)
        task = workload.task_for(seed=7, worker_id=0, sequence=0)
        cursor = workload.cursor(task, sleep_fn=lambda _seconds: None)
        self.assertEqual((), cursor.snapshot().revealed_targets)
        metadata = transaction_metadata("silo", retry_count=0)
        self.assertEqual([], metadata["planned_write_targets"])
        self.assertEqual("online_observed", metadata["access_set_visibility"])
        txn = manager.begin("credit-test", metadata, snapshot_object_ids=None, strategy="silo")
        observed_prefixes = []
        observed_commit_batches = []

        def observe(_txn, _kind, target):
            observed_prefixes.append((target, tuple(cursor.revealed_targets)))

        def observe_commit_batch(_txn, targets):
            observed_commit_batches.append((targets, tuple(cursor.revealed_targets)))
            return 1.25

        execution = cursor.execute(
            txn,
            before_access=observe,
            before_commit_batch=observe_commit_batch,
        )
        self.assertTrue(txn.commit("silo").committed)
        self.assertGreaterEqual(execution.operation_count, 13)
        first_branch = next(
            row for row in observed_prefixes if row[0].startswith("credit:policy:")
        )
        self.assertIn(f"credit:company:{task.company_id}:profile", first_branch[1])
        self.assertEqual(1, len(observed_commit_batches))
        commit_targets, observed_before_commit = observed_commit_batches[0]
        self.assertIn(f"credit:company:{task.company_id}:limit", commit_targets)
        self.assertNotIn(f"credit:company:{task.company_id}:limit", observed_before_commit)
        self.assertEqual(1.25, execution.commit_admission_wait_ms)
        trace = manager.traces()[-1]
        self.assertEqual([], trace["metadata"]["planned_write_targets"])
        self.assertEqual((), tuple(trace["metadata"]["_planned_snapshot_object_ids"]))

    def test_all_systems_execute_without_declared_targets(self):
        workload = CreditReviewWorkload(
            CreditReviewConfig(company_count=16, reasoning_scale=0.0, commit_apply_ms=0)
        )
        for system in ("2pl-wait-die", "bamboo", "silo", "polaris", "paper-atcc"):
            with self.subTest(system=system):
                manager = AgentTransactionManager(record_traces=True, collect_trajectories=False)
                workload.register(manager)
                task = workload.task_for(seed=11, worker_id=0, sequence=1)
                result, execution = execute_attempt(
                    manager,
                    workload,
                    task,
                    task_id=f"credit-{system}",
                    system=system,
                    attempt=0,
                )
                self.assertTrue(result.committed)
                self.assertGreater(execution.reasoning_tokens, 0)
                trace = manager.traces()[-1]
                self.assertEqual([], trace["metadata"]["planned_write_targets"])
                self.assertEqual(
                    "policy_commit_batch" if system == "paper-atcc" else "none",
                    trace["metadata"]["admission_scope"],
                )

    def test_aegis_main_path_does_not_use_workload_specific_admission(self):
        workload = CreditReviewWorkload(
            CreditReviewConfig(company_count=16, reasoning_scale=0.0, commit_apply_ms=0)
        )
        manager = AgentTransactionManager(record_traces=True, collect_trajectories=False)
        workload.register(manager)
        task = workload.task_for(seed=17, worker_id=0, sequence=0)

        with mock.patch(
            "scripts.unified_trace.run_credit_review_experiment.acquire_observed_commit_admission"
        ) as workload_gate:
            result, _execution = execute_attempt(
                manager,
                workload,
                task,
                task_id="credit-paper-policy-only",
                system="paper-atcc",
                attempt=0,
            )

        self.assertTrue(result.committed)
        workload_gate.assert_not_called()
        self.assertEqual(
            "policy_commit_batch",
            manager.traces()[-1]["metadata"]["admission_scope"],
        )

    def test_policy_commit_batch_uses_hotness_not_object_names(self):
        manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
        manager.register_object("generic-hot", "0", kind="row")
        manager.register_object("generic-cold", "0", kind="row")
        manager.register_object("generic-future-blind", "0", kind="row")
        for _ in range(16):
            manager.hotness_tracker.observe_access("generic-hot")
        for _ in range(2):
            manager.hotness_tracker.observe_contention(
                "generic-hot", "validation-failure"
            )
        txn = manager.begin(
            "generic-policy-batch",
            transaction_metadata("paper-atcc", retry_count=0),
            strategy="paper-atcc",
        )
        acquire_policy_commit_batch(
            manager,
            txn,
            ("generic-cold", "generic-hot", "generic-future-blind"),
        )

        self.assertEqual(
            ["generic-hot"],
            [
                target
                for target, _gate in txn.metadata["_observed_hotspot_admissions"]
            ],
        )
        self.assertEqual(set(), txn.context.held_write_locks)
        txn.abort("test complete", strategy="paper-atcc")

    def test_policy_commit_batch_admits_materialized_hot_blind_write(self):
        manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
        manager.register_object("generic-observed", "0", kind="row")
        manager.register_object("generic-future-blind", "0", kind="row")
        for _ in range(16):
            manager.hotness_tracker.observe_access("generic-observed")
        for _ in range(16):
            manager.hotness_tracker.observe_access("generic-future-blind")
        for _ in range(2):
            manager.hotness_tracker.observe_contention(
                "generic-future-blind", "validation-failure"
            )
        txn = manager.begin(
            "generic-observed-only",
            transaction_metadata("paper-atcc", retry_count=0),
            strategy="paper-atcc",
        )
        acquire_policy_commit_batch(
            manager,
            txn,
            ("generic-observed", "generic-future-blind"),
        )

        self.assertEqual(
            ["generic-future-blind"],
            [
                target
                for target, _gate in txn.metadata["_observed_hotspot_admissions"]
            ],
        )
        txn.abort("test complete", strategy="paper-atcc")

    def test_observed_commit_admission_ablation_admits_the_observed_suffix(self):
        workload = CreditReviewWorkload(
            CreditReviewConfig(company_count=16, reasoning_scale=0.0, commit_apply_ms=0)
        )
        manager = AgentTransactionManager(record_traces=True, collect_trajectories=False)
        workload.register(manager)
        task = workload.task_for(seed=17, worker_id=0, sequence=0)
        admission_calls = []
        acquire = manager.acquire_hotspot_admission

        def audited_acquire(txn, object_ids, *, timeout_s=5.0):
            targets = tuple(object_ids)
            admission_calls.append(targets)
            self.assertGreaterEqual(len(targets), 1)
            return acquire(txn, targets, timeout_s=timeout_s)

        manager.acquire_hotspot_admission = audited_acquire
        result, _execution = execute_attempt(
            manager,
            workload,
            task,
            task_id="credit-paper-observed",
            system="paper-atcc",
            attempt=0,
            observed_commit_admission=True,
        )
        self.assertTrue(result.committed)
        self.assertEqual(1, len(admission_calls))
        self.assertTrue(
            any(target.startswith("credit:compliance:") for target in admission_calls[0])
        )
        trace = manager.traces()[-1]
        self.assertEqual([], trace["metadata"]["planned_write_targets"])
        self.assertEqual("observed_commit_suffix", trace["metadata"]["admission_scope"])

    def test_external_service_order_is_identical_with_and_without_admission(self):
        workload = CreditReviewWorkload(
            CreditReviewConfig(company_count=16, reasoning_scale=0.0, commit_apply_ms=10)
        )
        task = workload.task_for(seed=13, worker_id=0, sequence=0)

        for use_admission in (False, True):
            with self.subTest(use_admission=use_admission):
                manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
                workload.register(manager)
                events = []
                cursor = workload.cursor(
                    task,
                    sleep_fn=lambda _seconds: events.append("service"),
                )
                txn = manager.begin(
                    f"credit-order-{use_admission}",
                    transaction_metadata("silo", retry_count=0),
                    snapshot_object_ids=None,
                    strategy="silo",
                )

                def observe(_txn, kind, target):
                    if kind == "read" and target.endswith(":limit"):
                        events.append("limit-read")

                before_commit = (
                    (lambda _txn, _targets: events.append("admission") or 0.0)
                    if use_admission
                    else None
                )
                cursor.execute(
                    txn,
                    before_access=observe,
                    before_commit_batch=before_commit,
                )
                self.assertTrue(txn.commit("silo").committed)
                service_index = max(index for index, value in enumerate(events) if value == "service")
                limit_index = events.index("limit-read")
                self.assertLess(service_index, limit_index)
                if use_admission:
                    self.assertLess(events.index("admission"), limit_index)

    def test_observed_admissions_accumulate_and_release(self):
        manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
        txn = manager.begin("admission-owner", {}, strategy="silo")
        self.assertTrue(manager.acquire_hotspot_admission(txn, ("a",)))
        self.assertTrue(manager.acquire_hotspot_admission(txn, ("b",)))
        self.assertEqual(
            ["a", "b"],
            [target for target, _gate in txn.metadata["_observed_hotspot_admissions"]],
        )
        manager.release_hotspot_admission(txn)
        other = manager.begin("admission-successor", {}, strategy="silo")
        self.assertTrue(manager.acquire_hotspot_admission(other, ("a", "b")))
        manager.release_hotspot_admission(other)

    def test_summary_contains_only_plot_level_statistics(self):
        rows = []
        for repeat, tps in ((0, 10.0), (1, 14.0)):
            rows.append(
                {
                    "clients": 8,
                    "cc": "silo",
                    "system": "Silo",
                    "seed": 100 + repeat,
                    "status": "ok",
                    "agent_tps": tps,
                    "commit_rate": 0.8,
                    "p50_latency_ms": 10.0,
                    "p95_latency_ms": 20.0,
                    "p99_latency_ms": 30.0,
                    "wasted_reasoning_ms_per_commit": 4.0,
                    "wasted_tokens_per_commit": 100.0,
                    "useful_tokens_per_commit": 5000.0,
                    "total_tokens_per_commit": 5100.0,
                    "avg_operations_per_attempt": 13.0,
                }
            )
        summary = summarize(rows)
        self.assertEqual(1, len(summary))
        self.assertEqual(12.0, summary[0]["agent_tps_mean"])
        self.assertEqual(2, summary[0]["n_seeds"])
        self.assertNotIn("error", summary[0])


if __name__ == "__main__":
    unittest.main()

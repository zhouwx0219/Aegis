import io
import json
import tempfile
import unittest
from pathlib import Path

from agent.evaluation.atcc_schema import atcc_artifact_schema_status
from agent.evaluation.atcc_retry_experiment import (
    _agent_phase_for_task,
    RetryRunSummary,
    aggregate_retry_runs,
    main,
    run_retry_matrix,
)
from agent.workloads import TPCCConfig, YCSBConfig, build_agent_workload


class RetryExperimentMetricTests(unittest.TestCase):
    def test_retry_matrix_reports_agent_latency_and_cost_metrics(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=4,
                field_count=1,
                requests_per_task=2,
                candidates_per_task=2,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
            ),
        )
        runs = run_retry_matrix(
            workload,
            ("occ",),
            workload_kind="ycsb",
            policy_variant="phase-rl",
            task_count=3,
            seed=7,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=2,
            tokens_per_operation=10.0,
        )

        run = runs[0]
        row = run.to_dict()
        aggregate = aggregate_retry_runs(runs)[0]

        self.assertEqual(len(run.task_latencies_s), 3)
        self.assertEqual(run.task_operation_counts, (4, 4, 4))
        self.assertGreater(row["estimated_tokens"], 0.0)
        self.assertIn("agent_latency_p95_s", row)
        self.assertIn("estimated_wasted_tokens_per_task", row)
        self.assertIn("prelock_queue_depth_avg", row)
        self.assertIn("prelock_handoff_per_task", row)
        self.assertEqual(len(aggregate["task_latencies_s"]), 3)
        self.assertEqual(aggregate["task_operation_counts"], [4, 4, 4])
        self.assertIn("agent_latency_p99_s", aggregate)
        self.assertIn("estimated_tokens_per_task", aggregate)
        self.assertIn("prelock_queue_depth_avg", aggregate)
        self.assertIn("prelock_handoff_count", aggregate)

    def test_retry_aggregate_reports_lock_queue_and_committing_metrics(self):
        runs = (
            RetryRunSummary(
                workload="agent-ycsb-semantic",
                strategy="adaptive-op-strict",
                policy_variant="phase-rl",
                seed=1,
                task_count=2,
                workers=1,
                agent_slots=1,
                agent_admission_mode="before-begin",
                max_attempts=2,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                committed_tasks=2,
                final_failed_tasks=0,
                rejected_tasks=0,
                total_attempts=2,
                conflict_aborts=0,
                conflict_object_counts={},
                conflict_object_class_counts={},
                operation_policy_counts={"pessimistic": 2},
                operation_rule_counts={"phase-atcc-commit-lock-hot-writes": 2},
                action_counts={"direct": 2},
                prelock_wait_s=0.10,
                elapsed_s=1.0,
                prelock_queue_depth_sum=4.0,
                prelock_queue_depth_observations=2,
                prelock_queue_depth_max=3,
                prelock_handoff_count=1,
                prelock_committing_enters=2,
                prelock_committing_exits=2,
            ),
            RetryRunSummary(
                workload="agent-ycsb-semantic",
                strategy="adaptive-op-strict",
                policy_variant="phase-rl",
                seed=2,
                task_count=2,
                workers=1,
                agent_slots=1,
                agent_admission_mode="before-begin",
                max_attempts=2,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                committed_tasks=2,
                final_failed_tasks=0,
                rejected_tasks=0,
                total_attempts=2,
                conflict_aborts=0,
                conflict_object_counts={},
                conflict_object_class_counts={},
                operation_policy_counts={"pessimistic": 1},
                operation_rule_counts={"phase-atcc-commit-lock-hot-writes": 1},
                action_counts={"direct": 2},
                prelock_wait_s=0.20,
                elapsed_s=1.0,
                prelock_queue_depth_sum=2.0,
                prelock_queue_depth_observations=1,
                prelock_queue_depth_max=2,
                prelock_handoff_count=2,
                prelock_committing_enters=1,
                prelock_committing_exits=1,
            ),
        )

        aggregate = aggregate_retry_runs(runs)[0]

        self.assertEqual(aggregate["prelock_queue_depth_observations"], 3)
        self.assertAlmostEqual(aggregate["prelock_queue_depth_avg"], 2.0)
        self.assertEqual(aggregate["prelock_queue_depth_max"], 3)
        self.assertEqual(aggregate["prelock_handoff_count"], 3)
        self.assertAlmostEqual(aggregate["prelock_handoff_per_task"], 0.75)
        self.assertEqual(aggregate["prelock_committing_enters"], 3)
        self.assertEqual(aggregate["prelock_committing_exits"], 3)

    def test_retry_aggregate_charges_tokens_for_lease_refresh_regeneration(self):
        runs = (
            RetryRunSummary(
                workload="agent-ycsb-semantic",
                strategy="adaptive-op-strict",
                policy_variant="phase-rl",
                seed=1,
                task_count=2,
                workers=1,
                agent_slots=1,
                agent_admission_mode="before-begin",
                max_attempts=2,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                committed_tasks=2,
                final_failed_tasks=0,
                rejected_tasks=0,
                total_attempts=2,
                conflict_aborts=0,
                conflict_object_counts={},
                conflict_object_class_counts={},
                operation_policy_counts={"pessimistic": 2},
                operation_rule_counts={"phase-atcc-commit-lock-hot-writes": 2},
                action_counts={"direct": 2},
                prelock_wait_s=0.0,
                elapsed_s=1.0,
                task_operation_counts=(4, 4),
                tokens_per_operation=10.0,
                estimated_tokens=80.0,
                estimated_wasted_tokens=0.0,
                lease_refresh_regenerations=1,
            ),
        )

        aggregate = aggregate_retry_runs(runs)[0]

        self.assertEqual(aggregate["lease_refresh_regenerations"], 1)
        self.assertEqual(aggregate["lease_refresh_regenerations_per_task"], 0.5)
        self.assertEqual(aggregate["estimated_refresh_tokens"], 40.0)
        self.assertEqual(aggregate["estimated_refresh_tokens_per_task"], 20.0)
        self.assertEqual(aggregate["estimated_tokens"], 120.0)
        self.assertEqual(aggregate["estimated_tokens_per_task"], 60.0)
        self.assertEqual(aggregate["estimated_wasted_tokens"], 40.0)
        self.assertEqual(aggregate["estimated_wasted_tokens_per_task"], 20.0)

    def test_agent_phase_sequence_advances_by_retry_attempt(self):
        workload = build_agent_workload(
            "tpcc",
            "semantic",
            tpcc_config=TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=4,
                order_lines=2,
                candidates_per_task=1,
                transaction_mix=(("new_order", 1.0),),
            ),
        )
        task = workload.generate_tasks(1, seed=3)[0]

        self.assertEqual(
            task.context["agent_phase_sequence"],
            ("explore", "refine", "commit"),
        )
        self.assertEqual(_agent_phase_for_task(task, 0), "explore")
        self.assertEqual(_agent_phase_for_task(task, 1), "refine")
        self.assertEqual(_agent_phase_for_task(task, 2), "commit")
        self.assertEqual(_agent_phase_for_task(task, 99), "commit")

    def test_retry_cli_reports_policy_artifact_schema_compatibility(self):
        artifact = {
            "artifact_type": "phase-aware-atcc-policy-artifact",
            "artifact_version": 2,
            "atcc_state_schema": {
                "name": "phase-aware-atcc-object-class-state",
                "version": 2,
                "dimensions": [
                    "workload",
                    "task",
                    "class",
                    "phase",
                    "reads",
                    "writes",
                    "hotR",
                    "hotW",
                    "retry",
                    "interval",
                    "priority",
                    "globalObs",
                    "globalAbort",
                    "globalLockWait",
                    "globalLatency",
                    "intent",
                ],
            },
            "operation_policy_table": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "policy.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            stdout = io.StringIO()

            exit_code = main(
                [
                    "--workload",
                    "ycsb",
                    "--strategies",
                    "occ",
                    "--policy-variant",
                    "phase-rl",
                    "--policy-artifact",
                    str(artifact_path),
                    "--task-count",
                    "2",
                    "--repeats",
                    "1",
                    "--workers",
                    "1",
                    "--agent-slots",
                    "0",
                    "--planning-delay-ms",
                    "0",
                    "--latency-distribution",
                    "fixed",
                    "--max-attempts",
                    "2",
                    "--records",
                    "4",
                    "--fields",
                    "1",
                    "--requests-per-task",
                    "2",
                    "--candidates",
                    "2",
                    "--read-weight",
                    "0",
                    "--update-weight",
                    "1",
                    "--zipf-theta",
                    "0",
                ],
                stdout=stdout,
            )

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        schema = report["policy_artifact_schema"]
        self.assertTrue(schema["loaded"])
        self.assertTrue(schema["compatible"])
        self.assertEqual(schema["state_schema_version"], 2)
        self.assertIn("class", schema["state_schema_dimensions"])

    def test_artifact_schema_status_flags_legacy_state_without_class(self):
        status = atcc_artifact_schema_status(
            {
                "artifact_type": "phase-aware-atcc-policy-artifact",
                "artifact_version": 1,
                "atcc_state_schema": {
                    "version": 1,
                    "dimensions": ["workload", "task", "phase"],
                },
            }
        )

        self.assertTrue(status["loaded"])
        self.assertFalse(status["compatible"])
        self.assertIn("class", status["missing_expected_dimensions"])

    def test_yield_refresh_regenerate_reports_refresh_events(self):
        workload = build_agent_workload(
            "tpcc",
            "semantic",
            tpcc_config=TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=4,
                order_lines=2,
                candidates_per_task=1,
                transaction_mix=(("new_order", 1.0),),
            ),
        )
        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="tpcc",
            policy_variant="phase-rl",
            task_count=2,
            seed=11,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=2,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
        )

        row = runs[0].to_dict()
        aggregate = aggregate_retry_runs(runs)[0]
        self.assertGreater(row["lease_refresh_regenerations"], 0)
        self.assertEqual(
            aggregate["lease_refresh_regenerations"],
            row["lease_refresh_regenerations"],
        )
        self.assertGreater(aggregate["estimated_refresh_tokens"], 0.0)
        self.assertGreater(
            aggregate["estimated_wasted_tokens_per_task"],
            row["estimated_wasted_tokens_per_task"],
        )

    def test_agent_admission_mode_is_reported(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=4,
                field_count=1,
                requests_per_task=1,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
            ),
        )
        runs = run_retry_matrix(
            workload,
            ("occ",),
            workload_kind="ycsb",
            policy_variant="phase-rl",
            task_count=2,
            seed=5,
            repeats=1,
            workers=1,
            agent_slots=1,
            agent_admission_mode="before-begin",
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
        )

        self.assertEqual(runs[0].to_dict()["agent_admission_mode"], "before-begin")
        self.assertEqual(
            aggregate_retry_runs(runs)[0]["agent_admission_mode"],
            "before-begin",
        )

    def test_lock_scheduler_and_prelock_budget_are_reported(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=4,
                field_count=1,
                requests_per_task=1,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
            ),
        )
        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="phase-rl",
            task_count=2,
            seed=13,
            repeats=1,
            workers=1,
            agent_slots=1,
            agent_admission_mode="before-begin",
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=2,
            tokens_per_operation=10.0,
            object_lock_scheduler="bounded-priority",
            object_lock_priority_burst=3,
            prelock_wait_budget_s=0.007,
            prelock_wait_budget_mode="object",
            prelock_lease_mode="defer-until-after-planning",
        )

        row = runs[0].to_dict()
        aggregate = aggregate_retry_runs(runs)[0]
        self.assertEqual(row["object_lock_scheduler"], "bounded-priority")
        self.assertEqual(row["object_lock_priority_burst"], 3)
        self.assertEqual(row["prelock_wait_budget_s"], 0.007)
        self.assertEqual(row["prelock_wait_budget_mode"], "object")
        self.assertEqual(row["prelock_lease_mode"], "defer-until-after-planning")
        self.assertEqual(aggregate["object_lock_scheduler"], "bounded-priority")
        self.assertEqual(aggregate["object_lock_priority_burst"], 3)
        self.assertEqual(aggregate["prelock_wait_budget_s"], 0.007)
        self.assertEqual(aggregate["prelock_wait_budget_mode"], "object")
        self.assertEqual(aggregate["prelock_lease_mode"], "defer-until-after-planning")


if __name__ == "__main__":
    unittest.main()

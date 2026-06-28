import io
import json
import tempfile
import unittest
from pathlib import Path

import agent.evaluation.atcc_retry_experiment as retry_experiment
from agent.evaluation.atcc_schema import atcc_artifact_schema_status
from agent.evaluation.atcc_retry_experiment import (
    _agent_phase_for_task,
    _background_targets,
    _replay_task_stages,
    _strategies_for_repeat,
    RetryRunSummary,
    aggregate_retry_runs,
    aggregate_selected_baseline_pairs,
    main,
    run_retry_matrix,
)
from agent.runtime import AgentTransactionManager, OperationPolicyTable
from agent.workloads import TPCCConfig, YCSBConfig, build_agent_workload
from agent.workloads.base import (
    AgentCandidate,
    AgentOperation,
    AgentTask,
    populate_task_stage,
    register_workload,
)


class RetryExperimentMetricTests(unittest.TestCase):
    def test_selected_baseline_pairs_compare_hybrid_to_chosen_family(self):
        runs = (
            RetryRunSummary(
                workload="agent-ycsb-semantic",
                strategy="adaptive-hybrid",
                policy_variant="ycsb-strict-tuned",
                seed=10,
                task_count=10,
                workers=1,
                agent_slots=0,
                agent_admission_mode="before-begin",
                max_attempts=1,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                committed_tasks=10,
                final_failed_tasks=0,
                rejected_tasks=0,
                total_attempts=10,
                conflict_aborts=0,
                conflict_object_counts={},
                conflict_object_class_counts={},
                operation_policy_counts={},
                operation_rule_counts={},
                action_counts={"direct": 10},
                prelock_wait_s=0.0,
                elapsed_s=2.0,
                selected_strategy_counts={"tictoc-full": 10},
            ),
            RetryRunSummary(
                workload="agent-ycsb-semantic",
                strategy="tictoc-full",
                policy_variant="ycsb-strict-tuned",
                seed=10,
                task_count=10,
                workers=1,
                agent_slots=0,
                agent_admission_mode="before-begin",
                max_attempts=1,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                committed_tasks=10,
                final_failed_tasks=0,
                rejected_tasks=0,
                total_attempts=10,
                conflict_aborts=0,
                conflict_object_counts={},
                conflict_object_class_counts={},
                operation_policy_counts={},
                operation_rule_counts={},
                action_counts={"direct": 10},
                prelock_wait_s=0.0,
                elapsed_s=2.5,
                selected_strategy_counts={"tictoc-full": 10},
            ),
            RetryRunSummary(
                workload="agent-ycsb-semantic",
                strategy="occ",
                policy_variant="ycsb-strict-tuned",
                seed=10,
                task_count=10,
                workers=1,
                agent_slots=0,
                agent_admission_mode="before-begin",
                max_attempts=1,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                committed_tasks=10,
                final_failed_tasks=0,
                rejected_tasks=0,
                total_attempts=10,
                conflict_aborts=0,
                conflict_object_counts={},
                conflict_object_class_counts={},
                operation_policy_counts={},
                operation_rule_counts={},
                action_counts={"direct": 10},
                prelock_wait_s=0.0,
                elapsed_s=1.0,
                selected_strategy_counts={"occ": 10},
            ),
        )

        report = aggregate_selected_baseline_pairs(runs)

        self.assertEqual(report["paired_runs"], 1)
        self.assertEqual(report["missing_baseline_runs"], 0)
        pair = report["pairs"][0]
        self.assertEqual(pair["seed"], 10)
        self.assertEqual(pair["selected_strategy"], "tictoc-full")
        self.assertEqual(pair["hybrid_tps"], 5.0)
        self.assertEqual(pair["baseline_tps"], 4.0)
        self.assertEqual(pair["hybrid_vs_selected_baseline"], 1.25)

    def test_strategy_order_can_rotate_by_repeat(self):
        strategies = ("occ", "tictoc-full", "adaptive-hybrid")

        self.assertEqual(
            _strategies_for_repeat(strategies, repeat_index=0, strategy_order="rotate"),
            ("occ", "tictoc-full", "adaptive-hybrid"),
        )
        self.assertEqual(
            _strategies_for_repeat(strategies, repeat_index=1, strategy_order="rotate"),
            ("tictoc-full", "adaptive-hybrid", "occ"),
        )
        self.assertEqual(
            _strategies_for_repeat(strategies, repeat_index=2, strategy_order="given"),
            strategies,
        )

    def test_strategy_order_can_pair_selected_baseline(self):
        strategies = ("occ", "mvcc-full", "tictoc-full", "adaptive-hybrid")

        self.assertEqual(
            _strategies_for_repeat(
                strategies,
                repeat_index=0,
                strategy_order="pair-selected-baseline",
                selected_baseline_strategy="mvcc-full",
                hybrid_strategy="adaptive-hybrid",
            ),
            ("occ", "tictoc-full", "mvcc-full", "adaptive-hybrid"),
        )
        self.assertEqual(
            _strategies_for_repeat(
                strategies,
                repeat_index=1,
                strategy_order="pair-selected-baseline",
                selected_baseline_strategy="mvcc-full",
                hybrid_strategy="adaptive-hybrid",
            ),
            ("tictoc-full", "occ", "adaptive-hybrid", "mvcc-full"),
        )

    def test_strategy_order_can_interleave_selected_baseline_blocks(self):
        strategies = ("mvcc-full", "tictoc-full", "adaptive-hybrid")

        self.assertEqual(
            retry_experiment._strategy_execution_blocks(
                strategies,
                repeat_index=0,
                strategy_order="interleave-selected-baseline",
                selected_baseline_strategy="mvcc-full",
                task_count=8,
                interleave_blocks=4,
                hybrid_strategy="adaptive-hybrid",
            ),
            (
                ("tictoc-full", 0, 8),
                ("mvcc-full", 0, 2),
                ("adaptive-hybrid", 0, 2),
                ("adaptive-hybrid", 2, 4),
                ("mvcc-full", 2, 4),
                ("mvcc-full", 4, 6),
                ("adaptive-hybrid", 4, 6),
                ("adaptive-hybrid", 6, 8),
                ("mvcc-full", 6, 8),
            ),
        )

    def test_strategy_order_can_interleave_all_strategy_blocks(self):
        strategies = ("mvcc-full", "tictoc-full", "adaptive-hybrid")

        self.assertEqual(
            retry_experiment._strategy_execution_blocks(
                strategies,
                repeat_index=0,
                strategy_order="interleave-all-strategies",
                selected_baseline_strategy="mvcc-full",
                task_count=6,
                interleave_blocks=3,
                hybrid_strategy="adaptive-hybrid",
            ),
            (
                ("mvcc-full", 0, 2),
                ("tictoc-full", 0, 2),
                ("adaptive-hybrid", 0, 2),
                ("tictoc-full", 2, 4),
                ("adaptive-hybrid", 2, 4),
                ("mvcc-full", 2, 4),
                ("adaptive-hybrid", 4, 6),
                ("mvcc-full", 4, 6),
                ("tictoc-full", 4, 6),
            ),
        )

    def test_selected_baseline_uses_adaptive_read_heavy_window_policy(self):
        tasks = (
            AgentTask(
                task_id="cold-read-only",
                workload="agent-ycsb-semantic",
                task_type="read-update",
                request="cold read-only task",
                context={
                    "hotspot_fraction": 0.0,
                    "hotspot_access_probability": 0.0,
                    "hot_record_count": 0,
                },
                candidates=(
                    AgentCandidate(
                        "candidate",
                        1.0,
                        (
                            AgentOperation.read("ycsb:record:1:field:0"),
                            AgentOperation.read("ycsb:record:1:field:1"),
                            AgentOperation.read("ycsb:record:1:field:2"),
                            AgentOperation.read("ycsb:record:1:field:3"),
                        ),
                    ),
                ),
            ),
            AgentTask(
                task_id="cold-higher-write",
                workload="agent-ycsb-semantic",
                task_type="read-update",
                request="cold higher-write task",
                context={
                    "hotspot_fraction": 0.0,
                    "hotspot_access_probability": 0.0,
                    "hot_record_count": 0,
                },
                candidates=(
                    AgentCandidate(
                        "candidate",
                        1.0,
                        (
                            AgentOperation.overwrite("ycsb:record:2:field:0", "v"),
                            AgentOperation.read("ycsb:record:2:field:1"),
                            AgentOperation.read("ycsb:record:2:field:2"),
                            AgentOperation.read("ycsb:record:2:field:3"),
                        ),
                    ),
                ),
            ),
        )
        policy_artifact = {
            "family_policy_table": {
                "cold_read_heavy_strategy": "adaptive-mvcc-tictoc",
                "adaptive_cold_mvcc_write_ratio_threshold": 0.10,
            }
        }

        selected = retry_experiment._selected_baseline_strategy_for_tasks(
            ("mvcc-full", "tictoc-full", "adaptive-hybrid"),
            tasks,
            workload_kind="ycsb",
            policy_artifact=policy_artifact,
        )

        self.assertEqual(selected, "mvcc-full")

    def test_adaptive_hybrid_reports_selected_strategy_counts(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=32,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=2,
                read_weight=0.90,
                update_weight=0.10,
                zipf_theta=0.7,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.50,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-hybrid",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=3,
            seed=920104,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        aggregate = aggregate_retry_runs(runs)[0]
        self.assertEqual(row["strategy"], "adaptive-hybrid")
        self.assertEqual(row["selected_strategy_counts"], {"tictoc-full": 3})
        self.assertEqual(
            aggregate["selected_strategy_counts"],
            {"tictoc-full": 3},
        )
        self.assertEqual(row["lease_refresh_regenerations"], 0)

    def test_adaptive_hybrid_can_fast_through_selected_operation_atcc(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=64,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=3,
                read_weight=0.50,
                update_weight=0.50,
                zipf_theta=0.99,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.75,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-hybrid", "adaptive-op-strict"),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=3,
            seed=920104,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
            hybrid_fast_through=True,
        )

        hybrid = next(run for run in runs if run.strategy == "adaptive-hybrid")
        self.assertEqual(hybrid.fast_through_strategy, "adaptive-op-strict")
        self.assertEqual(hybrid.selected_strategy_counts, {"adaptive-op-strict": 3})
        self.assertEqual(hybrid.to_dict()["fast_through_strategy"], "adaptive-op-strict")

    def test_adaptive_hybrid_does_not_fast_through_traditional_family(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=32,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=2,
                read_weight=0.90,
                update_weight=0.10,
                zipf_theta=0.7,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.50,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-hybrid", "tictoc-full"),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=3,
            seed=920104,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
            hybrid_fast_through=True,
        )

        hybrid = next(run for run in runs if run.strategy == "adaptive-hybrid")
        self.assertEqual(hybrid.fast_through_strategy, "")
        self.assertEqual(hybrid.selected_strategy_counts, {"tictoc-full": 3})

    def test_adaptive_hybrid_can_fast_through_selected_traditional_family(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=32,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=2,
                read_weight=0.90,
                update_weight=0.10,
                zipf_theta=0.7,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.50,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-hybrid", "tictoc-full"),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=3,
            seed=920104,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
            hybrid_selected_fast_through=True,
        )

        hybrid = next(run for run in runs if run.strategy == "adaptive-hybrid")
        self.assertEqual(hybrid.fast_through_strategy, "tictoc-full")
        self.assertEqual(hybrid.selected_strategy_counts, {"tictoc-full": 3})

    def test_adaptive_hybrid_loads_family_policy_from_artifact(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=32,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=2,
                read_weight=0.90,
                update_weight=0.10,
                zipf_theta=0.7,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.50,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-hybrid",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=3,
            seed=920104,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            policy_artifact={
                "family_policy_table": {
                    "read_heavy_strategy": "mvcc-full",
                    "hot_write_strategy": "adaptive-op-strict",
                    "fallback_strategy": "tictoc-full",
                }
            },
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        self.assertEqual(
            runs[0].to_dict()["selected_strategy_counts"],
            {"mvcc-full": 3},
        )

    def test_family_policy_artifact_can_set_fallback_strategy(self):
        artifact = retry_experiment._family_policy_artifact(
            read_heavy_strategy="tictoc-full",
            hot_write_strategy="adaptive-op-strict",
            fallback_strategy="adaptive-op-strict",
        )

        self.assertEqual(
            artifact["family_policy_table"]["fallback_strategy"],
            "adaptive-op-strict",
        )

    def test_refresh_candidate_write_bases_preserves_full_candidates(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=4,
                field_count=2,
                requests_per_task=2,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
            ),
        )
        task = workload.generate_tasks(1, seed=5)[0]
        manager = AgentTransactionManager(
            operation_policy=OperationPolicyTable.ycsb_strict_tuned_atcc()
        )
        register_workload(manager, workload)
        txn = manager.begin(task.task_id)
        populate_task_stage(txn, task, "commit")
        candidate = txn.candidates[0]
        original_targets = tuple(write.object_id for write in candidate._writes)
        refreshed_target = original_targets[0]
        original_version = candidate._writes[0].base_version

        concurrent = manager.begin("concurrent")
        concurrent.add_candidate("concurrent", quality=1, gen_cost=0).overwrite(
            refreshed_target,
            "concurrent-value",
        )
        self.assertTrue(concurrent.commit().committed)
        manager._snapshot_threadsafe = (  # type: ignore[method-assign]
            lambda: self.fail("refresh should not read a full manager snapshot")
        )

        refreshed = txn.refresh_candidate_write_bases(
            (refreshed_target,),
            reason="test-object-refresh",
            clear_read_set=True,
        )

        self.assertEqual(refreshed, 1)
        self.assertEqual(len(txn.candidates), 1)
        self.assertEqual(
            tuple(write.object_id for write in txn.candidates[0]._writes),
            original_targets,
        )
        self.assertGreater(txn.candidates[0]._writes[0].base_version, original_version)
        self.assertEqual(txn.candidates[0]._writes[0].base_value, "concurrent-value")

    def test_refresh_candidate_write_bases_can_scope_to_best_candidate(self):
        manager = AgentTransactionManager(
            operation_policy=OperationPolicyTable.ycsb_strict_tuned_atcc()
        )
        manager.register_object("hot", "0", kind="row")
        manager.register_object("cold", "0", kind="row")
        txn = manager.begin("agent")
        best = txn.add_candidate("best", quality=10, gen_cost=0)
        best.overwrite("hot", "best-hot")
        best.overwrite("cold", "best-cold")
        backup = txn.add_candidate("backup", quality=1, gen_cost=0)
        backup.overwrite("hot", "backup-hot")
        backup.overwrite("cold", "backup-cold")
        best_versions = tuple(write.base_version for write in best._writes)
        backup_versions = tuple(write.base_version for write in backup._writes)

        concurrent = manager.begin("concurrent")
        concurrent.add_candidate("concurrent", quality=1, gen_cost=0).overwrite(
            "hot",
            "new-hot",
        ).overwrite("cold", "new-cold")
        self.assertTrue(concurrent.commit().committed)

        refreshed = txn.refresh_candidate_write_bases(
            ("hot", "cold"),
            reason="test-best-candidate-refresh",
            clear_read_set=True,
            candidate_scope="best",
        )

        self.assertEqual(refreshed, 2)
        self.assertTrue(
            all(
                write.base_version > original
                for write, original in zip(best._writes, best_versions)
            )
        )
        self.assertEqual(tuple(write.base_version for write in backup._writes), backup_versions)

    def test_pre_snapshot_operation_plan_can_scope_to_best_agent_candidate(self):
        best = AgentCandidate(
            "best",
            10.0,
            tuple(
                AgentOperation.overwrite(f"ycsb:record:0:field:{index}", "best")
                for index in range(6)
            ),
        )
        backup = AgentCandidate(
            "backup",
            1.0,
            tuple(
                AgentOperation.overwrite(f"ycsb:record:1:field:{index}", "backup")
                for index in range(6)
            ),
        )
        manager = AgentTransactionManager(
            operation_policy=OperationPolicyTable.ycsb_strict_tuned_atcc()
        )
        metadata = {
            "workload": "agent-ycsb-semantic",
            "task_type": "read-update",
            "context": {
                "hot_record_count": 2,
                "agent_phase": "commit",
                "operation_candidate_scope": "best",
            },
            "agent_phase": "commit",
        }

        targets, decisions = manager.cc_registry.pre_snapshot_operation_plan(
            "adaptive-op-strict",
            (best, backup),
            metadata=metadata,
        )

        self.assertEqual(
            set(targets),
            {f"ycsb:record:0:field:{index}" for index in range(6)},
        )
        self.assertEqual({decision.candidate_count for decision in decisions}, {1})

    def test_replay_task_stages_can_refresh_only_commit_stage(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=16,
                field_count=4,
                requests_per_task=4,
                candidates_per_task=1,
                read_weight=0.75,
                update_weight=0.25,
                zipf_theta=0.0,
                hotspot_fraction=0.25,
                hotspot_access_probability=1.0,
            ),
        )
        task = next(
            task
            for task in workload.generate_tasks(20, seed=9)
            if any(
                stage["phase"] in {"explore", "refine"} and stage["operations"]
                for stage in task.context["agent_stages"]
            )
            and any(
                stage["phase"] == "commit" and stage["operations"]
                for stage in task.context["agent_stages"]
            )
        )
        manager = AgentTransactionManager()
        register_workload(manager, workload)
        txn = manager.begin(task.task_id)

        _replay_task_stages(txn, task, phases=("commit",))

        self.assertFalse(txn.read_set)
        self.assertTrue(txn.candidates)

    def test_staged_ycsb_refresh_can_rebase_commit_writes_without_replay(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=16,
                field_count=4,
                requests_per_task=4,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.7,
                hotspot_fraction=0.25,
                hotspot_access_probability=1.0,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=41,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        self.assertGreater(row["lease_refresh_regenerations"], 0)
        self.assertGreater(row["lease_refresh_rebased_writes"], 0)
        self.assertEqual(row["lease_refresh_replayed_operations"], 0)

    def test_staged_ycsb_yield_refresh_rebases_after_reacquiring_prelocks(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=16,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.7,
                hotspot_fraction=0.25,
                hotspot_access_probability=1.0,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=42,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        self.assertGreater(row["lease_refresh_regenerations"], 0)
        self.assertGreater(row["lease_refresh_rebased_writes"], 0)
        self.assertEqual(row["lease_refresh_replayed_operations"], 0)

    def test_staged_ycsb_object_refresh_rebases_only_best_candidate(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=16,
                field_count=4,
                requests_per_task=4,
                candidates_per_task=4,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.7,
                hotspot_fraction=0.25,
                hotspot_access_probability=1.0,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=43,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        self.assertGreater(row["lease_refresh_rebased_writes"], 0)
        self.assertLessEqual(row["lease_refresh_rebased_writes"], 8)
        self.assertEqual(row["lease_refresh_replayed_operations"], 0)

    def test_ycsb_object_refresh_targets_all_commit_writes_with_prelocks(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=8,
                field_count=4,
                requests_per_task=4,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
                hotspot_fraction=0.25,
                hotspot_access_probability=1.0,
            ),
        )
        task = workload.generate_tasks(1, seed=7)[0]
        commit_writes = {
            operation.object_id
            for operation in retry_experiment.stage_operations(task, "commit")
            if operation.kind != "read"
        }
        self.assertGreater(len(commit_writes), 1)
        txn = type(
            "Txn",
            (),
            {"prelocked_targets": (next(iter(commit_writes)),)},
        )()

        targets = set(retry_experiment._object_refresh_commit_targets(txn, task))

        self.assertEqual(commit_writes, targets)

    def test_retry_matrix_reports_agent_latency_and_cost_metrics(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=4,
                field_count=2,
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

    def test_retry_aggregate_uses_precise_refresh_replay_operations(self):
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
                task_operation_counts=(10, 10),
                tokens_per_operation=10.0,
                estimated_tokens=200.0,
                estimated_wasted_tokens=0.0,
                lease_refresh_regenerations=2,
                lease_refresh_replayed_operations=3,
            ),
        )

        aggregate = aggregate_retry_runs(runs)[0]

        self.assertEqual(aggregate["lease_refresh_replayed_operations"], 3)
        self.assertEqual(aggregate["estimated_refresh_tokens"], 30.0)
        self.assertEqual(aggregate["estimated_wasted_tokens_per_task"], 15.0)

    def test_retry_aggregate_charges_no_replay_tokens_for_object_refresh(self):
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
                task_operation_counts=(10, 10),
                tokens_per_operation=10.0,
                estimated_tokens=200.0,
                estimated_wasted_tokens=0.0,
                lease_refresh_regenerations=2,
                lease_refresh_replayed_operations=0,
                lease_refresh_rebased_writes=6,
            ),
        )

        aggregate = aggregate_retry_runs(runs)[0]

        self.assertEqual(aggregate["lease_refresh_rebased_writes"], 6)
        self.assertEqual(aggregate["estimated_refresh_tokens"], 0.0)
        self.assertEqual(aggregate["estimated_wasted_tokens_per_task"], 0.0)

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

    def test_retry_cli_reports_selected_baseline_pairs(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "adaptive-hybrid,tictoc-full",
                "--policy-variant",
                "ycsb-strict-tuned",
                "--task-count",
                "1",
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
                "1",
                "--records",
                "16",
                "--fields",
                "4",
                "--requests-per-task",
                "2",
                "--candidates",
                "1",
                "--read-weight",
                "1.0",
                "--update-weight",
                "0.0",
                "--zipf-theta",
                "0.0",
                "--hotspot-fraction",
                "0.10",
                "--hotspot-access-probability",
                "0.50",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        pairs = report["selected_baseline_pairs"]
        self.assertEqual(pairs["paired_runs"], 1)
        self.assertEqual(pairs["pairs"][0]["selected_strategy"], "tictoc-full")

    def test_retry_cli_reports_abort_retry_delay(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "occ",
                "--task-count",
                "1",
                "--repeats",
                "1",
                "--workers",
                "1",
                "--agent-slots",
                "0",
                "--planning-delay-ms",
                "0",
                "--abort-retry-delay-ms",
                "250",
                "--latency-distribution",
                "fixed",
                "--max-attempts",
                "1",
                "--records",
                "4",
                "--fields",
                "1",
                "--requests-per-task",
                "1",
                "--candidates",
                "1",
                "--read-weight",
                "1.0",
                "--update-weight",
                "0.0",
                "--zipf-theta",
                "0.0",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["abort_retry_delay_s"], 0.25)
        self.assertEqual(report["aggregates"][0]["abort_retry_delay_s"], 0.25)

    def test_retry_cli_loads_profile_eval_overrides_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            override_path = Path(tmp) / "profile-eval-overrides.json"
            override_path.write_text(
                json.dumps(
                    {
                        "ycsb-high": {
                            "planning_delay_ms": 100,
                            "abort_retry_delay_ms": 500,
                            "object_lock_scheduler": "bounded-priority",
                            "prelock_lease_mode": "yield-refresh-regenerate",
                            "prelock_wait_budget_mode": "object",
                            "prelock_wait_budget_ms": 70,
                            "workload_config": {"candidates_per_task": 5},
                        }
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            exit_code = main(
                [
                    "--workload",
                    "ycsb",
                    "--profile-name",
                    "ycsb-high",
                    "--strategies",
                    "occ",
                    "--task-count",
                    "1",
                    "--repeats",
                    "1",
                    "--workers",
                    "1",
                    "--agent-slots",
                    "0",
                    "--planning-delay-ms",
                    "0",
                    "--abort-retry-delay-ms",
                    "0",
                    "--latency-distribution",
                    "fixed",
                    "--max-attempts",
                    "1",
                    "--records",
                    "8",
                    "--fields",
                    "1",
                    "--requests-per-task",
                    "1",
                    "--candidates",
                    "1",
                    "--read-weight",
                    "1.0",
                    "--update-weight",
                    "0.0",
                    "--zipf-theta",
                    "0.0",
                    "--profile-eval-overrides-file",
                    str(override_path),
                ],
                stdout=stdout,
            )

            report = json.loads(stdout.getvalue())

            self.assertEqual(exit_code, 0)
            self.assertEqual(report["profile_name"], "ycsb-high")
            self.assertEqual(report["planning_delay_s"], 0.1)
            self.assertEqual(report["abort_retry_delay_s"], 0.5)
            self.assertEqual(report["object_lock_scheduler"], "bounded-priority")
            self.assertEqual(report["prelock_lease_mode"], "yield-refresh-regenerate")
            self.assertEqual(report["prelock_wait_budget_mode"], "object")
            self.assertEqual(report["prelock_wait_budget_s"], 0.07)
            self.assertEqual(report["workload_config"]["candidates_per_task"], 5)
            self.assertEqual(
                report["profile_eval_overrides"],
                {
                    "planning_delay_s": 0.1,
                    "abort_retry_delay_s": 0.5,
                    "object_lock_scheduler": "bounded-priority",
                    "prelock_lease_mode": "yield-refresh-regenerate",
                    "prelock_wait_budget_mode": "object",
                    "prelock_wait_budget_s": 0.07,
                    "workload_config": {"candidates_per_task": 5},
                },
            )

    def test_retry_cli_accepts_rotating_strategy_order(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "occ,adaptive-hybrid",
                "--strategy-order",
                "rotate",
                "--task-count",
                "1",
                "--repeats",
                "2",
                "--workers",
                "1",
                "--agent-slots",
                "0",
                "--planning-delay-ms",
                "0",
                "--latency-distribution",
                "fixed",
                "--max-attempts",
                "1",
                "--records",
                "4",
                "--fields",
                "1",
                "--requests-per-task",
                "1",
                "--candidates",
                "1",
                "--read-weight",
                "1.0",
                "--update-weight",
                "0.0",
                "--zipf-theta",
                "0.0",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["strategy_order"], "rotate")

    def test_retry_cli_accepts_pair_selected_baseline_order(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "adaptive-hybrid,mvcc-full,tictoc-full",
                "--strategy-order",
                "pair-selected-baseline",
                "--policy-variant",
                "ycsb-strict-tuned",
                "--task-count",
                "1",
                "--repeats",
                "2",
                "--workers",
                "1",
                "--agent-slots",
                "0",
                "--planning-delay-ms",
                "0",
                "--latency-distribution",
                "fixed",
                "--max-attempts",
                "1",
                "--records",
                "16",
                "--fields",
                "4",
                "--requests-per-task",
                "2",
                "--candidates",
                "1",
                "--read-weight",
                "1.0",
                "--update-weight",
                "0.0",
                "--zipf-theta",
                "0.0",
                "--hotspot-fraction",
                "0.0",
                "--hotspot-access-probability",
                "0.0",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["strategy_order"], "pair-selected-baseline")
        self.assertEqual(report["selected_baseline_pairs"]["paired_runs"], 2)

    def test_retry_cli_accepts_interleave_selected_baseline_order(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "adaptive-hybrid,mvcc-full,tictoc-full",
                "--strategy-order",
                "interleave-selected-baseline",
                "--interleave-blocks",
                "2",
                "--policy-variant",
                "ycsb-strict-tuned",
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
                "1",
                "--records",
                "16",
                "--fields",
                "4",
                "--requests-per-task",
                "2",
                "--candidates",
                "1",
                "--read-weight",
                "1.0",
                "--update-weight",
                "0.0",
                "--zipf-theta",
                "0.0",
                "--hotspot-fraction",
                "0.0",
                "--hotspot-access-probability",
                "0.0",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["strategy_order"], "interleave-selected-baseline")
        self.assertEqual(report["interleave_blocks"], 2)
        self.assertEqual(report["selected_baseline_pairs"]["paired_runs"], 1)

    def test_retry_cli_accepts_interleave_all_strategies_order(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "adaptive-hybrid,mvcc-full,tictoc-full",
                "--strategy-order",
                "interleave-all-strategies",
                "--interleave-blocks",
                "2",
                "--policy-variant",
                "ycsb-strict-tuned",
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
                "1",
                "--records",
                "16",
                "--fields",
                "4",
                "--requests-per-task",
                "2",
                "--candidates",
                "1",
                "--read-weight",
                "1.0",
                "--update-weight",
                "0.0",
                "--zipf-theta",
                "0.0",
                "--hotspot-fraction",
                "0.0",
                "--hotspot-access-probability",
                "0.0",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["strategy_order"], "interleave-all-strategies")
        self.assertEqual(report["interleave_blocks"], 2)
        self.assertEqual(report["selected_baseline_pairs"]["paired_runs"], 1)

    def test_retry_cli_accepts_hybrid_fast_through(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "adaptive-hybrid,adaptive-op-strict",
                "--hybrid-fast-through",
                "--policy-variant",
                "ycsb-strict-tuned",
                "--task-count",
                "1",
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
                "1",
                "--records",
                "16",
                "--fields",
                "4",
                "--requests-per-task",
                "2",
                "--candidates",
                "1",
                "--read-weight",
                "0.0",
                "--update-weight",
                "1.0",
                "--zipf-theta",
                "0.0",
                "--hotspot-fraction",
                "0.10",
                "--hotspot-access-probability",
                "0.75",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())
        hybrid = next(
            row for row in report["aggregates"] if row["strategy"] == "adaptive-hybrid"
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["hybrid_fast_through"])
        self.assertEqual(
            hybrid["fast_through_strategy_counts"],
            {"adaptive-op-strict": 1},
        )

    def test_retry_cli_accepts_hybrid_selected_fast_through(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "adaptive-hybrid,tictoc-full",
                "--hybrid-selected-fast-through",
                "--policy-variant",
                "ycsb-strict-tuned",
                "--task-count",
                "1",
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
                "1",
                "--records",
                "16",
                "--fields",
                "4",
                "--requests-per-task",
                "2",
                "--candidates",
                "1",
                "--read-weight",
                "1.0",
                "--update-weight",
                "0.0",
                "--zipf-theta",
                "0.7",
                "--hotspot-fraction",
                "0.10",
                "--hotspot-access-probability",
                "0.50",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())
        hybrid = next(
            row for row in report["aggregates"] if row["strategy"] == "adaptive-hybrid"
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["hybrid_selected_fast_through"])
        self.assertEqual(hybrid["fast_through_strategy_counts"], {"tictoc-full": 1})

    def test_abort_retry_delay_increases_retry_latency(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=1,
                field_count=1,
                requests_per_task=1,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
            ),
        )

        without_delay = run_retry_matrix(
            workload,
            ("occ",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=1,
            repeats=1,
            workers=2,
            agent_slots=0,
            planning_delay_s=0.010,
            abort_retry_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=2,
        )[0]
        with_delay = run_retry_matrix(
            workload,
            ("occ",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=1,
            repeats=1,
            workers=2,
            agent_slots=0,
            planning_delay_s=0.010,
            abort_retry_delay_s=0.050,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=2,
        )[0]

        self.assertEqual(without_delay.conflict_aborts, 1)
        self.assertEqual(with_delay.conflict_aborts, 1)
        self.assertGreater(with_delay.elapsed_s - without_delay.elapsed_s, 0.030)

    def test_family_search_cli_writes_best_family_policy_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_output = Path(tmp) / "best-family-policy.json"
            stdout = io.StringIO()

            exit_code = main(
                [
                    "--mode",
                    "family-search",
                    "--workload",
                    "ycsb",
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
                    "1",
                    "--records",
                    "8",
                    "--fields",
                    "2",
                    "--requests-per-task",
                    "2",
                    "--candidates",
                    "1",
                    "--read-weight",
                    "0.90",
                    "--update-weight",
                    "0.10",
                    "--zipf-theta",
                    "0.7",
                    "--hotspot-fraction",
                    "0.10",
                    "--hotspot-access-probability",
                    "0.50",
                    "--family-search-read-heavy-strategies",
                    "mvcc-full,tictoc-full",
                    "--family-policy-output",
                    str(artifact_output),
                ],
                stdout=stdout,
            )

            report = json.loads(stdout.getvalue())
            artifact = json.loads(artifact_output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["mode"], "family-search")
        self.assertEqual(
            report["family_search"]["training_method"],
            "offline-family-policy-search",
        )
        self.assertIn(
            report["family_search"]["best_read_heavy_strategy"],
            {"mvcc-full", "tictoc-full"},
        )
        self.assertEqual(artifact["artifact_type"], "atcc-family-policy-artifact")
        self.assertIn("family_policy_table", artifact)
        self.assertEqual(
            artifact["family_policy_table"]["read_heavy_strategy"],
            report["family_search"]["best_read_heavy_strategy"],
        )

    def test_family_search_cli_can_optimize_multiple_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_output = Path(tmp) / "joint-family-policy.json"
            stdout = io.StringIO()

            exit_code = main(
                [
                    "--mode",
                    "family-search",
                    "--workload",
                    "ycsb",
                    "--family-search-profiles",
                    "ycsb-medium,ycsb-high",
                    "--task-count",
                    "1",
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
                    "1",
                    "--family-search-read-heavy-strategies",
                    "mvcc-full,tictoc-full",
                    "--family-policy-output",
                    str(artifact_output),
                ],
                stdout=stdout,
            )

            report = json.loads(stdout.getvalue())
            artifact = json.loads(artifact_output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["mode"], "family-search")
        self.assertEqual(report["family_search"]["profile_count"], 2)
        self.assertEqual(
            {row["profile"] for row in report["family_search"]["profile_results"]},
            {"ycsb-medium", "ycsb-high"},
        )
        for candidate in report["family_search"]["candidates"]:
            self.assertEqual(
                {row["profile"] for row in candidate["profile_aggregates"]},
                {"ycsb-medium", "ycsb-high"},
            )
        self.assertEqual(
            artifact["family_policy_table"]["read_heavy_strategy"],
            report["family_search"]["best_read_heavy_strategy"],
        )

    def test_family_search_cli_can_score_against_profile_baselines(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--mode",
                "family-search",
                "--workload",
                "ycsb",
                "--family-search-profiles",
                "ycsb-low,ycsb-medium",
                "--task-count",
                "1",
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
                "1",
                "--family-search-read-heavy-strategies",
                "mvcc-full",
                "--family-search-cold-read-heavy-strategies",
                "occ",
                "--family-search-baseline-strategies",
                "occ,mvcc-full",
                "--family-search-score-mode",
                "baseline-relative",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            report["family_search"]["selection_metric"],
            "sum(profile_score / best_baseline_score) across profiles",
        )
        candidate = report["family_search"]["candidates"][0]
        for row in candidate["profile_aggregates"]:
            self.assertIn("baseline", row)
            self.assertIn(row["baseline"]["strategy"], {"occ", "mvcc-full"})
            self.assertGreater(row["baseline"]["score"], 0.0)
            self.assertGreater(row["relative_score"], 0.0)

    def test_family_search_cli_can_use_balanced_baseline_score(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--mode",
                "family-search",
                "--workload",
                "ycsb",
                "--family-search-profiles",
                "ycsb-low",
                "--task-count",
                "1",
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
                "1",
                "--family-search-read-heavy-strategies",
                "mvcc-full",
                "--family-search-cold-read-heavy-strategies",
                "occ",
                "--family-search-baseline-strategies",
                "occ,mvcc-full",
                "--family-search-score-mode",
                "baseline-balanced",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["family_search"]["score_mode"], "baseline-balanced")
        self.assertEqual(
            report["family_search"]["selection_metric"],
            "sum(relative_score - 2.0 * max(0, 1 - relative_score)) across profiles",
        )

    def test_family_search_cli_can_optimize_hot_write_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_output = Path(tmp) / "hot-write-family-policy.json"
            stdout = io.StringIO()

            exit_code = main(
                [
                    "--mode",
                    "family-search",
                    "--workload",
                    "ycsb",
                    "--family-search-profiles",
                    "ycsb-high",
                    "--task-count",
                    "1",
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
                    "1",
                    "--family-search-read-heavy-strategies",
                    "mvcc-full",
                    "--family-search-cold-read-heavy-strategies",
                    "occ",
                    "--family-search-hot-write-strategies",
                    "adaptive-op-strict,mvcc-full",
                    "--family-policy-output",
                    str(artifact_output),
                ],
                stdout=stdout,
            )

            report = json.loads(stdout.getvalue())
            artifact = json.loads(artifact_output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn(
            report["family_search"]["best_hot_write_strategy"],
            {"adaptive-op-strict", "mvcc-full"},
        )
        candidate = report["family_search"]["candidates"][0]
        self.assertIn("hot_write_strategy", candidate)
        self.assertEqual(
            artifact["family_policy_table"]["hot_write_strategy"],
            report["family_search"]["best_hot_write_strategy"],
        )

    def test_family_search_cli_can_optimize_hot_write_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_output = Path(tmp) / "threshold-family-policy.json"
            stdout = io.StringIO()

            exit_code = main(
                [
                    "--mode",
                    "family-search",
                    "--workload",
                    "ycsb",
                    "--family-search-profiles",
                    "ycsb-medium,ycsb-high",
                    "--task-count",
                    "1",
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
                    "1",
                    "--family-search-read-heavy-strategies",
                    "mvcc-full",
                    "--family-search-cold-read-heavy-strategies",
                    "occ",
                    "--family-search-hot-write-strategies",
                    "adaptive-op-strict",
                    "--family-search-hot-write-ratio-thresholds",
                    "0.25,0.55",
                    "--family-search-hotspot-probability-thresholds",
                    "0.50,0.80",
                    "--family-policy-output",
                    str(artifact_output),
                ],
                stdout=stdout,
            )

            report = json.loads(stdout.getvalue())
            artifact = json.loads(artifact_output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        search = report["family_search"]
        self.assertIn(search["best_hot_write_ratio_threshold"], {0.25, 0.55})
        self.assertIn(search["best_hotspot_probability_threshold"], {0.50, 0.80})
        candidate = search["candidates"][0]
        self.assertIn("hot_write_ratio_threshold", candidate)
        self.assertIn("hotspot_probability_threshold", candidate)
        self.assertEqual(
            artifact["family_policy_table"]["hot_write_ratio_threshold"],
            search["best_hot_write_ratio_threshold"],
        )
        self.assertEqual(
            artifact["family_policy_table"]["hotspot_probability_threshold"],
            search["best_hotspot_probability_threshold"],
        )

    def test_family_search_cli_can_optimize_runtime_commit_parameters(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--mode",
                "family-search",
                "--workload",
                "ycsb",
                "--family-search-profiles",
                "ycsb-high",
                "--task-count",
                "1",
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
                "1",
                "--family-search-read-heavy-strategies",
                "mvcc-full",
                "--family-search-cold-read-heavy-strategies",
                "occ",
                "--family-search-hot-write-strategies",
                "adaptive-op-strict",
                "--family-search-prelock-wait-budget-ms-values",
                "0,70",
                "--family-search-prelock-lease-modes",
                "hold,yield-refresh-regenerate",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        search = report["family_search"]
        self.assertIn(search["best_prelock_wait_budget_s"], {0.0, 0.07})
        self.assertIn(
            search["best_prelock_lease_mode"],
            {"hold", "yield-refresh-regenerate"},
        )
        candidate = search["candidates"][0]
        self.assertIn("prelock_wait_budget_s", candidate)
        self.assertIn("prelock_lease_mode", candidate)

    def test_family_search_cli_can_optimize_execution_runtime_parameters(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--mode",
                "family-search",
                "--workload",
                "ycsb",
                "--family-search-profiles",
                "ycsb-high",
                "--task-count",
                "1",
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
                "1",
                "--family-search-read-heavy-strategies",
                "mvcc-full",
                "--family-search-cold-read-heavy-strategies",
                "occ",
                "--family-search-hot-write-strategies",
                "adaptive-op-strict",
                "--family-search-agent-execution-modes",
                "staged,staged-local",
                "--family-search-snapshot-timings",
                "before-planning,after-planning",
                "--family-search-object-lock-schedulers",
                "bounded-priority,priority",
            ],
            stdout=stdout,
        )

        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        search = report["family_search"]
        self.assertIn(
            search["best_agent_execution_mode"],
            {"staged", "staged-local"},
        )
        self.assertIn(
            search["best_snapshot_timing"],
            {"before-planning", "after-planning"},
        )
        self.assertIn(
            search["best_object_lock_scheduler"],
            {"bounded-priority", "priority"},
        )
        candidate = search["candidates"][0]
        self.assertIn("agent_execution_mode", candidate)
        self.assertIn("snapshot_timing", candidate)
        self.assertIn("object_lock_scheduler", candidate)

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

    def test_staged_execution_reports_phase_counts_and_commit_atcc_rules(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=8,
                field_count=1,
                requests_per_task=2,
                candidates_per_task=2,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
                hotspot_fraction=0.25,
                hotspot_access_probability=1.0,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=3,
            seed=21,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=2,
            tokens_per_operation=10.0,
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        aggregate = aggregate_retry_runs(runs)[0]
        self.assertEqual(row["agent_execution_mode"], "staged")
        self.assertEqual(row["snapshot_timing"], "before-planning")
        self.assertEqual(row["stage_phase_counts"]["commit"], 3)
        self.assertEqual(aggregate["stage_phase_counts"]["commit"], 3)
        self.assertTrue(
            any(
                rule.startswith("phase-atcc-commit")
                for rule in row["operation_rule_counts"]
            )
        )

    def test_staged_local_atcc_does_not_classify_read_only_tasks_as_commit(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=8,
                field_count=2,
                requests_per_task=2,
                candidates_per_task=1,
                read_weight=1.0,
                update_weight=0.0,
                zipf_theta=0.0,
                hotspot_fraction=0.0,
                hotspot_access_probability=0.0,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=41,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            agent_execution_mode="staged-local",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        self.assertEqual(row["stage_phase_counts"].get("commit", 0), 0)
        self.assertTrue(row["operation_rule_counts"])
        self.assertFalse(
            any(
                rule.startswith("phase-atcc-commit")
                for rule in row["operation_rule_counts"]
            )
        )

    def test_staged_yield_refresh_regenerate_replays_before_commit(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=4,
                field_count=6,
                requests_per_task=6,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
                hotspot_fraction=0.25,
                hotspot_access_probability=1.0,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=23,
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
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        self.assertGreater(row["lease_refresh_regenerations"], 0)
        self.assertEqual(row["committed_tasks"], 2)
        self.assertEqual(row["stage_phase_counts"]["commit"], 2)

    def test_staged_yield_refresh_skips_refresh_without_prelocks(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=8,
                field_count=2,
                requests_per_task=2,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
                hotspot_fraction=0.0,
                hotspot_access_probability=0.0,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=29,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        self.assertEqual(row["operation_policy_counts"], {"optimistic": 4})
        self.assertEqual(row["lease_refresh_regenerations"], 0)

    def test_staged_yield_refresh_refreshes_hotspot_agent_without_prelocks(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=8,
                field_count=2,
                requests_per_task=2,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.7,
                hotspot_fraction=0.25,
                hotspot_access_probability=0.5,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=31,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        self.assertEqual(row["operation_policy_counts"], {"optimistic": 4})
        self.assertGreater(row["lease_refresh_regenerations"], 0)

    def test_staged_yield_refresh_does_not_refresh_after_commit_populate_aborts(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=8,
                field_count=2,
                requests_per_task=2,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.7,
                hotspot_fraction=0.25,
                hotspot_access_probability=0.5,
            ),
        )
        original_populate = retry_experiment.populate_task_stage

        def abort_after_commit_populate(txn, task, phase):
            result = original_populate(txn, task, phase)
            if phase == "commit":
                txn.abort("test-populate-abort")
            return result

        retry_experiment.populate_task_stage = abort_after_commit_populate
        try:
            runs = run_retry_matrix(
                workload,
                ("adaptive-op-strict",),
                workload_kind="ycsb",
                policy_variant="ycsb-strict-tuned",
                task_count=1,
                seed=31,
                repeats=1,
                workers=1,
                agent_slots=0,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                latency_cv=0.8,
                latency_max_s=0.0,
                max_attempts=1,
                tokens_per_operation=10.0,
                prelock_lease_mode="yield-refresh-regenerate",
                agent_execution_mode="staged",
                snapshot_timing="before-planning",
            )
        finally:
            retry_experiment.populate_task_stage = original_populate

        row = runs[0].to_dict()
        self.assertEqual(row["final_failed_tasks"], 1)
        self.assertEqual(row["committed_tasks"], 0)

    def test_staged_yield_refresh_keeps_read_heavy_medium_refresh_to_avoid_retries(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=16,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=1,
                read_weight=0.90,
                update_weight=0.10,
                zipf_theta=0.7,
                hotspot_fraction=0.25,
                hotspot_access_probability=0.5,
            ),
        )

        runs = run_retry_matrix(
            workload,
            ("adaptive-op-strict",),
            workload_kind="ycsb",
            policy_variant="ycsb-strict-tuned",
            task_count=2,
            seed=22,
            repeats=1,
            workers=1,
            agent_slots=0,
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=1,
            tokens_per_operation=10.0,
            prelock_lease_mode="yield-refresh-regenerate",
            agent_execution_mode="staged",
            snapshot_timing="before-planning",
        )

        row = runs[0].to_dict()
        self.assertGreater(row["lease_refresh_regenerations"], 0)

    def test_ycsb_background_targets_cover_configured_hotspot_set(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=10,
                field_count=2,
                requests_per_task=2,
                candidates_per_task=1,
                read_weight=0.5,
                update_weight=0.5,
                zipf_theta=0.7,
                hotspot_fraction=0.2,
                hotspot_access_probability=0.5,
            ),
        )

        targets = _background_targets(workload, "ycsb")

        self.assertEqual(
            set(targets),
            {
                "ycsb:record:0:field:0",
                "ycsb:record:0:field:1",
                "ycsb:record:1:field:0",
                "ycsb:record:1:field:1",
            },
        )

    def test_retry_cli_accepts_paper_aligned_ycsb_hotspot_and_staged_flags(self):
        stdout = io.StringIO()

        exit_code = main(
            [
                "--workload",
                "ycsb",
                "--strategies",
                "occ",
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
                "1",
                "--records",
                "10",
                "--fields",
                "1",
                "--requests-per-task",
                "2",
                "--candidates",
                "1",
                "--read-weight",
                "0.5",
                "--update-weight",
                "0.5",
                "--zipf-theta",
                "0",
                "--hotspot-fraction",
                "0.1",
                "--hotspot-access-probability",
                "0.75",
                "--agent-execution-mode",
                "staged",
                "--snapshot-timing",
                "after-planning",
            ],
            stdout=stdout,
        )

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["agent_execution_mode"], "staged")
        self.assertEqual(report["snapshot_timing"], "after-planning")
        self.assertEqual(report["workload_config"]["hotspot_fraction"], 0.1)
        self.assertEqual(report["workload_config"]["hotspot_access_probability"], 0.75)


if __name__ == "__main__":
    unittest.main()

import io
import json
import tempfile
import unittest
from pathlib import Path

from agent.evaluation import atcc_reward_cost_search as reward_cost_search
from agent.evaluation.atcc_reward_cost_search import main, run_reward_cost_search
from agent.workloads import YCSBConfig, build_agent_workload


class ATCCRewardCostSearchTests(unittest.TestCase):
    def test_candidate_ranking_prefers_balanced_system_score(self):
        fast_but_starving = {
            "label": "fast-but-starving",
            "atcc_commit_rate": 1.0,
            "atcc_throughput": 20.0,
            "atcc_p99_latency_s": 2.0,
            "atcc_wasted_tokens_per_task": 500.0,
            "atcc_prelock_wait_per_task_s": 0.30,
            "atcc_background_throughput": 1.0,
            "atcc_pessimistic_decisions": 120,
            "two_pl_p99_latency_s": 1.0,
            "atcc_vs_occ_throughput_x": 200.0,
            "atcc_vs_occ_waste_reduction_pct": -100.0,
            "atcc_vs_2pl_throughput_x": 2.0,
            "atcc_vs_2pl_background_throughput_x": 0.10,
            "atcc_vs_2pl_pessimistic_decision_delta": 80,
        }
        balanced = {
            "label": "balanced",
            "atcc_commit_rate": 1.0,
            "atcc_throughput": 15.0,
            "atcc_p99_latency_s": 0.75,
            "atcc_wasted_tokens_per_task": 50.0,
            "atcc_prelock_wait_per_task_s": 0.01,
            "atcc_background_throughput": 8.0,
            "atcc_pessimistic_decisions": 40,
            "two_pl_p99_latency_s": 1.0,
            "atcc_vs_occ_throughput_x": 150.0,
            "atcc_vs_occ_waste_reduction_pct": 80.0,
            "atcc_vs_2pl_throughput_x": 1.5,
            "atcc_vs_2pl_background_throughput_x": 0.80,
            "atcc_vs_2pl_pessimistic_decision_delta": -10,
        }

        ranked = reward_cost_search._rank_candidates(
            [fast_but_starving, balanced]
        )

        self.assertEqual(ranked[0]["label"], "balanced")
        self.assertIn("multi_objective_score", ranked[0])
        self.assertIn("ranking_score_components", ranked[0])
        self.assertGreater(
            ranked[0]["multi_objective_score"],
            ranked[1]["multi_objective_score"],
        )

    def test_lease_mode_semantics_flags_non_comparable_windows(self):
        self.assertEqual(
            reward_cost_search._lease_mode_semantics("hold"),
            "pre-planning-snapshot-held-locks",
        )
        self.assertEqual(
            reward_cost_search._lease_mode_semantics("yield-during-planning"),
            "pre-planning-snapshot-yielded-locks",
        )
        self.assertEqual(
            reward_cost_search._lease_mode_semantics("yield-refresh-regenerate"),
            "refresh-regenerate-after-planning",
        )
        self.assertEqual(
            reward_cost_search._lease_mode_semantics("defer-until-after-planning"),
            "post-planning-snapshot",
        )
        self.assertTrue(
            reward_cost_search._is_long_transaction_window_comparable("hold")
        )
        self.assertTrue(
            reward_cost_search._is_long_transaction_window_comparable(
                "yield-during-planning"
            )
        )
        self.assertFalse(
            reward_cost_search._is_long_transaction_window_comparable(
                "yield-refresh-regenerate"
            )
        )
        self.assertFalse(
            reward_cost_search._is_long_transaction_window_comparable(
                "defer-until-after-planning"
            )
        )

    def test_reward_cost_search_writes_artifacts_and_rankings(self):
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
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = run_reward_cost_search(
                workload,
                workload_kind="ycsb",
                workload_config={"record_count": 4},
                output_dir=output_dir,
                lock_wait_costs=(50.0, 100.0),
                lock_action_costs=(0.02,),
                lock_queue_depth_costs=(0.0, 0.11),
                lock_handoff_costs=(0.13,),
                committing_count_costs=(0.17,),
                train_episodes=1,
                train_task_count=2,
                eval_task_count=2,
                eval_repeats=1,
                seed=23,
                workers=1,
                agent_slots=0,
                agent_admission_mode="before-begin",
                planning_delay_s=0.0,
                latency_distribution="fixed",
                latency_cv=0.8,
                latency_max_s=0.0,
                max_attempts=2,
                tokens_per_operation=10.0,
                background_workers=0,
                background_interval_s=0.0,
                object_lock_scheduler="bounded-priority",
                object_lock_priority_burst=3,
                prelock_wait_budget_s=0.007,
                prelock_wait_budget_mode="object",
                prelock_lease_mode="defer-until-after-planning",
            )

            self.assertEqual(
                report["artifact_type"],
                "phase-aware-atcc-reward-cost-search",
            )
            self.assertEqual(len(report["candidates"]), 4)
            self.assertEqual(len(report["ranked_candidates"]), 4)
            self.assertEqual(report["config"]["lock_queue_depth_costs"], [0.0, 0.11])
            self.assertEqual(report["config"]["lock_handoff_costs"], [0.13])
            self.assertEqual(report["config"]["committing_count_costs"], [0.17])
            self.assertEqual(report["config"]["agent_admission_mode"], "before-begin")
            self.assertEqual(report["config"]["object_lock_scheduler"], "bounded-priority")
            self.assertEqual(report["config"]["object_lock_priority_burst"], 3)
            self.assertEqual(report["config"]["prelock_wait_budget_s"], 0.007)
            self.assertEqual(report["config"]["prelock_wait_budget_mode"], "object")
            self.assertEqual(report["config"]["prelock_lease_mode"], "defer-until-after-planning")
            self.assertEqual(report["ranking_metric"], "multi_objective_score")
            self.assertIn("background_throughput_vs_2pl", report["ranking_weights"])
            self.assertTrue(report["best_candidate"])
            self.assertIn("multi_objective_score", report["best_candidate"])
            self.assertTrue((output_dir / "atcc_reward_cost_search.json").exists())
            for row in report["candidates"]:
                self.assertTrue(Path(row["policy_artifact"]).exists())
                self.assertTrue(Path(row["evaluation"]).exists())
                self.assertIn("atcc_throughput", row)
                self.assertIn("atcc_total_throughput", row)
                self.assertIn("atcc_vs_occ_throughput_x", row)
                self.assertIn("atcc_background_throughput", row)
                self.assertIn("atcc_lease_refresh_regenerations", row)
                self.assertIn("atcc_estimated_refresh_tokens_per_task", row)
                self.assertIn("atcc_vs_2pl_total_throughput_x", row)
                self.assertIn("two_pl_background_throughput", row)
                self.assertIn("atcc_vs_2pl_background_throughput_x", row)
                self.assertIn("atcc_vs_2pl_tail_latency_x", row)
                self.assertIn("multi_objective_score", row)
                self.assertIn("ranking_score_components", row)
                self.assertIn("lock_queue_depth_cost", row)
                self.assertIn("lock_handoff_cost", row)
                self.assertIn("committing_count_cost", row)
                self.assertIn("prelock_lease_semantics", row)
                self.assertIn("atcc_long_transaction_window_comparable", row)
                artifact = json.loads(Path(row["policy_artifact"]).read_text(encoding="utf-8"))
                self.assertEqual(
                    artifact["training_config"]["atcc_lock_queue_depth_cost"],
                    row["lock_queue_depth_cost"],
                )
                self.assertEqual(
                    artifact["training_config"]["atcc_lock_handoff_cost"],
                    row["lock_handoff_cost"],
                )
                self.assertEqual(
                    artifact["training_config"]["atcc_committing_count_cost"],
                    row["committing_count_cost"],
                )
                self.assertEqual(
                    artifact["training_config"]["agent_admission_mode"],
                    "before-begin",
                )
                self.assertEqual(
                    artifact["training_config"]["object_lock_scheduler"],
                    "bounded-priority",
                )
                evaluation = json.loads(Path(row["evaluation"]).read_text(encoding="utf-8"))
                self.assertEqual(evaluation["agent_admission_mode"], "before-begin")
                self.assertEqual(evaluation["object_lock_scheduler"], "bounded-priority")
                self.assertEqual(evaluation["prelock_wait_budget_mode"], "object")
                self.assertEqual(
                    evaluation["aggregates"][0]["agent_admission_mode"],
                    "before-begin",
                )
                self.assertEqual(
                    evaluation["aggregates"][0]["object_lock_scheduler"],
                    "bounded-priority",
                )

    def test_reward_cost_search_can_search_scheduler_and_wait_budget(self):
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
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = run_reward_cost_search(
                workload,
                workload_kind="ycsb",
                workload_config={"record_count": 4},
                output_dir=output_dir,
                lock_wait_costs=(50.0,),
                lock_action_costs=(0.02,),
                lock_queue_depth_costs=(0.11,),
                lock_handoff_costs=(0.13,),
                committing_count_costs=(0.17,),
                train_episodes=1,
                train_task_count=2,
                eval_task_count=2,
                eval_repeats=1,
                seed=31,
                workers=1,
                agent_slots=0,
                agent_admission_mode="before-begin",
                planning_delay_s=0.0,
                latency_distribution="fixed",
                latency_cv=0.8,
                latency_max_s=0.0,
                max_attempts=2,
                tokens_per_operation=10.0,
                background_workers=0,
                background_interval_s=0.0,
                object_lock_schedulers=("race", "bounded-priority"),
                object_lock_priority_bursts=(1,),
                prelock_wait_budget_s_values=(0.0, 0.007),
                prelock_wait_budget_modes=("object",),
                prelock_lease_modes=("hold",),
            )

            self.assertEqual(len(report["candidates"]), 4)
            self.assertEqual(
                report["config"]["object_lock_schedulers"],
                ["race", "bounded-priority"],
            )
            self.assertEqual(
                report["config"]["prelock_wait_budget_s_values"],
                [0.0, 0.007],
            )
            matrix = {
                (
                    row["object_lock_scheduler"],
                    row["object_lock_priority_burst"],
                    row["prelock_wait_budget_s"],
                    row["prelock_wait_budget_mode"],
                    row["prelock_lease_mode"],
                )
                for row in report["candidates"]
            }
            self.assertEqual(
                matrix,
                {
                    ("race", 1, 0.0, "object", "hold"),
                    ("race", 1, 0.007, "object", "hold"),
                    ("bounded-priority", 1, 0.0, "object", "hold"),
                    ("bounded-priority", 1, 0.007, "object", "hold"),
                },
            )
            for row in report["candidates"]:
                artifact = json.loads(
                    Path(row["policy_artifact"]).read_text(encoding="utf-8")
                )
                evaluation = json.loads(
                    Path(row["evaluation"]).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    artifact["training_config"]["object_lock_scheduler"],
                    row["object_lock_scheduler"],
                )
                self.assertEqual(
                    artifact["training_config"]["prelock_wait_budget_s"],
                    row["prelock_wait_budget_s"],
                )
                self.assertEqual(
                    evaluation["object_lock_scheduler"],
                    row["object_lock_scheduler"],
                )
                self.assertEqual(
                    evaluation["prelock_wait_budget_s"],
                    row["prelock_wait_budget_s"],
                )

    def test_reward_cost_search_cli_emits_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            exit_code = main(
                [
                    "--workload",
                    "ycsb",
                    "--output-dir",
                    tmp,
                    "--lock-wait-costs",
                    "50",
                    "--lock-action-costs",
                    "0.02",
                    "--lock-queue-depth-costs",
                    "0.11",
                    "--lock-handoff-costs",
                    "0.13",
                    "--committing-count-costs",
                    "0.17",
                    "--train-episodes",
                    "1",
                    "--train-task-count",
                    "2",
                    "--eval-task-count",
                    "2",
                    "--eval-repeats",
                    "1",
                    "--seed",
                    "29",
                    "--workers",
                    "1",
                    "--agent-slots",
                    "0",
                    "--agent-admission-mode",
                    "before-begin",
                    "--planning-delay-ms",
                    "0",
                    "--latency-distribution",
                    "fixed",
                    "--latency-max-ms",
                    "0",
                    "--max-attempts",
                    "2",
                    "--tokens-per-operation",
                    "10",
                    "--background-workers",
                    "0",
                    "--object-lock-scheduler",
                    "bounded-priority",
                    "--object-lock-priority-burst",
                    "3",
                    "--prelock-wait-budget-ms",
                    "7",
                    "--prelock-wait-budget-mode",
                    "object",
                    "--prelock-lease-mode",
                    "defer-until-after-planning",
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
            self.assertEqual(len(report["candidates"]), 1)
            self.assertEqual(report["best_candidate"]["lock_wait_cost_per_s"], 50.0)
            self.assertEqual(report["best_candidate"]["lock_queue_depth_cost"], 0.11)
            self.assertEqual(report["best_candidate"]["lock_handoff_cost"], 0.13)
            self.assertEqual(report["best_candidate"]["committing_count_cost"], 0.17)
            self.assertEqual(report["config"]["agent_admission_mode"], "before-begin")
            self.assertEqual(report["config"]["object_lock_scheduler"], "bounded-priority")
            self.assertEqual(report["config"]["prelock_wait_budget_s"], 0.007)
            self.assertEqual(report["config"]["prelock_wait_budget_mode"], "object")
            self.assertEqual(report["config"]["prelock_lease_mode"], "defer-until-after-planning")


if __name__ == "__main__":
    unittest.main()

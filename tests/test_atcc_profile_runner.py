import io
import json
import tempfile
import unittest
from pathlib import Path

from agent.evaluation.atcc_profile_runner import PROFILE_BY_NAME, main, run_profile_suite


class ATCCProfileRunnerTests(unittest.TestCase):
    def test_profile_catalog_has_high_medium_low_for_ycsb_and_tpcc(self):
        self.assertIn("ycsb-low", PROFILE_BY_NAME)
        self.assertIn("ycsb-medium", PROFILE_BY_NAME)
        self.assertIn("ycsb-high", PROFILE_BY_NAME)
        self.assertIn("tpcc-low", PROFILE_BY_NAME)
        self.assertIn("tpcc-medium", PROFILE_BY_NAME)
        self.assertIn("tpcc-high", PROFILE_BY_NAME)

    def test_ycsb_profiles_use_paper_aligned_hotspot_parameters(self):
        low = PROFILE_BY_NAME["ycsb-low"].config
        medium = PROFILE_BY_NAME["ycsb-medium"].config
        high = PROFILE_BY_NAME["ycsb-high"].config

        self.assertEqual(low["read_weight"], 0.95)
        self.assertEqual(low["update_weight"], 0.05)
        self.assertEqual(low["hotspot_fraction"], 0.0)
        self.assertEqual(medium["read_weight"], 0.90)
        self.assertEqual(medium["update_weight"], 0.10)
        self.assertEqual(medium["hotspot_fraction"], 0.10)
        self.assertEqual(medium["hotspot_access_probability"], 0.50)
        self.assertEqual(high["read_weight"], 0.50)
        self.assertEqual(high["update_weight"], 0.50)
        self.assertEqual(high["hotspot_fraction"], 0.10)
        self.assertEqual(high["hotspot_access_probability"], 0.75)

    def test_profile_suite_writes_training_eval_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = run_profile_suite(
                profiles=("ycsb-low",),
                output_dir=output_dir,
                train_episodes=1,
                train_task_count=2,
                eval_task_count=2,
                eval_repeats=1,
                seed=17,
                workers=1,
                agent_slots=0,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                latency_cv=0.8,
                latency_max_s=0.0,
                max_attempts=2,
                tokens_per_operation=10.0,
                background_workers=0,
                background_interval_s=0.0,
            )

            suite_path = output_dir / "phase_atcc_profile_suite.json"
            markdown_path = output_dir / "phase_atcc_profile_suite.md"
            policy_path = output_dir / "phase_atcc_ycsb-high_policy.json"
            eval_path = output_dir / "phase_atcc_ycsb-low_eval.json"

            self.assertTrue(suite_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(policy_path.exists())
            self.assertTrue(eval_path.exists())
            self.assertEqual(report["artifact_type"], "phase-aware-atcc-profile-suite")
            self.assertEqual(report["artifact_version"], 2)
            self.assertIn("class", report["atcc_state_schema"]["dimensions"])
            self.assertEqual(report["profiles"][0]["profile"], "ycsb-low")
            self.assertTrue(
                report["profiles"][0]["policy_artifact_schema"]["compatible"]
            )
            self.assertIn("## Artifact Schema", report["markdown"])
            self.assertIn("| ycsb-low |", report["markdown"])
            self.assertIn("| 2 | yes |", report["markdown"])
            self.assertIn("ATCC", report["markdown"])
            self.assertIn("adaptive-op-strict_vs_occ", report["profiles"][0]["comparisons"])

            eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertIn("class", eval_report["atcc_state_schema"]["dimensions"])
            self.assertTrue(eval_report["policy_artifact_schema"]["compatible"])
            self.assertEqual(len(eval_report["aggregates"]), 3)
            self.assertEqual(
                {row["strategy"] for row in eval_report["aggregates"]},
                {"occ", "2pl", "adaptive-op-strict"},
            )
            self.assertIn(
                "adaptive-op-strict_vs_2pl",
                report["profiles"][0]["comparisons"],
            )

    def test_profile_suite_reports_agent_like_execution_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = run_profile_suite(
                profiles=("ycsb-low",),
                output_dir=output_dir,
                train_episodes=1,
                train_task_count=2,
                eval_task_count=2,
                eval_repeats=1,
                seed=19,
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
                agent_execution_mode="staged",
                snapshot_timing="before-planning",
            )

            self.assertEqual(report["agent_admission_mode"], "before-begin")
            self.assertEqual(report["object_lock_scheduler"], "bounded-priority")
            self.assertEqual(report["object_lock_priority_burst"], 3)
            self.assertEqual(report["prelock_wait_budget_s"], 0.007)
            self.assertEqual(report["prelock_wait_budget_mode"], "object")
            self.assertEqual(report["prelock_lease_mode"], "defer-until-after-planning")
            self.assertEqual(report["agent_execution_mode"], "staged")
            self.assertEqual(report["snapshot_timing"], "before-planning")

            policy_path = output_dir / "phase_atcc_ycsb-high_policy.json"
            eval_path = output_dir / "phase_atcc_ycsb-low_eval.json"
            artifact = json.loads(policy_path.read_text(encoding="utf-8"))
            eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["training_config"]["agent_admission_mode"],
                "before-begin",
            )
            self.assertEqual(
                artifact["training_config"]["object_lock_scheduler"],
                "bounded-priority",
            )
            self.assertEqual(
                artifact["training_config"]["prelock_wait_budget_s"],
                0.007,
            )
            self.assertEqual(eval_report["agent_admission_mode"], "before-begin")
            self.assertEqual(eval_report["object_lock_scheduler"], "bounded-priority")
            self.assertEqual(eval_report["prelock_wait_budget_mode"], "object")
            self.assertEqual(eval_report["agent_execution_mode"], "staged")
            self.assertEqual(eval_report["snapshot_timing"], "before-planning")
            self.assertEqual(
                artifact["training_config"]["agent_execution_mode"],
                "staged",
            )
            self.assertEqual(
                artifact["training_config"]["snapshot_timing"],
                "before-planning",
            )
            self.assertTrue(artifact["runs"])
            self.assertTrue(eval_report["runs"])
            self.assertEqual(
                artifact["runs"][0]["agent_execution_mode"],
                "staged",
            )
            self.assertEqual(
                artifact["runs"][0]["snapshot_timing"],
                "before-planning",
            )
            self.assertEqual(
                eval_report["runs"][0]["agent_execution_mode"],
                "staged",
            )
            self.assertEqual(
                eval_report["runs"][0]["snapshot_timing"],
                "before-planning",
            )

    def test_profile_suite_can_apply_profile_specific_eval_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = run_profile_suite(
                profiles=("ycsb-low", "ycsb-high"),
                output_dir=output_dir,
                train_episodes=1,
                train_task_count=2,
                eval_task_count=2,
                eval_repeats=1,
                seed=29,
                workers=1,
                agent_slots=0,
                planning_delay_s=0.0,
                abort_retry_delay_s=0.0,
                latency_distribution="fixed",
                latency_cv=0.8,
                latency_max_s=0.0,
                max_attempts=2,
                tokens_per_operation=10.0,
                background_workers=0,
                background_interval_s=0.0,
                profile_eval_overrides={
                    "ycsb-high": {
                        "planning_delay_s": 0.123,
                        "abort_retry_delay_s": 0.456,
                        "workload_config": {"candidates_per_task": 5},
                    }
                },
            )

            low_eval = json.loads(
                (output_dir / "phase_atcc_ycsb-low_eval.json").read_text(
                    encoding="utf-8"
                )
            )
            high_eval = json.loads(
                (output_dir / "phase_atcc_ycsb-high_eval.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(low_eval["planning_delay_s"], 0.0)
            self.assertEqual(low_eval["abort_retry_delay_s"], 0.0)
            self.assertEqual(low_eval["workload_config"]["candidates_per_task"], 4)
            self.assertEqual(high_eval["planning_delay_s"], 0.123)
            self.assertEqual(high_eval["abort_retry_delay_s"], 0.456)
            self.assertEqual(high_eval["workload_config"]["candidates_per_task"], 5)
            self.assertEqual(
                report["profiles"][1]["eval_overrides"]["planning_delay_s"],
                0.123,
            )

    def test_profile_runner_cli_emits_suite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            exit_code = main(
                [
                    "--profiles",
                    "ycsb-low",
                    "--output-dir",
                    tmp,
                    "--train-episodes",
                    "1",
                    "--train-task-count",
                    "2",
                    "--eval-task-count",
                    "2",
                    "--eval-repeats",
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
                    "--tokens-per-operation",
                    "10",
                    "--background-workers",
                    "0",
                ],
                stdout=stdout,
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["atcc_state_schema"]["version"], 2)
            self.assertEqual(report["profiles"][0]["profile"], "ycsb-low")
            self.assertTrue(
                (Path(tmp) / "phase_atcc_profile_suite.md").exists()
            )

    def test_profile_runner_cli_accepts_profile_eval_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            exit_code = main(
                [
                    "--profiles",
                    "ycsb-high",
                    "--output-dir",
                    tmp,
                    "--train-episodes",
                    "1",
                    "--train-task-count",
                    "2",
                    "--eval-task-count",
                    "2",
                    "--eval-repeats",
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
                    "2",
                    "--tokens-per-operation",
                    "10",
                    "--background-workers",
                    "0",
                    "--profile-eval-overrides",
                    '{"ycsb-high":{"planning_delay_ms":123,"abort_retry_delay_ms":456,"workload_config":{"candidates_per_task":5}}}',
                ],
                stdout=stdout,
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(stdout.getvalue())
            eval_report = json.loads(
                (Path(tmp) / "phase_atcc_ycsb-high_eval.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(eval_report["planning_delay_s"], 0.123)
            self.assertEqual(eval_report["abort_retry_delay_s"], 0.456)
            self.assertEqual(eval_report["workload_config"]["candidates_per_task"], 5)
            self.assertEqual(
                report["config"]["profile_eval_overrides"]["ycsb-high"][
                    "planning_delay_s"
                ],
                0.123,
            )

    def test_profile_runner_cli_loads_profile_eval_overrides_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            override_path = Path(tmp) / "profile-eval-overrides.json"
            override_path.write_text(
                json.dumps(
                    {
                        "ycsb-high": {
                            "planning_delay_ms": 100,
                            "abort_retry_delay_ms": 500,
                            "workload_config": {"candidates_per_task": 5},
                        }
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            exit_code = main(
                [
                    "--profiles",
                    "ycsb-high",
                    "--output-dir",
                    tmp,
                    "--train-episodes",
                    "1",
                    "--train-task-count",
                    "2",
                    "--eval-task-count",
                    "2",
                    "--eval-repeats",
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
                    "2",
                    "--tokens-per-operation",
                    "10",
                    "--background-workers",
                    "0",
                    "--profile-eval-overrides-file",
                    str(override_path),
                    "--profile-eval-overrides",
                    '{"ycsb-high":{"planning_delay_ms":123}}',
                ],
                stdout=stdout,
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(stdout.getvalue())
            eval_report = json.loads(
                (Path(tmp) / "phase_atcc_ycsb-high_eval.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(eval_report["planning_delay_s"], 0.123)
            self.assertEqual(eval_report["abort_retry_delay_s"], 0.5)
            self.assertEqual(eval_report["workload_config"]["candidates_per_task"], 5)
            self.assertEqual(
                report["config"]["profile_eval_overrides"]["ycsb-high"],
                {
                    "planning_delay_s": 0.123,
                    "abort_retry_delay_s": 0.5,
                    "workload_config": {"candidates_per_task": 5},
                },
            )

    def test_profile_suite_can_run_full_traditional_cc_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_profile_suite(
                profiles=("ycsb-low",),
                output_dir=Path(tmp),
                strategy_set="full",
                train_episodes=1,
                train_task_count=2,
                eval_task_count=2,
                eval_repeats=1,
                seed=23,
                workers=1,
                agent_slots=0,
                planning_delay_s=0.0,
                latency_distribution="fixed",
                latency_cv=0.8,
                latency_max_s=0.0,
                max_attempts=2,
                tokens_per_operation=10.0,
                background_workers=0,
                background_interval_s=0.0,
            )

            self.assertEqual(
                set(report["config"]["strategies"]),
                {
                    "occ",
                    "2pl-nowait",
                    "2pl-wait-die",
                    "mvcc-full",
                    "silo-full",
                    "tictoc-full",
                    "adaptive-op-strict",
                },
            )
            self.assertEqual(
                {row["strategy"] for row in report["profiles"][0]["aggregates"]},
                set(report["config"]["strategies"]),
            )


if __name__ == "__main__":
    unittest.main()

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
            )

            self.assertEqual(report["agent_admission_mode"], "before-begin")
            self.assertEqual(report["object_lock_scheduler"], "bounded-priority")
            self.assertEqual(report["object_lock_priority_burst"], 3)
            self.assertEqual(report["prelock_wait_budget_s"], 0.007)
            self.assertEqual(report["prelock_wait_budget_mode"], "object")
            self.assertEqual(report["prelock_lease_mode"], "defer-until-after-planning")

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


if __name__ == "__main__":
    unittest.main()

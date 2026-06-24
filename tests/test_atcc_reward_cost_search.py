import io
import json
import tempfile
import unittest
from pathlib import Path

from agent.evaluation.atcc_reward_cost_search import main, run_reward_cost_search
from agent.workloads import YCSBConfig, build_agent_workload


class ATCCRewardCostSearchTests(unittest.TestCase):
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
                report["artifact_type"],
                "phase-aware-atcc-reward-cost-search",
            )
            self.assertEqual(len(report["candidates"]), 2)
            self.assertEqual(len(report["ranked_candidates"]), 2)
            self.assertTrue(report["best_candidate"])
            self.assertTrue((output_dir / "atcc_reward_cost_search.json").exists())
            for row in report["candidates"]:
                self.assertTrue(Path(row["policy_artifact"]).exists())
                self.assertTrue(Path(row["evaluation"]).exists())
                self.assertIn("atcc_throughput", row)
                self.assertIn("atcc_vs_occ_throughput_x", row)

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


if __name__ == "__main__":
    unittest.main()

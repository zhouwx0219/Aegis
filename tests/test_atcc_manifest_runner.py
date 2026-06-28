import json
import tempfile
import unittest
from pathlib import Path

from agent.evaluation.atcc_manifest_runner import run_manifest_suite


class ATCCManifestRunnerTests(unittest.TestCase):
    def test_manifest_runner_writes_runs_and_combined_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest = {
                "defaults": {
                    "strategies": "occ,adaptive-hybrid",
                    "strategy_order": "interleave-all-strategies",
                    "interleave_blocks": 1,
                    "hybrid_selected_fast_through": True,
                    "task_count": 1,
                    "repeats": 1,
                    "workers": 1,
                    "agent_slots": 0,
                    "planning_delay_ms": 0,
                    "abort_retry_delay_ms": 0,
                    "latency_distribution": "fixed",
                    "latency_max_ms": 0,
                    "max_attempts": 1,
                    "tokens_per_operation": 10,
                    "background_workers": 0,
                    "agent_execution_mode": "staged",
                    "snapshot_timing": "before-planning",
                },
                "profiles": [
                    {
                        "name": "ycsb-low",
                        "workload": "ycsb",
                        "seeds": [3],
                        "workload_config": {
                            "record_count": 8,
                            "field_count": 1,
                            "requests_per_task": 1,
                            "candidates_per_task": 1,
                            "read_weight": 1.0,
                            "update_weight": 0.0,
                            "zipf_theta": 0.0,
                            "hotspot_fraction": 0.0,
                            "hotspot_access_probability": 0.0,
                        },
                    }
                ],
            }
            manifest_path = base / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            output_dir = base / "out"

            report = run_manifest_suite(manifest_path, output_dir=output_dir)

            self.assertEqual(report["artifact_type"], "atcc-retry-manifest-suite")
            self.assertEqual(report["profiles"][0]["name"], "ycsb-low")
            self.assertEqual(report["profiles"][0]["runs"][0]["seed"], 3)
            self.assertTrue((output_dir / "ycsb-low-seed3-r1.json").exists())
            self.assertTrue((output_dir / "manifest-combined-summary.csv").exists())
            stats_path = output_dir / "manifest-combined-stats.json"
            self.assertTrue(stats_path.exists())
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["ycsb-low"]["runs"], 1)
            self.assertIn("mean", stats["ycsb-low"])


if __name__ == "__main__":
    unittest.main()

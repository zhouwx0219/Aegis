import csv
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class CompareScriptContractTests(unittest.TestCase):
    def test_ycsb_compare_accepts_policy_artifact(self):
        script = (REPO_ROOT / "scripts" / "run_ycsb_compare.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("[string]$PolicyArtifact", script)
        self.assertIn("[double]$PolicyEpsilon", script)
        self.assertIn("--policy-artifact", script)
        self.assertIn("--policy-epsilon", script)
        self.assertIn("Convert-ToWslPath $PolicyArtifact", script)
        self.assertIn("adaptive-hybrid", script)

    def test_tpcc_compare_accepts_policy_artifact(self):
        script = (REPO_ROOT / "scripts" / "run_tpcc_compare.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("[string]$PolicyArtifact", script)
        self.assertIn("[double]$PolicyEpsilon", script)
        self.assertIn("--policy-artifact", script)
        self.assertIn("--policy-epsilon", script)
        self.assertIn("Convert-ToWslPath $PolicyArtifact", script)
        self.assertIn("adaptive-hybrid", script)

    def test_compare_scripts_append_optional_policy_args_before_join(self):
        for name in ("run_ycsb_compare.ps1", "run_tpcc_compare.ps1"):
            script = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")

            self.assertIn("$CommonArgs += $PolicyArtifactArgs", script)
            self.assertIn("$Common = $CommonArgs -join", script)
            self.assertNotIn("$PolicyArtifactArgs,", script)

    def test_summarizer_preserves_refresh_rebase_metrics(self):
        spec = importlib.util.spec_from_file_location(
            "summarize_retry_results",
            REPO_ROOT / "scripts" / "summarize_retry_results.py",
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ycsb-medium.json").write_text(
                json.dumps(
                    {
                        "aggregates": [
                            {
                                "strategy": "adaptive-op-strict",
                                "policy_variant": "ycsb-strict-tuned",
                                "committed_throughput": 12.5,
                                "lease_refresh_regenerations": 3,
                                "lease_refresh_replayed_operations": 0,
                                "lease_refresh_rebased_writes": 9,
                                "lease_refresh_rebased_writes_per_task": 4.5,
                                "selected_strategy_counts": {"mvcc-full": 2},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = root / "summary.csv"
            module.summarize(root, output)

            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["lease_refresh_regenerations"], "3")
            self.assertEqual(rows[0]["lease_refresh_replayed_operations"], "0")
            self.assertEqual(rows[0]["lease_refresh_rebased_writes"], "9")
            self.assertEqual(rows[0]["lease_refresh_rebased_writes_per_task"], "4.5")
            self.assertEqual(rows[0]["selected_strategy_counts"], '{"mvcc-full": 2}')


if __name__ == "__main__":
    unittest.main()

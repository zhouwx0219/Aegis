import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_aegis_reproduction import VerificationError, verify_reproduction


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifyAegisReproductionTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> Path:
        paper = root / "paper.pdf"
        policy = root / "policy.json"
        report = root / "report.json"
        paper.write_bytes(b"paper")
        policy.write_text(json.dumps({"medoids_per_group": 1}), encoding="utf-8")
        report.write_text("{}", encoding="utf-8")

        guarded_files = {}
        zero_files = {}
        for seed, aegis_tps, baseline_tps in ((11, 30.0, 10.0), (12, 33.0, 11.0)):
            guarded = root / f"guarded-{seed}.csv"
            zero = root / f"zero-{seed}.csv"
            self.write_results(guarded, seed, aegis_tps, baseline_tps, retry_count=0.5)
            self.write_results(zero, seed, aegis_tps - 3.0, baseline_tps, retry_count=0.0)
            guarded_files[str(seed)] = {"path": guarded.name, "sha256": sha256(guarded)}
            zero_files[str(seed)] = {"path": zero.name, "sha256": sha256(zero)}

        manifest = {
            "artifact_type": "cast-das-aegis-small-scale-reproduction-manifest",
            "paper": {"path": paper.name, "sha256": sha256(paper)},
            "training": {
                "seeds": [1, 2],
                "policy": {"path": policy.name, "sha256": sha256(policy)},
                "report": {"path": report.name, "sha256": sha256(report)},
            },
            "evaluation": {
                "seeds": [11, 12],
                "training_seed_overlap": [],
            },
            "guarded_path": {
                "files": guarded_files,
                "aegis_tps_mean": 31.5,
                "best_mean_baseline": "silo",
                "best_mean_baseline_tps": 10.5,
                "speedup_vs_best_mean_baseline": 3.0,
            },
            "pure_policy_path": {
                "files": {},
                "speedup_vs_best_mean_baseline": 2.85,
            },
            "zero_retry_path": {
                "max_attempts": 1,
                "retry_budget": 0,
                "allow_retries": False,
                "files": zero_files,
                "aegis_tps_mean": 28.5,
                "best_mean_baseline": "silo",
                "best_mean_baseline_tps": 10.5,
                "speedup_vs_best_mean_baseline": 28.5 / 10.5,
            },
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    @staticmethod
    def write_results(
        path: Path,
        seed: int,
        aegis_tps: float,
        baseline_tps: float,
        *,
        retry_count: float,
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "seed",
                    "cc",
                    "status",
                    "agent_tps",
                    "agent_commit_rate",
                    "agent_avg_retry_count",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "seed": seed,
                    "cc": "silo",
                    "status": "ok",
                    "agent_tps": baseline_tps,
                    "agent_commit_rate": 0.2,
                    "agent_avg_retry_count": retry_count,
                }
            )
            writer.writerow(
                {
                    "seed": seed,
                    "cc": "paper-atcc",
                    "status": "ok",
                    "agent_tps": aegis_tps,
                    "agent_commit_rate": 0.6,
                    "agent_avg_retry_count": retry_count,
                }
            )

    def test_verifies_hashes_seed_isolation_metrics_and_zero_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.build_fixture(root)

            result = verify_reproduction(manifest, repo_root=root)

            self.assertEqual(1, result["policy_medoids_per_group"])
            self.assertEqual(3.0, result["guarded_speedup"])
            self.assertEqual(0.0, result["zero_retry_all_average_retries"])

    def test_rejects_result_changed_after_manifest_was_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.build_fixture(root)
            changed = root / "zero-11.csv"
            changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            with self.assertRaisesRegex(VerificationError, "SHA-256 mismatch"):
                verify_reproduction(manifest, repo_root=root)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from scripts.verify_five_retry_paper_matrix import verify_matrix


ROOT = Path(__file__).resolve().parents[1]


class VerifyFiveRetryPaperMatrixTests(unittest.TestCase):
    def test_archived_matrix_hashes_metrics_and_paper_retry_invariants(self):
        result = verify_matrix(
            ROOT / "results/reproduction/five_retry_paper_acceptance_manifest.json"
        )

        self.assertEqual(6, result["runtime"]["max_attempts"])
        self.assertEqual(5, result["runtime"]["retry_budget"])
        self.assertTrue(result["all_checks_pass"])
        self.assertGreater(result["credit"]["speedup_vs_best"], 1.24)
        self.assertGreater(
            result["figure13"]["write_0.9"]["speedup_vs_2pl"],
            result["figure13"]["write_0.9"]["speedup_vs_best"],
        )


if __name__ == "__main__":
    unittest.main()

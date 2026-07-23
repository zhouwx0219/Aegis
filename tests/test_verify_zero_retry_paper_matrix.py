import unittest
from pathlib import Path

from scripts.verify_zero_retry_paper_matrix import verify_matrix


ROOT = Path(__file__).resolve().parents[1]


class VerifyZeroRetryPaperMatrixTests(unittest.TestCase):
    def test_archived_matrix_hashes_metrics_and_zero_retry_invariants(self):
        result = verify_matrix(
            ROOT / "results/reproduction/zero_retry_paper_acceptance_manifest.json"
        )

        self.assertTrue(result["zero_retry_invariants"])
        self.assertTrue(result["mechanism_checks_pass"])
        self.assertFalse(result["full_paper_claims_pass"])
        self.assertGreaterEqual(result["credit"]["aegis"]["commit_rate"], 0.98)


if __name__ == "__main__":
    unittest.main()

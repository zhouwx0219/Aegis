import io
import json
import unittest

from agent.cli import benchmark


class DeliveryBenchmarkCliTests(unittest.TestCase):
    def test_benchmark_cli_emits_delivery_report(self):
        stdout = io.StringIO()
        status = benchmark.main(
            [
                "--workload",
                "ycsb",
                "--profile",
                "low",
                "--strategies",
                "quick",
                "--task-count",
                "3",
                "--workers",
                "1",
                "--agent-slots",
                "1",
                "--planning-delay-ms",
                "0",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        self.assertEqual("delivery-benchmark", report["mode"])
        self.assertEqual("low", report["profile"])
        self.assertEqual(["occ", "adaptive-op-strict", "transaction-atcc-strict"], report["strategies"])
        self.assertEqual(1, len(report["workloads"]))
        self.assertTrue(report["aggregates"])
        self.assertIn("committed_throughput", report["aggregates"][0])


if __name__ == "__main__":
    unittest.main()

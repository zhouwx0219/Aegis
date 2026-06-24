import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from agent.evaluation.cc_matrix_cli import main


class StrategyMatrixCliTests(unittest.TestCase):
    def test_json_stdout_report_is_reproducible_envelope(self):
        stdout = io.StringIO()
        rc = main(
            [
                "--workload",
                "ycsb",
                "--records",
                "4",
                "--fields",
                "2",
                "--requests-per-task",
                "2",
                "--candidates",
                "2",
                "--task-count",
                "3",
                "--seed",
                "5",
                "--repeats",
                "2",
                "--contention-window",
                "2",
                "--strategies",
                "semantic,occ",
            ],
            stdout=stdout,
        )
        self.assertEqual(rc, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["workload"], "agent-ycsb-semantic")
        self.assertEqual(report["workload_kind"], "ycsb")
        self.assertEqual(report["workload_layer"], "semantic")
        self.assertEqual(report["workload_manifest"]["benchmark_family"], "YCSB")
        self.assertEqual(report["workload_manifest"]["source_system"], "DBx1000")
        self.assertEqual(report["strategies"], ["semantic", "occ"])
        self.assertEqual(report["task_count"], 3)
        self.assertEqual(report["seed"], 5)
        self.assertEqual(report["seeds"], [5, 6])
        self.assertEqual(report["repeats"], 2)
        self.assertEqual(len(report["summaries"]), 4)
        self.assertEqual(len(report["aggregates"]), 2)
        self.assertEqual(
            {summary["strategy"] for summary in report["summaries"]},
            {"semantic", "occ"},
        )
        self.assertEqual(
            {aggregate["runs"] for aggregate in report["aggregates"]},
            {2},
        )

    def test_csv_output_file_contains_one_row_per_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "matrix.csv"
            rc = main(
                [
                    "--workload",
                    "tpcc",
                    "--warehouses",
                    "1",
                    "--districts-per-warehouse",
                    "1",
                    "--customers-per-district",
                    "1",
                    "--items",
                    "2",
                    "--initial-stock",
                    "100",
                    "--order-lines",
                    "1",
                    "--candidates",
                    "1",
                    "--transaction-mix",
                    "new_order:1.0",
                    "--task-count",
                    "2",
                    "--contention-window",
                    "2",
                    "--strategies",
                    "semantic,occ",
                    "--format",
                    "csv",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["strategy"] for row in rows], ["semantic", "occ"])
        self.assertEqual(
            [row["workload"] for row in rows],
            ["agent-tpcc-semantic", "agent-tpcc-semantic"],
        )
        self.assertEqual([row["benchmark_family"] for row in rows], ["TPC-C", "TPC-C"])
        self.assertEqual([row["source_system"] for row in rows], ["DBx1000", "DBx1000"])
        self.assertEqual([row["task_count"] for row in rows], ["2", "2"])

    def test_csv_aggregate_section_contains_one_row_per_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "aggregate.csv"
            rc = main(
                [
                    "--workload",
                    "ycsb",
                    "--records",
                    "4",
                    "--fields",
                    "2",
                    "--requests-per-task",
                    "2",
                    "--candidates",
                    "2",
                    "--task-count",
                    "2",
                    "--repeats",
                    "2",
                    "--strategies",
                    "semantic,occ",
                    "--format",
                    "csv",
                    "--csv-section",
                    "aggregates",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["strategy"] for row in rows], ["semantic", "occ"])
        self.assertEqual([row["runs"] for row in rows], ["2", "2"])
        self.assertEqual([row["task_count_per_run"] for row in rows], ["2", "2"])
        self.assertEqual([row["total_task_count"] for row in rows], ["4", "4"])


if __name__ == "__main__":
    unittest.main()

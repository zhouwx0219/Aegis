import unittest
from pathlib import Path

from agent.evaluation.dbx1000_native import (
    Dbx1000NativeConfig,
    list_native_strategies,
    parse_summary,
)


class Dbx1000NativeRunnerTests(unittest.TestCase):
    def test_native_strategy_catalog_contains_authoritative_dbx1000_cc(self):
        strategies = list_native_strategies()
        for name in ("occ", "mvcc", "tictoc", "silo", "no_wait", "wait_die", "dl_detect"):
            self.assertIn(name, strategies)
        self.assertEqual(strategies["tictoc"]["cc_alg"], "TICTOC")
        self.assertEqual(strategies["no_wait"]["family"], "pessimistic")

    def test_summary_parser_extracts_numeric_fields(self):
        summary = parse_summary(
            "[summary] txn_cnt=10, abort_cnt=2, run_time=0.500000, latency=0.01"
        )
        self.assertEqual(summary["txn_cnt"], 10)
        self.assertEqual(summary["abort_cnt"], 2)
        self.assertEqual(summary["run_time"], 0.5)
        self.assertEqual(summary["latency"], 0.01)

    def test_native_runtime_args_follow_workload_family(self):
        ycsb = Dbx1000NativeConfig(workload="ycsb", ycsb_req_per_query=4)
        self.assertIn("-R4", ycsb.runtime_args(Path("out.txt")))

        tpcc = Dbx1000NativeConfig(workload="tpcc", tpcc_payment_perc=0.0)
        args = tpcc.runtime_args(Path("out.txt"))
        self.assertIn("-n1", args)
        self.assertIn("-Tp0.0", args)


if __name__ == "__main__":
    unittest.main()

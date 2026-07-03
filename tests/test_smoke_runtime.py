import unittest

from agent.cli.smoke import run_smoke_checks


class DeliverySmokeRuntimeTests(unittest.TestCase):
    def test_core_runtime_and_workloads_are_runnable(self):
        report = run_smoke_checks()

        self.assertTrue(report["ok"])
        self.assertEqual("dbx1000", report["native_backend"])
        self.assertEqual("1", report["runtime_counter"])
        self.assertGreater(report["transaction_atcc_decisions"], 0)
        self.assertTrue(report["ycsb_task_committed"])
        self.assertTrue(report["tpcc_task_committed"])


if __name__ == "__main__":
    unittest.main()

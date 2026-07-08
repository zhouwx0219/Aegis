import unittest

from agent.cli.smoke import run_smoke_checks
from agent.runtime import AgentTransactionManager
from agent.workloads import build_workload, execute_task, register_workload


class SmokeRuntimeTests(unittest.TestCase):
    def test_core_runtime_and_workloads_are_runnable(self):
        report = run_smoke_checks()

        self.assertTrue(report["ok"])
        self.assertEqual("dbx1000", report["native_backend"])
        self.assertEqual("1", report["runtime_counter"])
        self.assertTrue(report["ycsb_task_committed"])
        self.assertTrue(report["tpcc_task_committed"])

    def test_conflicting_transactions_abort_on_stale_version(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", "0", kind="counter")

        first = manager.begin("first")
        second = manager.begin("second")
        first.read("counter")
        first.write("counter", "1")
        second.read("counter")
        second.write("counter", "1")

        first_result = first.commit("occ")
        second_result = second.commit("occ")

        self.assertTrue(first_result.committed)
        self.assertFalse(second_result.committed)
        self.assertEqual("aborted", second_result.state.value)
        self.assertIn("counter", second_result.conflict_object_ids)

    def test_workload_task_can_commit(self):
        manager = AgentTransactionManager()
        workload = build_workload("ycsb", "low")
        register_workload(manager, workload)
        task = workload.generate_tasks(1, seed=1)[0]

        result = execute_task(manager, task, cc="dynamic-atcc")

        self.assertTrue(result.committed)


if __name__ == "__main__":
    unittest.main()

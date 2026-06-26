import threading
import time
import unittest

from agent.runtime import AgentTransactionManager, TransactionState


class FullTraditionalCCCatalogTests(unittest.TestCase):
    def test_catalog_exposes_full_traditional_cc_without_replacing_adapters(self):
        manager = AgentTransactionManager()
        strategies = manager.cc_strategies()

        for name in (
            "2pl-nowait",
            "2pl-wait-die",
            "mvcc-full",
            "silo-full",
            "tictoc-full",
        ):
            self.assertIn(name, strategies)
            self.assertEqual(strategies[name]["source"], "agent-full")
            self.assertEqual(strategies[name]["selector"], "transaction_protocol")

        self.assertEqual(strategies["mvcc"]["source"], "DBx1000-inspired")
        self.assertEqual(strategies["silo"]["source"], "DBx1000-inspired")
        self.assertEqual(strategies["tictoc"]["source"], "DBx1000-inspired")


class FullTraditionalCCBehaviorTests(unittest.TestCase):
    def _single_write_commit(self, strategy: str):
        manager = AgentTransactionManager()
        manager.register_object("row", "old", kind="row")
        txn = manager.begin("writer")
        txn.add_candidate("writer", quality=1, gen_cost=0).overwrite("row", "new")

        result = txn.commit(strategy=strategy)

        self.assertTrue(result.committed)
        self.assertEqual(manager.value_of("row"), "new")
        validate = [
            event
            for event in manager.traces()[-1]["events"]
            if event["kind"] == "validate"
        ][0]
        self.assertEqual(validate["detail"]["selected_cc"], strategy)

    def test_full_traditional_cc_strategies_commit_single_write(self):
        for strategy in (
            "2pl-nowait",
            "2pl-wait-die",
            "mvcc-full",
            "silo-full",
            "tictoc-full",
        ):
            with self.subTest(strategy=strategy):
                self._single_write_commit(strategy)

    def test_2pl_nowait_aborts_immediately_on_conflicting_lock(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "old", kind="row")
        holder = manager.begin("holder")
        waiter = manager.begin("waiter")
        waiter.add_candidate("waiter", quality=1, gen_cost=0).overwrite("row", "new")

        lock_table = manager.commit_protocol.traditional_executor.two_phase_locks
        with lock_table.acquire(["row"], owner=holder, mode="x", policy="nowait"):
            result = waiter.commit(strategy="2pl-nowait")

        self.assertEqual(result.state, TransactionState.ABORTED)
        self.assertEqual(result.action, "regenerate_required")
        self.assertIn("no-wait", result.reason)
        self.assertEqual(manager.value_of("row"), "old")

    def test_2pl_wait_die_aborts_younger_waiter_on_older_holder(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "old", kind="row")
        older = manager.begin("older")
        time.sleep(0.002)
        younger = manager.begin("younger")
        younger.add_candidate("younger", quality=1, gen_cost=0).overwrite("row", "new")

        lock_table = manager.commit_protocol.traditional_executor.two_phase_locks
        with lock_table.acquire(["row"], owner=older, mode="x", policy="wait-die"):
            result = younger.commit(strategy="2pl-wait-die")

        self.assertEqual(result.state, TransactionState.ABORTED)
        self.assertEqual(result.action, "regenerate_required")
        self.assertIn("wait-die", result.reason)
        self.assertEqual(manager.value_of("row"), "old")

    def test_mvcc_full_allows_snapshot_read_with_disjoint_write(self):
        manager = AgentTransactionManager()
        manager.register_object("read-row", "r0", kind="row")
        manager.register_object("write-row", "w0", kind="row")

        snapshot_txn = manager.begin("snapshot")
        snapshot_txn.read("read-row")
        snapshot_txn.add_candidate("snapshot", quality=1, gen_cost=0).overwrite(
            "write-row", "w1"
        )
        concurrent = manager.begin("concurrent")
        concurrent.add_candidate("concurrent", quality=1, gen_cost=0).overwrite(
            "read-row", "r1"
        )
        self.assertTrue(concurrent.commit(strategy="occ").committed)

        result = snapshot_txn.commit(strategy="mvcc-full")

        self.assertTrue(result.committed)
        self.assertEqual(manager.value_of("read-row"), "r1")
        self.assertEqual(manager.value_of("write-row"), "w1")

    def test_mvcc_full_aborts_on_write_write_conflict(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "old", kind="row")
        older = manager.begin("older")
        older.add_candidate("older", quality=1, gen_cost=0).overwrite("row", "older")
        newer = manager.begin("newer")
        newer.add_candidate("newer", quality=1, gen_cost=0).overwrite("row", "newer")
        self.assertTrue(newer.commit(strategy="mvcc-full").committed)

        result = older.commit(strategy="mvcc-full")

        self.assertEqual(result.state, TransactionState.ABORTED)
        self.assertEqual(result.action, "regenerate_required")
        self.assertEqual(manager.value_of("row"), "newer")

    def test_silo_full_aborts_when_read_tid_changes(self):
        manager = AgentTransactionManager()
        manager.register_object("read-row", "r0", kind="row")
        manager.register_object("write-row", "w0", kind="row")

        silo = manager.begin("silo")
        silo.read("read-row")
        silo.add_candidate("silo", quality=1, gen_cost=0).overwrite("write-row", "w1")
        concurrent = manager.begin("concurrent")
        concurrent.add_candidate("concurrent", quality=1, gen_cost=0).overwrite(
            "read-row", "r1"
        )
        self.assertTrue(concurrent.commit(strategy="silo-full").committed)

        result = silo.commit(strategy="silo-full")

        self.assertEqual(result.state, TransactionState.ABORTED)
        self.assertEqual(result.action, "regenerate_required")
        self.assertEqual(manager.value_of("write-row"), "w0")

    def test_tictoc_full_allows_timestamp_valid_disjoint_read_write(self):
        manager = AgentTransactionManager()
        manager.register_object("read-row", "r0", kind="row")
        manager.register_object("write-row", "w0", kind="row")

        tictoc = manager.begin("tictoc")
        tictoc.read("read-row")
        tictoc.add_candidate("tictoc", quality=1, gen_cost=0).overwrite(
            "write-row", "w1"
        )
        concurrent = manager.begin("concurrent")
        concurrent.add_candidate("concurrent", quality=1, gen_cost=0).overwrite(
            "read-row", "r1"
        )
        self.assertTrue(concurrent.commit(strategy="tictoc-full").committed)

        result = tictoc.commit(strategy="tictoc-full")

        self.assertTrue(result.committed)
        self.assertEqual(manager.value_of("read-row"), "r1")
        self.assertEqual(manager.value_of("write-row"), "w1")


if __name__ == "__main__":
    unittest.main()

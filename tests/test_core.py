import io
import json
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError

import cast_core as cc

from agent.llm import deepseek_client
from agent.runtime import (
    AgentTransactionManager,
    FirstCandidateBranchSemantics,
    TransactionState,
)


def overwrite_branch(store, branch_id, object_id, value, quality=1.0):
    current = store.get(object_id)
    intent = cc.WriteIntent()
    intent.object_id = object_id
    intent.intent_type = cc.IntentType.kOverwrite
    write = cc.BranchWrite()
    write.object_id = object_id
    write.base_value = current.value
    write.base_version = current.version
    write.branch_value = value
    write.intent = intent
    branch = cc.SpeculativeBranch()
    branch.branch_id = branch_id
    branch.writes = [write]
    branch.quality = quality
    return branch


class StoreAndCommitTests(unittest.TestCase):
    def test_escrow_rejects_invalid_or_exhausted_reservations(self):
        account = cc.EscrowAccount(1, 0)
        self.assertTrue(account.reserve(1))
        self.assertFalse(account.reserve(1))
        self.assertFalse(account.oversold())
        with self.assertRaises(ValueError):
            account.reserve(-1)

    def test_delete_keeps_version_monotonic(self):
        store = cc.VersionedObjectStore()
        store.put("key", "value")
        version = store.get_version("key")
        self.assertTrue(store.delete_if_version("key", version))
        deleted = store.get("key")
        self.assertFalse(deleted.exists)
        self.assertGreater(deleted.version, version)
        self.assertTrue(store.put_if_version("key", deleted.version, "new"))
        self.assertGreater(store.get_version("key"), deleted.version)

    def test_raw_duplicate_write_is_rejected(self):
        store = cc.VersionedObjectStore()
        store.put("counter", "0")
        branch = overwrite_branch(store, "duplicate", "counter", "1")
        duplicate = overwrite_branch(store, "duplicate-2", "counter", "2").writes[0]
        branch.writes = [branch.writes[0], duplicate]
        stats = cc.CostStats()
        kernel = cc.CostAsymmetricCommit(store, cc.CostModel(1.0, 0.01))
        outcome = kernel.commit_task(
            [branch], cc.SemanticConcurrencyControl(), stats
        )
        self.assertTrue(outcome.rejected)
        self.assertIn("duplicate write target", outcome.reason)
        self.assertEqual(store.get("counter").value, "0")

    def test_reselect_uses_highest_quality_valid_candidate(self):
        store = cc.VersionedObjectStore()
        for key in ("top", "low", "middle"):
            store.put(key, "old")
        top = overwrite_branch(store, "top", "top", "top-value", quality=10)
        low = overwrite_branch(store, "low", "low", "low-value", quality=1)
        middle = overwrite_branch(store, "middle", "middle", "middle-value", quality=5)
        store.put("top", "concurrent")
        stats = cc.CostStats()
        kernel = cc.CostAsymmetricCommit(store, cc.CostModel(1.0, 0.01))
        outcome = kernel.commit_task(
            [top, low, middle], cc.SemanticConcurrencyControl(), stats
        )
        self.assertTrue(outcome.committed)
        self.assertEqual(outcome.winner_branch_id, "middle")
        self.assertEqual(store.get("low").value, "old")
        self.assertEqual(store.get("middle").value, "middle-value")

    def test_raw_conflict_outcome_reports_precise_object_ids(self):
        store = cc.VersionedObjectStore()
        store.put("guard", "old")
        store.put("output", "")

        read = cc.BranchRead()
        read.object_id = "guard"
        read.version = store.get_version("guard")
        branch = overwrite_branch(store, "stale-read", "output", "saw-old")
        branch.read_set = [read]

        store.put("guard", "new")
        stats = cc.CostStats()
        kernel = cc.CostAsymmetricCommit(store, cc.CostModel(1.0, 0.01))
        outcome = kernel.commit_task([branch], cc.StrictOccConcurrencyControl(), stats)

        self.assertTrue(outcome.needs_regeneration)
        self.assertEqual(outcome.conflict_object_ids, ["guard"])
        self.assertEqual(store.get("output").value, "")


class RuntimeTests(unittest.TestCase):
    def test_custom_branch_semantics_can_limit_candidate_scope(self):
        manager = AgentTransactionManager(
            branch_semantics=FirstCandidateBranchSemantics()
        )
        manager.register_object("row", "old", kind="row")

        txn = manager.begin("first-only")
        txn.add_candidate("first", quality=1, gen_cost=0).overwrite("row", "first")
        txn.add_candidate("better", quality=10, gen_cost=0).overwrite("row", "better")

        result = txn.commit()
        self.assertTrue(result.committed)
        self.assertEqual(result.winner_branch_id, "first")
        self.assertEqual(manager.value_of("row"), "first")

    def test_candidate_rejects_duplicate_object(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", 0, kind="counter")
        txn = manager.begin("duplicate")
        candidate = txn.add_candidate("candidate", quality=1, gen_cost=0)
        candidate.delta("counter", 1)
        with self.assertRaisesRegex(ValueError, "already writes object"):
            candidate.delta("counter", 1)

    def test_conflict_without_regenerator_preserves_concurrent_value(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "initial", kind="row")
        stale = manager.begin("stale")
        stale.add_candidate("stale", quality=1, gen_cost=0).overwrite("row", "stale")
        fresh = manager.begin("fresh")
        fresh.add_candidate("fresh", quality=1, gen_cost=0).overwrite("row", "fresh")
        self.assertTrue(fresh.commit().committed)
        result = stale.commit()
        self.assertEqual(result.state, TransactionState.ABORTED)
        self.assertEqual(result.action, "regenerate_required")
        self.assertEqual(result.n_regen, 0)
        self.assertEqual(manager.value_of("row"), "fresh")
        finish = manager.traces()[-1]["events"][-1]
        self.assertEqual(finish["detail"]["conflict_object_ids"], ["row"])

    def test_regenerator_receives_latest_snapshot(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        stale = manager.begin("stale")
        stale.add_candidate("stale", quality=1, gen_cost=0).overwrite("row", "1")
        fresh = manager.begin("fresh")
        fresh.add_candidate("fresh", quality=1, gen_cost=0).overwrite("row", "10")
        fresh.commit()

        def regenerate(txn):
            current = txn.read("row")
            txn.record_model_call(model="test", latency_s=0.25)
            txn.add_candidate("regenerated", quality=1, gen_cost=0.25).overwrite(
                "row", str(int(current.value) + 1)
            )

        result = stale.commit(regenerator=regenerate)
        self.assertTrue(result.committed)
        self.assertEqual(result.action, "regenerate")
        self.assertEqual(result.winner_branch_id, "regenerated")
        self.assertEqual(result.n_regen, 1)
        self.assertEqual(result.model_latency_s, 0.25)
        self.assertEqual(manager.value_of("row"), "11")

    def test_stale_read_set_requires_regeneration(self):
        manager = AgentTransactionManager()
        manager.register_object("guard", "old", kind="row")
        manager.register_object("output", "", kind="row")

        stale = manager.begin("stale")
        seen = stale.read("guard").value
        stale.add_candidate("stale", quality=1, gen_cost=0).overwrite(
            "output", f"saw:{seen}"
        )

        fresh = manager.begin("fresh")
        fresh.add_candidate("fresh", quality=1, gen_cost=0).overwrite("guard", "new")
        self.assertTrue(fresh.commit().committed)

        result = stale.commit()
        self.assertEqual(result.state, TransactionState.ABORTED)
        self.assertEqual(result.action, "regenerate_required")
        self.assertEqual(manager.value_of("guard"), "new")
        self.assertEqual(manager.value_of("output"), "")
        finish = manager.traces()[-1]["events"][-1]
        self.assertEqual(finish["detail"]["conflict_object_ids"], ["guard"])

    def test_regenerated_read_set_uses_latest_snapshot(self):
        manager = AgentTransactionManager()
        manager.register_object("guard", "old", kind="row")
        manager.register_object("output", "", kind="row")

        stale = manager.begin("stale")
        seen = stale.read("guard").value
        stale.add_candidate("stale", quality=1, gen_cost=0).overwrite(
            "output", f"saw:{seen}"
        )

        fresh = manager.begin("fresh")
        fresh.add_candidate("fresh", quality=1, gen_cost=0).overwrite("guard", "new")
        self.assertTrue(fresh.commit().committed)

        def regenerate(txn):
            current = txn.read("guard")
            txn.add_candidate("regenerated", quality=1, gen_cost=0).overwrite(
                "output", f"saw:{current.value}"
            )

        result = stale.commit(regenerator=regenerate)
        self.assertTrue(result.committed)
        self.assertEqual(result.action, "regenerate")
        self.assertEqual(result.n_regen, 1)
        self.assertEqual(manager.value_of("output"), "saw:new")

    def test_transaction_commit_is_single_use_under_concurrency(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", 0, kind="counter")
        txn = manager.begin("same")
        txn.add_candidate("same", quality=1, gen_cost=0).delta("counter", 1)

        original_commit = manager._commit_locked
        entered = threading.Event()
        release = threading.Event()

        def delayed_commit(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return original_commit(*args, **kwargs)

        manager._commit_locked = delayed_commit
        results = []
        results_lock = threading.Lock()

        def commit_once():
            try:
                result = txn.commit().committed
            except Exception as exc:  # noqa: BLE001 - test captures the public error type.
                result = type(exc).__name__
            with results_lock:
                results.append(result)

        try:
            first = threading.Thread(target=commit_once)
            second = threading.Thread(target=commit_once)
            first.start()
            self.assertTrue(entered.wait(5))
            second.start()
            release.set()
            first.join(5)
            second.join(5)
        finally:
            release.set()
            manager._commit_locked = original_commit

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(manager.value_of("counter"), "1")
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count("RuntimeError"), 1)

    def test_multi_object_stock_rejection_is_atomic(self):
        manager = AgentTransactionManager()
        manager.register_object("available", 1, kind="counter")
        manager.register_object("sold-out", 0, kind="counter")
        txn = manager.begin("atomic-stock")
        txn.add_candidate("both", quality=1, gen_cost=0).delta(
            "available", -1, constrained=True
        ).delta("sold-out", -1, constrained=True)
        result = txn.commit()
        self.assertEqual(result.state, TransactionState.REJECTED)
        self.assertEqual(manager.value_of("available"), "1")
        self.assertEqual(manager.value_of("sold-out"), "0")

    def test_sold_out_candidate_reselects_available_candidate(self):
        manager = AgentTransactionManager()
        manager.register_object("sold-out", 0, kind="counter")
        manager.register_object("available", 1, kind="counter")
        txn = manager.begin("fallback")
        txn.add_candidate("sold-out", quality=2, gen_cost=0).delta(
            "sold-out", -1, constrained=True
        )
        txn.add_candidate("available", quality=1, gen_cost=0).delta(
            "available", -1, constrained=True
        )
        result = txn.commit()
        self.assertTrue(result.committed)
        self.assertEqual(result.action, "reselect")
        self.assertEqual(result.winner_branch_id, "available")
        self.assertEqual(manager.value_of("available"), "0")

    def test_commutative_delta_rebases_without_regeneration(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", 0, kind="counter")
        older = manager.begin("older")
        older.add_candidate("older", quality=1, gen_cost=0).delta("counter", 1)
        newer = manager.begin("newer")
        newer.add_candidate("newer", quality=1, gen_cost=0).delta("counter", 1)
        newer.commit()
        result = older.commit()
        self.assertTrue(result.committed)
        self.assertEqual(result.action, "merge")
        self.assertEqual(result.n_regen, 0)
        self.assertEqual(manager.value_of("counter"), "2")


class DeepSeekClientTests(unittest.TestCase):
    def test_retry_latency_covers_the_whole_operation(self):
        payload = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        throttled = HTTPError(
            deepseek_client.API_URL,
            429,
            "busy",
            hdrs=None,
            fp=io.BytesIO(b"busy"),
        )
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
             mock.patch.object(
                 deepseek_client.urllib.request,
                 "urlopen",
                 side_effect=[throttled, Response()],
             ), \
             mock.patch.object(deepseek_client.time, "sleep") as sleep, \
             mock.patch.object(
                 deepseek_client.time, "perf_counter", side_effect=[10.0, 15.0]
             ):
            result = deepseek_client.chat(
                [{"role": "user", "content": "hello"}], retries=2
            )
        self.assertEqual(result["latency_s"], 5.0)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()

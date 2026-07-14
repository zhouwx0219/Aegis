import json
import random
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

from agent.native import load_cast_core
from agent.runtime import (
    AgentTransactionManager,
    CompiledPhasePolicy,
    CompiledPolicyEntry,
    LockAction,
    LockClass,
    TransactionContext,
    TransactionStatus,
)


cc = load_cast_core()


class SimulatedCrash(RuntimeError):
    pass


@dataclass(frozen=True)
class HistoryTxn:
    tid: str
    reads: dict[str, int]
    writes: dict[str, int]


def assert_serializable(testcase, initial_versions, committed, final_versions):
    """Check version dependencies and return one valid serial order."""
    nodes = {txn.tid for txn in committed}
    edges = {tid: set() for tid in nodes}
    writers = {}
    for txn in committed:
        for key, version in txn.writes.items():
            testcase.assertNotIn((key, version), writers)
            writers[(key, version)] = txn.tid

    for key, initial in initial_versions.items():
        ordered = sorted(
            (version, tid) for (write_key, version), tid in writers.items() if write_key == key
        )
        previous_tid = None
        expected = initial + 1
        for version, tid in ordered:
            testcase.assertEqual(expected, version)
            if previous_tid is not None:
                edges[previous_tid].add(tid)
            previous_tid = tid
            expected += 1
        testcase.assertEqual(expected - 1, final_versions[key])

    for txn in committed:
        for key, version in txn.reads.items():
            source = writers.get((key, version))
            if source is not None and source != txn.tid:
                edges[source].add(txn.tid)
            for (write_key, write_version), writer in writers.items():
                if write_key == key and write_version > version and writer != txn.tid:
                    edges[txn.tid].add(writer)

    indegree = {tid: 0 for tid in nodes}
    for source in nodes:
        edges[source].discard(source)
        for target in edges[source]:
            indegree[target] += 1
    ready = sorted(tid for tid, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        tid = ready.pop(0)
        order.append(tid)
        for target in sorted(edges[tid]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    testcase.assertEqual(len(nodes), len(order), f"serialization cycle: {edges}")
    return order


class StoreProxy:
    def __init__(self, inner):
        self.inner = inner

    @property
    def backend_name(self):
        return self.inner.backend_name

    def __getattr__(self, name):
        return getattr(self.inner, name)


class PartialInstallStore(StoreProxy):
    def __init__(self, inner):
        super().__init__(inner)
        self.fail_once = True

    def batch_put_if_version(self, checks, writes):
        if self.fail_once and len(writes) > 1:
            self.fail_once = False
            expected = dict(checks)[writes[0][0]]
            if not self.inner.put_if_version(writes[0][0], expected, writes[0][1]):
                return False
            raise SimulatedCrash("partial install")
        return self.inner.batch_put_if_version(checks, writes)


class SecondVersionPauseStore(StoreProxy):
    def __init__(self, inner):
        super().__init__(inner)
        self.calls = 0
        self.second_call = threading.Event()
        self.resume = threading.Event()

    def get_version(self, key):
        self.calls += 1
        if self.calls == 2:
            self.second_call.set()
            self.resume.wait(2)
        return self.inner.get_version(key)


class PaperATCCCorrectnessStressTests(unittest.TestCase):
    def full_lock_policy(self):
        return CompiledPhasePolicy(
            [CompiledPolicyEntry(phase="refine", action=15)], generation=1
        )

    def test_random_histories_are_conflict_serializable(self):
        for history_seed in range(40):
            rng = random.Random(700_000 + history_seed)
            manager = AgentTransactionManager(paper_policy=self.full_lock_policy(), record_traces=False)
            keys = [f"k{i}" for i in range(4)]
            for key in keys:
                manager.register_object(key, "0", kind="row")
            initial = {key: manager.store.get_version(key) for key in keys}
            barrier = threading.Barrier(9)
            records = []
            errors = []
            mutex = threading.Lock()

            def worker(index):
                local = random.Random(rng.randrange(1 << 30))
                read_keys = local.sample(keys, 2)
                write_key = local.choice(read_keys)
                txn = manager.begin(
                    f"h{history_seed}-t{index}",
                    {"paper_atcc": True},
                    snapshot_object_ids=read_keys,
                )
                prepared = False
                try:
                    values = [int(txn.read(key).value) for key in read_keys]
                    txn.write(write_key, str(sum(values) + index + 1))
                    prepared = True
                except Exception as exc:
                    if txn.context.status not in {TransactionStatus.ABORTED, TransactionStatus.ABORTING}:
                        with mutex:
                            errors.append(exc)
                finally:
                    barrier.wait()
                if not prepared or txn.context.status != TransactionStatus.ACTIVE:
                    return
                try:
                    txn.enter_phase("refine")
                    result = txn.commit("paper-atcc")
                    if result.committed:
                        record = HistoryTxn(
                            txn.context.tid,
                            {key: txn.read_set[key].version for key in read_keys},
                            {write_key: txn.write_set[write_key].base_version + 1},
                        )
                        with mutex:
                            records.append(record)
                except Exception as exc:
                    if txn.context.status not in {TransactionStatus.ABORTED, TransactionStatus.ABORTING}:
                        with mutex:
                            errors.append(exc)

            threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(6)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertFalse(errors)
            final = {key: manager.store.get_version(key) for key in keys}
            assert_serializable(self, initial, records, final)

    def test_dirty_reads_are_not_visible(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "old", kind="row")
        writer = manager.begin("writer")
        writer.write("row", "uncommitted")
        observer = manager.begin("observer")
        self.assertEqual("old", observer.read("row").value)
        self.assertEqual("old", manager.store.get("row").value)
        writer.abort("test abort")
        self.assertEqual("old", manager.store.get("row").value)

    def test_lock_hold_time_is_accounted_once(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("lock-hold")
        manager.atcc_locks.wlock("row", txn.context)
        time.sleep(0.003)
        held = txn.context.current_lock_hold_ms()
        self.assertGreaterEqual(held, 2.0)
        manager.atcc_locks.release_all(txn.context)
        released = txn.context.current_lock_hold_ms()
        manager.atcc_locks.release_all(txn.context)
        self.assertGreaterEqual(released, held)
        self.assertAlmostEqual(released, txn.context.current_lock_hold_ms(), delta=0.2)

    def test_repeated_lock_acquisition_is_idempotent(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("idempotent-lock")
        manager.atcc_locks.wlock("row", txn.context)
        manager.atcc_locks.wlock("row", txn.context)
        manager.atcc_locks.validate_and_rlock(
            "row",
            txn.context,
            manager.store.get_version("row"),
            lambda: manager.store.get_version("row"),
        )
        self.assertEqual({"row"}, txn.context.held_write_locks)
        self.assertEqual(0, manager.atcc_locks.global_waiter_count())
        manager.atcc_locks.release_all(txn.context)

    def test_background_externality_is_attributed_to_actual_lock_owner(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        agent = manager.begin("agent-owner", {"paper_atcc": True})
        manager.atcc_locks.wlock("row", agent.context)
        self.assertTrue(manager.atcc_locks.begin_committing(agent.context))
        agent.context.transition(TransactionStatus.COMMITTED)
        background = manager.begin("background", {"paper_atcc_backend": True})
        acquired = threading.Event()

        def wait_for_agent():
            manager.atcc_locks.wlock("row", background.context, timeout_s=1)
            acquired.set()

        thread = threading.Thread(target=wait_for_agent)
        thread.start()
        time.sleep(0.01)
        manager.atcc_locks.release_all(agent.context)
        thread.join(2)
        self.assertTrue(acquired.is_set())
        self.assertGreater(agent.context.background_blocked_ms_caused, 0.0)
        manager.atcc_locks.release_all(background.context)

        background_owner = manager.begin("background-owner", {"paper_atcc_backend": True})
        manager.atcc_locks.wlock("row", background_owner.context)
        agent_requester = manager.begin("agent-requester", {"paper_atcc": True})
        manager.atcc_locks.update_priority(agent_requester.context, 100)
        manager.atcc_locks.wlock("row", agent_requester.context)
        self.assertEqual(1, agent_requester.context.background_aborts_caused)
        self.assertEqual(TransactionStatus.ABORTED, background_owner.context.status)
        manager.atcc_locks.release_all(agent_requester.context)

    def test_coordinated_backend_yields_quickly_to_agent_write_lock(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        agent = manager.begin("agent-owner", {"paper_atcc": True})
        manager.atcc_locks.wlock("row", agent.context)

        background = manager.begin(
            "background-commit",
            {"paper_atcc_backend": True},
            snapshot_object_ids=("row",),
        )
        background.read("row")
        background.write("row", "1")
        started = time.perf_counter()
        result = background.commit("occ")
        elapsed = time.perf_counter() - started

        self.assertFalse(result.committed)
        self.assertLess(elapsed, 0.5)
        self.assertEqual("0", manager.store.get("row").value)
        manager.atcc_locks.release_all(agent.context)
        diagnostics = manager.atcc_locks.snapshot_diagnostics()
        self.assertGreater(diagnostics["background_publish_fallbacks"], 0)
        self.assertGreater(
            diagnostics["background_publish_fallback_active_writer"], 0
        )
        self.assertEqual(1, diagnostics["background_lock_wait_events"])
        self.assertGreater(diagnostics["background_lock_wait_ms"], 0.0)
        self.assertEqual(0, diagnostics.get("agent_lock_wait_events", 0))

    def test_uncontended_backend_commit_publishes_without_materialized_locks(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        background = manager.begin(
            "background-fast-publish",
            {"paper_atcc_backend": True},
            snapshot_object_ids=("row",),
        )
        background.read("row")
        background.write("row", "1")

        result = background.commit("occ")

        self.assertTrue(result.committed)
        self.assertEqual("1", manager.store.get("row").value)
        self.assertEqual("", manager.atcc_locks.snapshot("row")["writer"])
        diagnostics = manager.atcc_locks.snapshot_diagnostics()
        self.assertEqual(1, diagnostics["background_fast_publishes"])
        self.assertEqual(0, diagnostics.get("write_lock_acquires", 0))

    def test_background_publish_bypasses_reader_and_preserves_pinned_version(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        agent = manager.begin("agent-reader", {"paper_atcc": True})
        self.assertEqual("0", agent.read("row").value)
        manager.transition_atcc_action(agent, LockAction(LockClass.COLD_READ))

        background = manager.begin(
            "background-writer",
            {"paper_atcc_backend": True},
        )
        background.write("row", "1")
        self.assertTrue(background.commit("occ").committed)

        self.assertEqual("0", agent.read("row").value)
        fresh = manager.begin("fresh-reader")
        self.assertEqual("1", fresh.read("row").value)
        self.assertTrue(fresh.commit("occ").committed)
        self.assertTrue(agent.commit("paper-atcc").committed)
        diagnostics = manager.atcc_locks.snapshot_diagnostics()
        self.assertEqual(1, diagnostics["background_fast_publishes"])
        self.assertEqual(0, diagnostics["background_publish_fallbacks"])

    def test_multi_object_publish_is_atomic_across_pinned_agent_snapshot(self):
        manager = AgentTransactionManager()
        manager.register_object("a", "old-a", kind="row")
        manager.register_object("b", "old-b", kind="row")
        agent = manager.begin("pinned-agent", {"paper_atcc": True})
        self.assertEqual("old-a", agent.read("a").value)
        manager.transition_atcc_action(agent, LockAction(LockClass.COLD_READ))

        background = manager.begin(
            "multi-publish",
            {"paper_atcc_backend": True},
        )
        background.write("a", "new-a")
        background.write("b", "new-b")
        self.assertTrue(background.commit("occ").committed)

        self.assertEqual("old-b", agent.read("b").value)
        fresh = manager.begin("fresh-snapshot")
        self.assertEqual(
            ("new-a", "new-b"),
            (fresh.read("a").value, fresh.read("b").value),
        )
        self.assertTrue(fresh.commit("occ").committed)
        self.assertTrue(agent.commit("paper-atcc").committed)
        versions = manager.version_manager.snapshot_diagnostics()
        self.assertGreaterEqual(versions["atomic_publishes"], 1)
        self.assertEqual(0, versions["private_transactions"])

    def test_disjoint_private_publishes_overlap_and_snapshot_waits_for_boundary(self):
        manager = AgentTransactionManager()
        manager.register_object("a", "old-a", kind="row")
        manager.register_object("b", "old-b", kind="row")
        version_manager = manager.version_manager
        store = manager.store
        entered = threading.Barrier(2)
        release = threading.Event()
        installed = {"a": threading.Event(), "b": threading.Event()}
        results = {}
        errors = []

        def publisher(object_id, value):
            expected = store.get_version(object_id)

            def install():
                entered.wait(2)
                ok = store.put_if_version(object_id, expected, value)
                installed[object_id].set()
                release.wait(2)
                return ok

            try:
                results[object_id] = version_manager.atomic_publish(
                    f"publish-{object_id}",
                    ((object_id, value),),
                    install,
                    background=True,
                    published_version=store.get_version,
                )
            except BaseException as exc:  # pragma: no cover - diagnostic path
                errors.append(exc)

        threads = [
            threading.Thread(target=publisher, args=("a", "new-a")),
            threading.Thread(target=publisher, args=("b", "new-b")),
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(installed["a"].wait(2))
        self.assertTrue(installed["b"].wait(2))

        snapshot_result = {}
        snapshot_done = threading.Event()

        def snapshot_reader():
            snapshot_result["value"] = version_manager.snapshot_current(("a", "b"))
            snapshot_done.set()

        reader = threading.Thread(target=snapshot_reader)
        reader.start()
        time.sleep(0.03)
        self.assertFalse(snapshot_done.is_set())
        release.set()

        for thread in threads:
            thread.join(2)
        reader.join(2)
        self.assertFalse(errors)
        self.assertEqual({"a": True, "b": True}, results)
        self.assertTrue(snapshot_done.is_set())
        _, snapshot = snapshot_result["value"]
        self.assertEqual("new-a", snapshot["a"].value)
        self.assertEqual("new-b", snapshot["b"].value)

    def test_failed_background_private_version_is_discarded_with_reason(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        first = manager.begin("first", {"paper_atcc_backend": True})
        second = manager.begin("second", {"paper_atcc_backend": True})
        first.read("row")
        second.read("row")
        first.write("row", "1")
        second.write("row", "2")
        self.assertTrue(first.commit("occ").committed)

        result = second.commit("occ")

        self.assertFalse(result.committed)
        versions = manager.version_manager.snapshot_diagnostics()
        self.assertEqual(0, versions["private_transactions"])
        self.assertGreaterEqual(versions["private_discards"], 1)
        diagnostics = manager.atcc_locks.snapshot_diagnostics()
        self.assertGreaterEqual(
            diagnostics["background_publish_fallback_version_mismatch"], 1
        )

    def test_background_blind_write_rebases_under_publish_latch(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        first = manager.begin("first-blind", {"paper_atcc_backend": True})
        second = manager.begin("second-blind", {"paper_atcc_backend": True})
        first.write("row", "1")
        second.write("row", "2")
        self.assertTrue(first.commit("occ").committed)

        result = second.commit("occ")

        self.assertTrue(result.committed)
        self.assertEqual("2", manager.store.get("row").value)

    def test_fast_background_publish_excludes_agent_lock_acquisition(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        background = manager.begin("background-publish", {"paper_atcc_backend": True})
        agent = manager.begin("agent-locker", {"paper_atcc": True})
        publish_entered = threading.Event()
        allow_publish = threading.Event()
        agent_acquired = threading.Event()
        outcome = []

        def publish():
            publish_entered.set()
            allow_publish.wait(1)
            background.context.transition(TransactionStatus.COMMITTED)
            return "published"

        publisher = threading.Thread(
            target=lambda: outcome.append(
                manager.atcc_locks.try_uncontended_background_publish(
                    ("row",), background.context, publish
                )
            )
        )
        publisher.start()
        self.assertTrue(publish_entered.wait(1))

        locker = threading.Thread(
            target=lambda: (
                manager.atcc_locks.wlock("row", agent.context, timeout_s=1),
                agent_acquired.set(),
            )
        )
        locker.start()
        self.assertFalse(agent_acquired.wait(0.02))
        allow_publish.set()
        publisher.join(1)
        locker.join(1)

        self.assertEqual([(True, "published")], outcome)
        self.assertTrue(agent_acquired.is_set())
        manager.atcc_locks.release_all(agent.context)

    def test_disjoint_background_publish_intents_run_concurrently(self):
        manager = AgentTransactionManager()
        manager.register_object("a", "0", kind="row")
        manager.register_object("b", "0", kind="row")
        first = manager.begin("background-a", {"paper_atcc_backend": True})
        second = manager.begin("background-b", {"paper_atcc_backend": True})
        first_entered = threading.Event()
        second_entered = threading.Event()
        allow_finish = threading.Event()
        outcomes = []

        def run_publish(txn, object_id, entered):
            def publish():
                entered.set()
                allow_finish.wait(1)
                txn.context.transition(TransactionStatus.COMMITTED)
                return object_id

            outcomes.append(
                manager.atcc_locks.try_uncontended_background_publish(
                    (object_id,), txn.context, publish
                )
            )

        first_thread = threading.Thread(target=run_publish, args=(first, "a", first_entered))
        second_thread = threading.Thread(target=run_publish, args=(second, "b", second_entered))
        first_thread.start()
        self.assertTrue(first_entered.wait(1))
        second_thread.start()
        self.assertTrue(second_entered.wait(0.2))
        allow_finish.set()
        first_thread.join(1)
        second_thread.join(1)

        self.assertCountEqual([(True, "a"), (True, "b")], outcomes)
        self.assertEqual("", manager.atcc_locks.snapshot("a")["publisher"])
        self.assertEqual("", manager.atcc_locks.snapshot("b")["publisher"])

    def test_same_object_background_publish_intents_enter_fast_path_concurrently(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        first = manager.begin("background-1", {"paper_atcc_backend": True})
        second = manager.begin("background-2", {"paper_atcc_backend": True})
        first_entered = threading.Event()
        second_entered = threading.Event()
        allow_finish = threading.Event()
        outcomes = []

        def publish(txn, entered):
            def callback():
                entered.set()
                allow_finish.wait(1)
                txn.context.transition(TransactionStatus.COMMITTED)
                return txn.context.tid

            outcomes.append(
                manager.atcc_locks.try_uncontended_background_publish(
                    ("row",), txn.context, callback
                )
            )

        threads = [
            threading.Thread(target=publish, args=(first, first_entered)),
            threading.Thread(target=publish, args=(second, second_entered)),
        ]
        threads[0].start()
        self.assertTrue(first_entered.wait(1))
        threads[1].start()
        self.assertTrue(second_entered.wait(0.2))
        allow_finish.set()
        for thread in threads:
            thread.join(1)

        self.assertEqual(2, len(outcomes))
        self.assertTrue(all(used for used, _result in outcomes))
        self.assertEqual("", manager.atcc_locks.snapshot("row")["publisher"])

    def test_lost_update_is_rejected(self):
        manager = AgentTransactionManager(paper_policy=self.full_lock_policy())
        manager.register_object("counter", "0", kind="counter")
        barrier = threading.Barrier(17)
        committed = []

        def increment(index):
            txn = manager.begin(f"inc-{index}", {"paper_atcc": True})
            value = int(txn.read("counter").value)
            txn.write("counter", str(value + 1))
            barrier.wait()
            try:
                txn.enter_phase("refine")
                result = txn.commit("paper-atcc")
                if result.committed:
                    committed.append(index)
            except Exception:
                pass

        threads = [threading.Thread(target=increment, args=(index,)) for index in range(16)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(3)
        self.assertEqual(1, len(committed))
        self.assertEqual("1", manager.store.get("counter").value)

    def test_first_blind_write_refreshes_snapshot_after_wlock_wait(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        owner = manager.begin("owner", {"paper_atcc": True})
        manager.atcc_locks.wlock("row", owner.context)
        manager.atcc_locks.update_priority(owner.context, 1_000_000)
        waiter = manager.begin("waiter", {"paper_atcc": True})
        finished = threading.Event()
        outcome = []

        def write_after_wait():
            manager.atcc_locks.wlock("row", waiter.context, timeout_s=1)
            waiter.refresh_unobserved_locked_snapshot("row")
            waiter.write("row", "2")
            outcome.append(waiter.commit("paper-atcc").committed)
            finished.set()

        thread = threading.Thread(target=write_after_wait)
        thread.start()
        time.sleep(0.01)
        owner.write("row", "1")
        self.assertTrue(owner.commit("paper-atcc").committed)
        thread.join(2)
        self.assertTrue(finished.is_set())
        self.assertEqual([True], outcome)
        self.assertEqual("2", manager.store.get("row").value)

    def test_committed_readers_never_observe_partial_visibility(self):
        manager = AgentTransactionManager(record_traces=False)
        manager.register_object("a", "0", kind="row")
        manager.register_object("b", "0", kind="row")
        stop = threading.Event()
        observed = []

        def writer():
            for value in range(1, 150):
                while True:
                    txn = manager.begin(f"writer-{value}")
                    txn.write("a", str(value))
                    txn.write("b", str(value))
                    if txn.commit("occ").committed:
                        break
            stop.set()

        def reader(index):
            sequence = 0
            while not stop.is_set():
                txn = manager.begin(f"reader-{index}-{sequence}")
                pair = (txn.read("a").value, txn.read("b").value)
                if txn.commit("occ").committed:
                    observed.append(pair)
                sequence += 1

        writer_thread = threading.Thread(target=writer)
        readers = [threading.Thread(target=reader, args=(index,)) for index in range(6)]
        for thread in readers + [writer_thread]:
            thread.start()
        writer_thread.join(5)
        stop.set()
        for thread in readers:
            thread.join(2)
        self.assertTrue(observed)
        self.assertTrue(all(left == right for left, right in observed))

    def test_action_change_rejects_pinned_version_after_agent_update(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        agent = manager.begin("agent")
        agent.read("row")
        writer = manager.begin("concurrent-writer")
        writer.write("row", "1")
        self.assertTrue(writer.commit("occ").committed)

        with self.assertRaises(Exception):
            manager.transition_atcc_action(agent, LockAction(LockClass.COLD_READ))

        self.assertEqual(TransactionStatus.ABORTED, agent.context.status)

    def test_dynamic_priority_has_no_deadlock_or_finite_starvation(self):
        manager = AgentTransactionManager(record_traces=False)
        manager.register_object("a", "0", kind="row")
        manager.register_object("b", "0", kind="row")
        barrier = threading.Barrier(3)
        finished = []

        def reverse_locker(name, first, second, priority):
            txn = manager.begin(name)
            manager.atcc_locks.update_priority(txn.context, priority)
            try:
                manager.atcc_locks.wlock(first, txn.context, timeout_s=1)
                barrier.wait()
                manager.atcc_locks.wlock(second, txn.context, timeout_s=1)
                finished.append(name)
            except Exception:
                finished.append(name + ":aborted")
            finally:
                manager.atcc_locks.release_all(txn.context)

        threads = [
            threading.Thread(target=reverse_locker, args=("low", "a", "b", 1)),
            threading.Thread(target=reverse_locker, args=("high", "b", "a", 10)),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(2, len(finished))
        self.assertIn("high", finished)

        owner = manager.begin("committing-owner")
        manager.atcc_locks.wlock("a", owner.context)
        owner.context.transition(TransactionStatus.COMMITTING)
        manager.atcc_locks.enter_committing(owner.context)
        order = []
        priority_epochs = []

        def waiter(priority):
            txn = manager.begin(f"waiter-{priority}")
            manager.atcc_locks.update_priority(txn.context, priority)
            manager.atcc_locks.wlock("a", txn.context, timeout_s=2)
            order.append(priority)
            priority_epochs.append(txn.context.priority_epoch)
            manager.atcc_locks.release_all(txn.context)

        waiters = [threading.Thread(target=waiter, args=(priority,)) for priority in range(1, 7)]
        for thread in waiters:
            thread.start()
        time.sleep(0.25)
        manager.atcc_locks.release_all(owner.context)
        for thread in waiters:
            thread.join(3)
        self.assertEqual(6, len(order))
        self.assertTrue(all(epoch >= 1 for epoch in priority_epochs))
        self.assertEqual(0, manager.atcc_locks.global_waiter_count())

    def test_undo_fault_boundaries_and_recovery_idempotence(self):
        for stage, expected in (
            ("after_undo_flush_before_install", ("old-a", "old-b", 1)),
            ("after_install_before_publish", ("old-a", "old-b", 1)),
            ("after_publish", ("new-a", "new-b", 0)),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "undo.jsonl"
                manager = AgentTransactionManager(undo_log_path=path)
                manager.register_object("a", "old-a", kind="row")
                manager.register_object("b", "old-b", kind="row")
                txn = manager.begin("fault", {"paper_atcc": True})
                txn.write("a", "new-a")
                txn.write("b", "new-b")

                def inject(current_stage, _txn):
                    if current_stage == stage:
                        raise SimulatedCrash(stage)

                manager._commit_fault_injector = inject
                with self.assertRaises(SimulatedCrash):
                    txn.commit("paper-atcc")
                restarted = AgentTransactionManager(store=manager.store, undo_log_path=path)
                recovered = restarted.recover()
                self.assertEqual(expected[2], len(recovered))
                self.assertEqual(expected[:2], (restarted.value_of("a"), restarted.value_of("b")))
                self.assertEqual([], restarted.recover())

    def test_partial_install_is_fully_undone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inner = cc.Dbx1000VersionedKVStore()
            store = PartialInstallStore(inner)
            path = Path(tmpdir) / "undo.jsonl"
            manager = AgentTransactionManager(store=store, undo_log_path=path)
            manager.register_object("a", "old-a", kind="row")
            manager.register_object("b", "old-b", kind="row")
            txn = manager.begin("partial", {"paper_atcc": True})
            txn.write("a", "new-a")
            txn.write("b", "new-b")
            with self.assertRaises(SimulatedCrash):
                txn.commit("paper-atcc")
            restarted = AgentTransactionManager(store=store, undo_log_path=path)
            self.assertEqual([txn.context.tid], restarted.recover())
            self.assertEqual(("old-a", "old-b"), (store.get("a").value, store.get("b").value))

    def test_undo_log_rejects_truncation_and_lsn_damage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "undo.jsonl"
            manager = AgentTransactionManager(undo_log_path=path)
            manager.undo_log.begin("txn")
            original = path.read_text(encoding="utf-8")
            path.write_text(original[:-7], encoding="utf-8")
            with self.assertRaises(RuntimeError):
                AgentTransactionManager(undo_log_path=path)

            path.write_text(original, encoding="utf-8")
            row = json.loads(original)
            row["lsn"] = 3
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                AgentTransactionManager(undo_log_path=path)


if __name__ == "__main__":
    unittest.main()

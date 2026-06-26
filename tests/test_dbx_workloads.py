import json
import dataclasses
import threading
import time
import unittest

import cast_core as cc

from agent.evaluation import run_strategy_matrix
from agent.runtime import (
    AdaptivePolicyRule,
    AdaptivePolicyTable,
    AgentTransactionManager,
    OperationPolicyDecision,
    OperationPolicyProfile,
    OperationPolicyQLearner,
    OperationPolicyTable,
    TransactionState,
)
from agent.runtime.commit_protocol import ObjectLockTimeout
from agent.workloads import (
    TPCCAgentWorkload,
    TPCCFaithfulAgentWorkload,
    TPCCConfig,
    YCSBAgentWorkload,
    YCSBFaithfulAgentWorkload,
    YCSBConfig,
    build_agent_workload,
    execute_task,
    prepare_task_transaction,
    register_workload,
)


class Dbx1000VersionedKVTests(unittest.TestCase):
    def test_dbx1000_backend_handles_bucket_collisions(self):
        store = cc.Dbx1000VersionedKVStore(bucket_count=1)
        self.assertEqual(store.backend_name, "dbx1000")

        for index in range(64):
            store.put(f"key-{index}", f"value-{index}")
        for index in range(64):
            value = store.get(f"key-{index}")
            self.assertTrue(value.exists)
            self.assertEqual(value.value, f"value-{index}")
            self.assertEqual(value.version, 1)

    def test_compare_and_set_and_tombstone_versions_are_monotonic(self):
        store = cc.Dbx1000VersionedKVStore()
        self.assertTrue(store.put_if_version("object", 0, "first"))
        first_version = store.get_version("object")
        self.assertFalse(store.put_if_version("object", 0, "stale"))
        self.assertTrue(store.delete_if_version("object", first_version))

        tombstone = store.get("object")
        self.assertFalse(tombstone.exists)
        self.assertGreater(tombstone.version, first_version)
        self.assertTrue(
            store.put_if_version("object", tombstone.version, "recreated")
        )
        self.assertGreater(store.get_version("object"), tombstone.version)

    def test_configured_row_capacity_is_enforced(self):
        store = cc.Dbx1000VersionedKVStore(
            max_key_bytes=4, max_value_bytes=5, bucket_count=2
        )
        with self.assertRaises((ValueError, RuntimeError)):
            store.put("", "value")
        with self.assertRaises((ValueError, RuntimeError)):
            store.put("12345", "value")
        with self.assertRaises((ValueError, RuntimeError)):
            store.put("key", "123456")


class PluggableConcurrencyControlTests(unittest.TestCase):
    def test_pre_snapshot_lock_is_released_when_empty_transaction_aborts(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", 0, kind="counter")
        transaction = manager.begin("empty", prelock_targets=("counter",))
        trace = transaction.to_trace()
        self.assertIn("counter", trace["prelock_target_wait_s"])

        result = transaction.commit(strategy="2pl-pre")
        self.assertEqual(result.state, TransactionState.ABORTED)

        acquired = threading.Event()

        def acquire_after_abort():
            with manager.object_locks.acquire(("counter",)):
                acquired.set()

        worker = threading.Thread(target=acquire_after_abort)
        worker.start()
        self.assertTrue(acquired.wait(1), "pre-snapshot lock leaked after abort")
        worker.join(1)
        self.assertFalse(worker.is_alive())

    def test_priority_pre_snapshot_lock_wounds_lower_priority_holder(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", 0, kind="counter")
        low = manager.begin(
            "low",
            {"priority": 1},
            prelock_targets=("counter",),
        )
        acquired = threading.Event()
        high_holder = {}

        def acquire_high_priority():
            high_holder["txn"] = manager.begin(
                "high",
                {"priority": 10},
                prelock_targets=("counter",),
            )
            acquired.set()

        worker = threading.Thread(target=acquire_high_priority)
        worker.start()
        self.assertTrue(acquired.wait(1), "high-priority lock request did not wound")
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(low.state, TransactionState.ABORTED)
        self.assertEqual(low.result.reason, "priority_wound")
        self.assertEqual(low.commit(strategy="occ").reason, "priority_wound")
        high_holder["txn"].abort("done")

    def test_object_lock_grants_highest_priority_waiter_first(self):
        manager = AgentTransactionManager(object_lock_queue_policy="priority")
        manager.register_object("counter", 0, kind="counter")
        holder = manager.object_locks.acquire_lease(("counter",), priority=1)
        order = []
        low_acquired = threading.Event()
        high_acquired = threading.Event()
        release_low = threading.Event()
        release_high = threading.Event()

        def acquire_waiter(
            name: str,
            priority: int,
            acquired: threading.Event,
            release: threading.Event,
        ) -> None:
            with manager.object_locks.acquire(("counter",), priority=priority):
                order.append(name)
                acquired.set()
                release.wait(1)

        low_worker = threading.Thread(
            target=acquire_waiter,
            args=("low", 1, low_acquired, release_low),
        )
        high_worker = threading.Thread(
            target=acquire_waiter,
            args=("high", 10, high_acquired, release_high),
        )
        low_worker.start()
        time.sleep(0.02)
        high_worker.start()
        time.sleep(0.02)

        holder.release()
        self.assertTrue(high_acquired.wait(1), "high-priority waiter was not first")
        self.assertEqual(order[:1], ["high"])
        release_high.set()
        self.assertTrue(low_acquired.wait(1), "low-priority waiter never acquired")
        release_low.set()
        low_worker.join(1)
        high_worker.join(1)
        self.assertFalse(low_worker.is_alive())
        self.assertFalse(high_worker.is_alive())

    def test_object_lock_lease_records_queue_depth(self):
        manager = AgentTransactionManager(object_lock_queue_policy="priority")
        manager.register_object("counter", 0, kind="counter")
        holder = manager.object_locks.acquire_lease(("counter",), priority=1)
        waiting = threading.Event()
        release_low = threading.Event()

        def low_waiter() -> None:
            waiting.set()
            with manager.object_locks.acquire(("counter",), priority=1):
                release_low.wait(1)

        low_worker = threading.Thread(target=low_waiter)
        low_worker.start()
        self.assertTrue(waiting.wait(1))
        time.sleep(0.02)

        high_holder = {}

        def high_waiter() -> None:
            high_holder["lease"] = manager.object_locks.acquire_lease(
                ("counter",),
                priority=10,
            )

        high_worker = threading.Thread(target=high_waiter)
        high_worker.start()
        time.sleep(0.02)
        holder.release()
        high_worker.join(1)
        self.assertFalse(high_worker.is_alive())
        high_lease = high_holder["lease"]
        self.assertGreaterEqual(high_lease.target_queue_depth["counter"], 1)
        self.assertEqual(high_lease.target_owner_priority["counter"], 1)
        high_lease.release()
        release_low.set()
        low_worker.join(1)
        self.assertFalse(low_worker.is_alive())

    def test_committing_lock_owner_is_not_wounded(self):
        manager = AgentTransactionManager(object_lock_queue_policy="priority")
        manager.register_object("counter", 0, kind="counter")
        holder = manager.object_locks.acquire_lease(("counter",), priority=1)
        holder.enter_committing()

        outcome = {}

        def high_priority_waiter() -> None:
            try:
                manager.object_locks.acquire_lease(
                    ("counter",),
                    priority=10,
                    wait_timeout_s=0.03,
                )
            except BaseException as exc:  # noqa: BLE001 - assert public timeout behavior.
                outcome["error"] = exc

        worker = threading.Thread(target=high_priority_waiter)
        worker.start()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(outcome.get("error"), ObjectLockTimeout)
        self.assertFalse(holder.wounded)
        holder.exit_committing()
        holder.release()

    def test_transaction_records_prelock_committing_pressure(self):
        manager = AgentTransactionManager(object_lock_queue_policy="priority")
        manager.register_object("counter", 0, kind="counter")
        txn = manager.begin("commit-pressure", prelock_targets=("counter",))

        txn._enter_prelock_committing()
        txn._exit_prelock_committing()
        trace = txn.to_trace()

        self.assertEqual(trace["prelock_committing_enters"], 1)
        self.assertEqual(trace["prelock_committing_exits"], 1)
        self.assertEqual(trace["prelock_committing_target_count"], 1)
        txn.abort("cleanup")

    def test_bounded_priority_lock_gives_low_priority_waiter_a_turn(self):
        manager = AgentTransactionManager(object_lock_queue_policy="bounded-priority")
        manager.register_object("counter", 0, kind="counter")
        holder = manager.object_locks.acquire_lease(("counter",), priority=0)
        order = []
        acquired = {name: threading.Event() for name in ("low", "h1", "h2", "h3")}
        releases = {name: threading.Event() for name in ("low", "h1", "h2", "h3")}

        def acquire_waiter(name: str, priority: int) -> None:
            with manager.object_locks.acquire(("counter",), priority=priority):
                order.append(name)
                acquired[name].set()
                releases[name].wait(1)

        workers = [
            threading.Thread(target=acquire_waiter, args=("low", 0)),
            threading.Thread(target=acquire_waiter, args=("h1", 10)),
            threading.Thread(target=acquire_waiter, args=("h2", 10)),
            threading.Thread(target=acquire_waiter, args=("h3", 10)),
        ]
        for worker in workers:
            worker.start()
            time.sleep(0.01)

        holder.release()
        self.assertTrue(acquired["h1"].wait(1))
        releases["h1"].set()
        self.assertTrue(acquired["h2"].wait(1))
        releases["h2"].set()
        self.assertTrue(
            acquired["low"].wait(1),
            "bounded-priority did not give low-priority waiter a turn",
        )
        self.assertEqual(order[:3], ["h1", "h2", "low"])
        releases["low"].set()
        self.assertTrue(acquired["h3"].wait(1))
        releases["h3"].set()
        for worker in workers:
            worker.join(1)
            self.assertFalse(worker.is_alive())

    def test_prelock_wait_budget_falls_back_to_optimistic_decision(self):
        manager = AgentTransactionManager(prelock_wait_budget_s=0.01)
        manager.register_object("counter", 0, kind="counter")
        holder = manager.object_locks.acquire_lease(("counter",), priority=10)
        decision = OperationPolicyDecision(
            object_id="counter",
            access_kind="write",
            intent_name="delta",
            policy="pessimistic",
            rule="test-pessimistic",
        )

        try:
            txn = manager.begin(
                "budgeted",
                {"workload": "test"},
                prelock_targets=("counter",),
                operation_policy_decisions=(decision,),
            )
        finally:
            holder.release()

        self.assertEqual(txn.prelocked_targets, ())
        self.assertEqual(
            txn.metadata["prelock_fallback"]["reason"],
            "prelock_wait_budget_exceeded",
        )
        self.assertEqual(
            txn.precomputed_operation_policy_decisions[0].policy,
            "optimistic",
        )
        self.assertEqual(
            txn.precomputed_operation_policy_decisions[0].rule,
            "prelock-budget-fallback-optimistic",
        )
        txn.abort("done")

    def test_prelock_wait_budget_does_not_change_full_2pl_prelock(self):
        manager = AgentTransactionManager(prelock_wait_budget_s=0.01)
        manager.register_object("counter", 0, kind="counter")
        holder = manager.object_locks.acquire_lease(("counter",), priority=10)
        decision = OperationPolicyDecision(
            object_id="counter",
            access_kind="write",
            intent_name="delta",
            policy="pessimistic",
            rule="pre-snapshot-2pl-all-operations",
        )
        acquired = threading.Event()
        txns = {}

        def begin_full_2pl() -> None:
            txns["txn"] = manager.begin(
                "two-pl",
                {"workload": "test"},
                prelock_targets=("counter",),
                operation_policy_decisions=(decision,),
            )
            acquired.set()

        worker = threading.Thread(target=begin_full_2pl)
        worker.start()
        self.assertFalse(
            acquired.wait(0.05),
            "2PL-pre prelock should not be downgraded by adaptive wait budget",
        )
        holder.release()
        self.assertTrue(acquired.wait(1))
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(txns["txn"].prelocked_targets, ("counter",))
        self.assertNotIn("prelock_fallback", txns["txn"].metadata)
        txns["txn"].abort("done")

    def test_prelock_lease_can_yield_and_reacquire(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", 0, kind="counter")
        txn = manager.begin(
            "lease",
            {"workload": "test"},
            prelock_targets=("counter",),
        )
        self.assertEqual(txn.prelocked_targets, ("counter",))

        txn.yield_prelocks_for_planning()
        self.assertEqual(txn.prelocked_targets, ())
        with manager.object_locks.acquire(("counter",)):
            pass

        txn.reacquire_yielded_prelocks()
        self.assertEqual(txn.prelocked_targets, ("counter",))
        event_kinds = [event.kind for event in txn.events]
        self.assertIn("prelock_yield", event_kinds)
        self.assertIn("prelock_reacquire", event_kinds)
        txn.abort("done")

        acquired = threading.Event()

        def acquire_after_abort() -> None:
            with manager.object_locks.acquire(("counter",)):
                acquired.set()

        worker = threading.Thread(target=acquire_after_abort)
        worker.start()
        self.assertTrue(acquired.wait(1), "reacquired prelock leaked after abort")
        worker.join(1)
        self.assertFalse(worker.is_alive())

    def test_object_prelock_wait_budget_falls_back_only_timed_out_object(self):
        manager = AgentTransactionManager(
            prelock_wait_budget_s=0.01,
            prelock_wait_budget_mode="object",
        )
        manager.register_object("cold", 0, kind="counter")
        manager.register_object("hot", 0, kind="counter")
        holder = manager.object_locks.acquire_lease(("hot",), priority=10)
        decisions = (
            OperationPolicyDecision(
                object_id="cold",
                access_kind="write",
                intent_name="delta",
                policy="pessimistic",
                rule="test-pessimistic",
            ),
            OperationPolicyDecision(
                object_id="hot",
                access_kind="write",
                intent_name="delta",
                policy="pessimistic",
                rule="test-pessimistic",
            ),
        )

        try:
            txn = manager.begin(
                "object-budgeted",
                {"workload": "test"},
                prelock_targets=("cold", "hot"),
                operation_policy_decisions=decisions,
            )
        finally:
            holder.release()

        policies = {
            decision.object_id: decision.policy
            for decision in txn.precomputed_operation_policy_decisions
        }
        self.assertEqual(txn.prelocked_targets, ("cold",))
        self.assertEqual(policies["cold"], "pessimistic")
        self.assertEqual(policies["hot"], "optimistic")
        self.assertEqual(
            txn.metadata["prelock_fallback"]["reason"],
            "prelock_object_wait_budget_exceeded",
        )
        self.assertEqual(txn.metadata["prelock_fallback"]["targets"], ["hot"])
        txn.abort("done")

    def test_operation_feedback_uses_object_level_lock_wait(self):
        table = OperationPolicyTable(
            rules=(),
            online_feedback=True,
            fallback_policy="optimistic",
        )
        decisions = (
            OperationPolicyDecision(
                object_id="hot",
                access_kind="write",
                intent_name="overwrite",
                policy="pessimistic",
                rule="test",
                profile_key="hot-profile",
            ),
            OperationPolicyDecision(
                object_id="cold",
                access_kind="write",
                intent_name="overwrite",
                policy="pessimistic",
                rule="test",
                profile_key="cold-profile",
            ),
        )

        table.observe_result(
            decisions,
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=1.0,
            lock_wait_by_object={"hot": 0.2, "cold": 0.0},
            lock_queue_by_object={"hot": 3, "cold": 0},
        )

        self.assertAlmostEqual(
            table.telemetry.stats_for("hot-profile").ewma_lock_wait_s,
            0.2,
        )
        self.assertAlmostEqual(
            table.telemetry.stats_for("cold-profile").ewma_lock_wait_s,
            0.0,
        )
        self.assertAlmostEqual(
            table.telemetry.stats_for("hot-profile").ewma_lock_queue_depth,
            3.0,
        )
        self.assertAlmostEqual(
            table.telemetry.stats_for("cold-profile").ewma_lock_queue_depth,
            0.0,
        )

    def test_builtin_cc_strategy_catalog_exposes_traditional_and_adaptive_names(self):
        manager = AgentTransactionManager()
        strategies = manager.cc_strategies()
        for name in (
            "semantic",
            "cast",
            "occ",
            "mvcc",
            "silo",
            "tictoc",
            "2pl",
            "adaptive",
            "atcc",
            "adaptive-op",
            "adaptive-op-strict",
            "2pl-pre",
        ):
            self.assertIn(name, strategies)

        self.assertTrue(strategies["semantic"]["allows_semantic_rebase"])
        self.assertEqual(strategies["mvcc"]["source"], "DBx1000-inspired")
        self.assertEqual(strategies["dbx1000-silo"]["canonical_name"], "silo")
        self.assertTrue(strategies["2pl"]["requires_object_locks"])
        self.assertEqual(strategies["adaptive"]["selector"], "policy_table")
        self.assertEqual(
            strategies["adaptive"]["policy_table"]["rules"][0]["target_strategy"],
            "semantic",
        )
        self.assertEqual(
            strategies["adaptive-op"]["selector"], "operation_policy_table"
        )
        self.assertEqual(
            strategies["adaptive-op-strict"]["lock_phase"], "pre_snapshot"
        )
        self.assertFalse(
            strategies["adaptive-op-strict"]["allows_semantic_rebase"]
        )
        self.assertEqual(strategies["2pl-pre"]["selector"], "all_operations_pessimistic")

    def test_traditional_2pl_does_not_use_pre_snapshot_oracle_locks(self):
        manager = AgentTransactionManager()
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=4,
                field_count=1,
                requests_per_task=2,
                candidates_per_task=1,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
            ),
        )
        register_workload(manager, workload)
        task = workload.generate_tasks(1, seed=7)[0]

        traditional_2pl = prepare_task_transaction(
            manager,
            task,
            strategy="2pl",
        )
        pre_snapshot_2pl = prepare_task_transaction(
            manager,
            task,
            strategy="2pl-pre",
        )

        self.assertEqual(traditional_2pl.prelocked_targets, ())
        self.assertGreater(len(pre_snapshot_2pl.prelocked_targets), 0)
        self.assertTrue(
            all(
                decision.rule == "pre-snapshot-2pl-all-operations"
                for decision in pre_snapshot_2pl.precomputed_operation_policy_decisions
            )
        )
        traditional_2pl.abort("done")
        pre_snapshot_2pl.abort("done")

    def test_registered_semantic_cc_rebases_a_stale_delta(self):
        manager = AgentTransactionManager()
        self.assertEqual(manager.backend_name, "dbx1000")
        manager.register_cc("semantic-copy", cc.SemanticConcurrencyControl())
        manager.register_object("counter", 0, kind="counter")

        older = manager.begin("older")
        older.add_candidate("older", quality=1, gen_cost=0).delta("counter", 2)
        newer = manager.begin("newer")
        newer.add_candidate("newer", quality=1, gen_cost=0).delta("counter", 3)
        self.assertTrue(newer.commit(strategy="semantic-copy").committed)

        result = older.commit(strategy="semantic-copy")
        self.assertTrue(result.committed)
        self.assertEqual(result.action, "merge")
        self.assertEqual(manager.value_of("counter"), "5")

    def test_occ_plugin_requires_regeneration_for_a_stale_delta(self):
        manager = AgentTransactionManager()
        manager.register_object("counter", 0, kind="counter")
        older = manager.begin("older")
        older.add_candidate("older", quality=1, gen_cost=0).delta("counter", 2)
        newer = manager.begin("newer")
        newer.add_candidate("newer", quality=1, gen_cost=0).delta("counter", 3)
        newer.commit(strategy="occ")

        result = older.commit(strategy="occ")
        self.assertEqual(result.state, TransactionState.ABORTED)
        self.assertEqual(result.action, "regenerate_required")
        self.assertEqual(manager.value_of("counter"), "3")

    def test_adaptive_strategy_selects_strict_or_semantic_module_by_intent(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "old", kind="row")
        manager.register_object("counter", 0, kind="counter")

        strict = manager.begin("strict")
        strict.add_candidate("strict", quality=1, gen_cost=0).overwrite("row", "new")
        self.assertTrue(strict.commit(strategy="adaptive").committed)
        strict_validate = [
            event
            for event in manager.traces()[-1]["events"]
            if event["kind"] == "validate"
        ][0]
        self.assertEqual(strict_validate["detail"]["selected_cc"], "occ")

        older = manager.begin("older")
        older.add_candidate("older", quality=1, gen_cost=0).delta("counter", 2)
        newer = manager.begin("newer")
        newer.add_candidate("newer", quality=1, gen_cost=0).delta("counter", 3)
        self.assertTrue(newer.commit(strategy="adaptive").committed)

        result = older.commit(strategy="adaptive")
        self.assertTrue(result.committed)
        self.assertEqual(result.action, "merge")
        self.assertEqual(manager.value_of("counter"), "5")
        semantic_validate = [
            event
            for event in manager.traces()[-1]["events"]
            if event["kind"] == "validate"
        ][0]
        self.assertEqual(semantic_validate["detail"]["selected_cc"], "semantic")

    def test_custom_adaptive_policy_table_can_select_pessimistic_strategy(self):
        policy = AdaptivePolicyTable(
            rules=(
                AdaptivePolicyRule(
                    name="force-wide-overwrite-to-2pl",
                    target_strategy="2pl",
                    min_writes=1,
                    overwrite_only=True,
                ),
            ),
            fallback_strategy="occ",
            name="test-table",
        )
        manager = AgentTransactionManager(adaptive_policy=policy)
        self.assertEqual(manager.adaptive_policy()["name"], "test-table")
        manager.register_object("row", "old", kind="row")

        txn = manager.begin("custom-policy")
        txn.add_candidate("overwrite", quality=1, gen_cost=0).overwrite("row", "new")
        result = txn.commit(strategy="adaptive")

        self.assertTrue(result.committed)
        validate = [
            event
            for event in manager.traces()[-1]["events"]
            if event["kind"] == "validate"
        ][0]
        self.assertEqual(validate["detail"]["selected_cc"], "2pl")

    def test_new_order_policy_switches_by_write_footprint(self):
        narrow_policy = AdaptivePolicyTable.new_order()
        narrow_manager = AgentTransactionManager(adaptive_policy=narrow_policy)
        narrow_workload = TPCCAgentWorkload(
            TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=4,
                order_lines=1,
                candidates_per_task=1,
                transaction_mix=(("new_order", 1.0),),
            )
        )
        register_workload(narrow_manager, narrow_workload)
        narrow_task = narrow_workload.generate_tasks(1, seed=1)[0]
        self.assertTrue(execute_task(narrow_manager, narrow_task, cc="adaptive").committed)
        narrow_validate = [
            event
            for event in narrow_manager.traces()[-1]["events"]
            if event["kind"] == "validate"
        ][0]
        self.assertEqual(narrow_validate["detail"]["selected_cc"], "semantic")

        wide_manager = AgentTransactionManager(
            adaptive_policy=AdaptivePolicyTable.new_order()
        )
        wide_workload = TPCCAgentWorkload(
            TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=8,
                order_lines=4,
                candidates_per_task=1,
                transaction_mix=(("new_order", 1.0),),
            )
        )
        register_workload(wide_manager, wide_workload)
        wide_task = wide_workload.generate_tasks(1, seed=1)[0]
        self.assertTrue(execute_task(wide_manager, wide_task, cc="adaptive").committed)
        wide_validate = [
            event
            for event in wide_manager.traces()[-1]["events"]
            if event["kind"] == "validate"
        ][0]
        self.assertEqual(wide_validate["detail"]["selected_cc"], "2pl")

    def test_operation_level_new_order_policy_mixes_optimistic_and_pessimistic_ops(self):
        manager = AgentTransactionManager(
            operation_policy=OperationPolicyTable.tpcc_new_order()
        )
        workload = TPCCAgentWorkload(
            TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=4,
                order_lines=2,
                candidates_per_task=2,
                transaction_mix=(("new_order", 1.0),),
            )
        )
        register_workload(manager, workload)
        task = workload.generate_tasks(1, seed=1)[0]
        result = execute_task(manager, task, cc="adaptive-op")
        self.assertTrue(result.committed)

        validate = [
            event
            for event in manager.traces()[-1]["events"]
            if event["kind"] == "validate"
        ][0]
        decisions = validate["detail"]["operation_policy_decisions"]
        policies = {decision["policy"] for decision in decisions}
        self.assertIn("optimistic", policies)
        self.assertIn("pessimistic", policies)
        self.assertTrue(validate["detail"]["operation_lock_targets"])

    def test_online_operation_atcc_uses_feedback_not_only_static_thresholds(self):
        policy = OperationPolicyTable.ycsb_atcc()
        profile = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            candidate_count=1,
            operation_count_for_object=1,
            total_writes=1,
        )

        initial = policy.select_profiles((profile,))[0]
        self.assertEqual(initial.policy, "optimistic")

        for _ in range(policy.min_feedback_observations):
            policy.observe_result(
                (initial,),
                committed=False,
                rejected=False,
                conflict_abort=True,
            )

        conflict_adapted = policy.select_profiles((profile,))[0]
        self.assertEqual(conflict_adapted.policy, "pessimistic")
        self.assertEqual(
            conflict_adapted.rule, "feedback-conflict-risk-pessimistic"
        )

        for _ in range(policy.min_feedback_observations * 2):
            policy.observe_result(
                (conflict_adapted,),
                committed=True,
                rejected=False,
                conflict_abort=False,
                lock_wait_s=0.1,
            )

        wait_adapted = policy.select_profiles((profile,))[0]
        self.assertEqual(wait_adapted.policy, "optimistic")
        self.assertEqual(wait_adapted.rule, "feedback-lock-cost-optimistic")

    def test_online_operation_atcc_attributes_conflict_to_stale_object(self):
        policy = OperationPolicyTable.ycsb_atcc()
        hot = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
        )
        cold = OperationPolicyProfile(
            object_id="ycsb:record:1:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
        )
        initial = policy.select_profiles((hot, cold))

        for _ in range(policy.min_feedback_observations):
            policy.observe_result(
                initial,
                committed=False,
                rejected=False,
                conflict_abort=True,
                conflict_object_ids=("ycsb:record:0:field:0",),
            )

        adapted = {
            decision.object_id: decision
            for decision in policy.select_profiles((hot, cold))
        }
        self.assertEqual(adapted["ycsb:record:0:field:0"].policy, "pessimistic")
        self.assertEqual(adapted["ycsb:record:1:field:0"].policy, "optimistic")

    def test_rl_operation_atcc_learns_action_value_from_conflict_reward(self):
        policy = dataclasses.replace(
            OperationPolicyTable.ycsb_rl_atcc(),
            rl_learner=OperationPolicyQLearner(
                learning_rate=1.0,
                epsilon=0.0,
                min_epsilon=0.0,
                epsilon_decay=1.0,
            ),
        )
        profile = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
        )

        first = policy.select_profiles((profile,))[0]
        self.assertEqual(first.policy, "optimistic")
        policy.observe_result(
            (first,),
            committed=False,
            rejected=False,
            conflict_abort=True,
            conflict_object_ids=("ycsb:record:0:field:0",),
        )

        learned = policy.select_profiles((profile,))[0]
        self.assertEqual(learned.policy, "pessimistic")
        self.assertEqual(learned.rule, "rl-q-pessimistic")
        self.assertLess(learned.rl_q_optimistic, learned.rl_q_pessimistic)

    def test_phase_aware_atcc_protects_tpcc_order_counter(self):
        policy = OperationPolicyTable.tpcc_phase_rl_atcc()
        policy.atcc_module.learner.epsilon = 0.0
        profiles = (
            OperationPolicyProfile(
                object_id="tpcc:district:1:1:next_order_id",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                candidate_count=4,
                operation_count_for_object=4,
                total_writes=12,
                agent_interval_s=0.120,
                agent_phase="commit",
            ),
            OperationPolicyProfile(
                object_id="tpcc:stock:1:7:quantity",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                candidate_count=4,
                operation_count_for_object=1,
                total_writes=12,
                agent_interval_s=0.120,
                agent_phase="commit",
            ),
        )

        decisions = {decision.object_id: decision for decision in policy.select_profiles(profiles)}
        counter = decisions["tpcc:district:1:1:next_order_id"]
        stock = decisions["tpcc:stock:1:7:quantity"]
        self.assertEqual(counter.policy, "pessimistic")
        self.assertEqual(stock.policy, "optimistic")
        self.assertEqual(counter.atcc_phase, "commit")
        self.assertTrue(counter.atcc_state_key)
        self.assertNotEqual(counter.atcc_state_key, stock.atcc_state_key)
        self.assertIn("class=tpcc:district:next_order_id", counter.atcc_state_key)
        self.assertIn("class=tpcc:stock:quantity", stock.atcc_state_key)
        self.assertIn(counter.atcc_action, {"lock-hot-writes", "lock-write-set"})

    def test_phase_aware_atcc_keeps_low_risk_tpcc_counter_optimistic(self):
        policy = OperationPolicyTable.tpcc_phase_rl_atcc()
        policy.atcc_module.learner.epsilon = 0.0
        low_risk = OperationPolicyProfile(
            object_id="tpcc:district:8:7:next_order_id",
            access_kind="write",
            intent_name="delta",
            task_type="new_order",
            workload="agent-tpcc-semantic",
            candidate_count=1,
            operation_count_for_object=1,
            total_writes=6,
            retry_count=0,
            agent_interval_s=0.010,
            agent_phase="commit",
        )

        initial = policy.select_profiles((low_risk,))[0]
        self.assertEqual(initial.policy, "optimistic")

        policy.atcc_module.learner.update(
            initial.atcc_state_key,
            "lock-write-set",
            10.0,
        )
        guarded = policy.select_profiles((low_risk,))[0]
        self.assertEqual(guarded.policy, "optimistic")
        self.assertEqual(guarded.atcc_action, "occ")
        self.assertEqual(guarded.rule, "phase-atcc-commit-occ")

    def test_phase_aware_atcc_overrides_low_abort_lock_pressure_q_lock_to_occ(self):
        policy = OperationPolicyTable.tpcc_phase_rl_atcc()
        policy.atcc_module.learner.epsilon = 0.0
        lock_pressure = OperationPolicyProfile(
            object_id="tpcc:district:8:7:next_order_id",
            access_kind="write",
            intent_name="delta",
            task_type="new_order",
            workload="agent-tpcc-semantic",
            candidate_count=4,
            operation_count_for_object=1,
            total_writes=6,
            retry_count=0,
            agent_interval_s=0.060,
            agent_phase="refine",
        )
        for _ in range(4):
            policy.atcc_runtime_stats.observe(
                committed=True,
                rejected=False,
                conflict_abort=False,
                lock_wait_s=0.020,
                latency_s=0.080,
                lock_queue_depth=2.0,
                lock_handoff_count=2.0,
                committing_count=0.0,
            )

        initial = policy.select_profiles((lock_pressure,))[0]
        policy.atcc_module.learner.update(
            initial.atcc_state_key,
            "lock-write-set",
            10.0,
        )

        guarded = policy.select_profiles((lock_pressure,))[0]

        self.assertEqual(guarded.policy, "optimistic")
        self.assertEqual(guarded.atcc_action, "occ")
        self.assertEqual(guarded.rule, "phase-atcc-refine-occ")

    def test_phase_aware_atcc_keeps_high_abort_counter_pessimistic(self):
        policy = OperationPolicyTable.tpcc_phase_rl_atcc()
        policy.atcc_module.learner.epsilon = 0.0
        hot_counter = OperationPolicyProfile(
            object_id="tpcc:district:0:0:next_order_id",
            access_kind="write",
            intent_name="delta",
            task_type="new_order",
            workload="agent-tpcc-semantic",
            candidate_count=4,
            operation_count_for_object=1,
            total_writes=6,
            retry_count=1,
            agent_interval_s=0.120,
            agent_phase="commit",
        )
        policy.atcc_runtime_stats.observe(
            committed=False,
            rejected=False,
            conflict_abort=True,
            lock_wait_s=0.0,
            latency_s=0.120,
        )

        decision = policy.select_profiles((hot_counter,))[0]

        self.assertEqual(decision.policy, "pessimistic")
        self.assertNotEqual(decision.atcc_action, "occ")

    def test_phase_aware_atcc_state_tracks_retry_and_agent_interval(self):
        policy = OperationPolicyTable.ycsb_phase_rl_atcc()
        policy.atcc_module.learner.epsilon = 0.0
        cold = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            candidate_count=4,
            operation_count_for_object=1,
            total_writes=4,
            retry_count=2,
            agent_interval_s=0.250,
            agent_phase="commit",
        )

        decision = policy.select_profiles((cold,))[0]
        self.assertEqual(decision.atcc_phase, "commit")
        self.assertIn("retry=2", decision.atcc_state_key)
        self.assertIn("interval=200ms-1s", decision.atcc_state_key)
        self.assertGreaterEqual(decision.atcc_priority, 1)
        policy.observe_result(
            (decision,),
            committed=False,
            rejected=False,
            conflict_abort=True,
            conflict_object_ids=("ycsb:record:0:field:0",),
            latency_s=0.25,
        )
        learner_state = policy.to_dict()["atcc_module"]["learner"]
        self.assertGreater(learner_state["updates"], 0)
        runtime_stats = policy.to_dict()["atcc_runtime_stats"]
        self.assertEqual(runtime_stats["observations"], 1)
        self.assertGreater(runtime_stats["ewma_abort_rate"], 0.0)
        next_decision = policy.select_profiles((cold,))[0]
        self.assertIn("globalAbort=", next_decision.atcc_state_key)
        self.assertIn("globalLatency=", next_decision.atcc_state_key)

    def test_atcc_runtime_stats_tracks_queue_pressure(self):
        from agent.runtime.atcc import ATCCRuntimeStats

        stats = ATCCRuntimeStats()
        stats.observe(
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=0.020,
            latency_s=0.100,
            lock_queue_depth=3.0,
            lock_handoff_count=2,
            committing_count=1,
        )

        exported = stats.to_dict()
        self.assertEqual(exported["observations"], 1)
        self.assertAlmostEqual(exported["ewma_lock_queue_depth"], 3.0)
        self.assertAlmostEqual(exported["ewma_lock_handoff_count"], 2.0)
        self.assertAlmostEqual(exported["ewma_committing_count"], 1.0)

        restored = ATCCRuntimeStats.from_dict(exported)
        self.assertEqual(restored.state_buckets()[4], "3-4")
        self.assertEqual(restored.state_buckets()[5], "2")
        self.assertEqual(restored.state_buckets()[6], "1")

    def test_phase_aware_atcc_state_tracks_queue_pressure(self):
        policy = OperationPolicyTable.tpcc_phase_rl_atcc()
        policy.atcc_module.learner.epsilon = 0.0
        profile = OperationPolicyProfile(
            object_id="tpcc:district:1:1:next_order_id",
            access_kind="write",
            intent_name="delta",
            task_type="new_order",
            workload="agent-tpcc-semantic",
            candidate_count=4,
            operation_count_for_object=4,
            total_writes=12,
            retry_count=1,
            agent_interval_s=0.120,
            agent_phase="commit",
        )

        first = policy.select_profiles((profile,))[0]
        self.assertEqual(first.policy, "pessimistic")
        policy.observe_result(
            (first,),
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=0.030,
            lock_queue_by_object={"tpcc:district:1:1:next_order_id": 4.0},
            latency_s=0.100,
        )

        second = policy.select_profiles((profile,))[0]
        self.assertIn("globalQueueDepth=3-4", second.atcc_state_key)
        self.assertIn("globalHandoff=", second.atcc_state_key)
        self.assertIn("globalCommitting=", second.atcc_state_key)

    def test_phase_aware_atcc_omits_zero_queue_pressure_from_state_key(self):
        policy = OperationPolicyTable.ycsb_phase_rl_atcc()
        policy.atcc_module.learner.epsilon = 0.0
        profile = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            candidate_count=4,
            operation_count_for_object=1,
            total_writes=4,
            retry_count=0,
            agent_interval_s=0.010,
            agent_phase="commit",
        )

        decision = policy.select_profiles((profile,))[0]

        self.assertNotIn("globalQueueDepth=", decision.atcc_state_key)
        self.assertNotIn("globalHandoff=", decision.atcc_state_key)
        self.assertNotIn("globalCommitting=", decision.atcc_state_key)

    def test_phase_aware_atcc_reward_penalizes_queue_pressure(self):
        module = OperationPolicyTable.ycsb_phase_rl_atcc().atcc_module
        optimistic = module.reward(
            "occ",
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=0.0,
            retry_count=0,
            agent_interval_s=0.050,
            operation_count=4,
            lock_queue_depth=8.0,
        )
        pessimistic_low_queue = module.reward(
            "lock-hot-writes",
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=0.001,
            retry_count=0,
            agent_interval_s=0.050,
            operation_count=4,
            lock_queue_depth=0.0,
        )
        pessimistic_high_queue = module.reward(
            "lock-hot-writes",
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=0.001,
            retry_count=0,
            agent_interval_s=0.050,
            operation_count=4,
            lock_queue_depth=8.0,
        )

        self.assertEqual(optimistic, module.commit_reward)
        self.assertLess(pessimistic_high_queue, pessimistic_low_queue)

    def test_phase_aware_atcc_reward_penalizes_handoff_and_committing_pressure(self):
        module = OperationPolicyTable.ycsb_phase_rl_atcc().atcc_module
        pessimistic_low_pressure = module.reward(
            "lock-hot-writes",
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=0.001,
            retry_count=0,
            agent_interval_s=0.050,
            operation_count=4,
            lock_queue_depth=0.0,
            lock_handoff_count=0.0,
            committing_count=0.0,
        )
        pessimistic_high_pressure = module.reward(
            "lock-hot-writes",
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=0.001,
            retry_count=0,
            agent_interval_s=0.050,
            operation_count=4,
            lock_queue_depth=0.0,
            lock_handoff_count=3.0,
            committing_count=4.0,
        )

        self.assertLess(pessimistic_high_pressure, pessimistic_low_pressure)

    def test_phase_aware_atcc_observes_handoff_and_committing_pressure(self):
        policy = OperationPolicyTable.tpcc_phase_rl_atcc()
        policy.atcc_module.learner.epsilon = 0.0
        profile = OperationPolicyProfile(
            object_id="tpcc:district:1:1:next_order_id",
            access_kind="write",
            intent_name="delta",
            task_type="new_order",
            workload="agent-tpcc-semantic",
            candidate_count=4,
            operation_count_for_object=4,
            total_writes=12,
            retry_count=1,
            agent_interval_s=0.120,
            agent_phase="commit",
        )

        first = policy.select_profiles((profile,))[0]
        self.assertEqual(first.policy, "pessimistic")
        policy.observe_result(
            (first,),
            committed=True,
            rejected=False,
            conflict_abort=False,
            lock_wait_s=0.030,
            lock_queue_by_object={"tpcc:district:1:1:next_order_id": 1.0},
            lock_handoff_by_object={"tpcc:district:1:1:next_order_id": 2.0},
            committing_count=3.0,
            latency_s=0.100,
        )

        exported = policy.to_dict()["atcc_runtime_stats"]
        self.assertAlmostEqual(exported["ewma_lock_handoff_count"], 2.0)
        self.assertAlmostEqual(exported["ewma_committing_count"], 3.0)
        second = policy.select_profiles((profile,))[0]
        self.assertIn("globalHandoff=2", second.atcc_state_key)
        self.assertIn("globalCommitting=3-4", second.atcc_state_key)

    def test_phase_atcc_runtime_stats_loads_legacy_artifact_without_queue_pressure(self):
        policy = OperationPolicyTable.ycsb_phase_rl_atcc()
        artifact = policy.to_dict()
        runtime = artifact["atcc_runtime_stats"]
        runtime.pop("ewma_lock_queue_depth", None)
        runtime.pop("ewma_lock_handoff_count", None)
        runtime.pop("ewma_committing_count", None)

        loaded = OperationPolicyTable.ycsb_phase_rl_atcc().with_learned_state(
            artifact,
            load_runtime_stats=True,
        )

        exported = loaded.to_dict()["atcc_runtime_stats"]
        self.assertEqual(exported["ewma_lock_queue_depth"], 0.0)
        self.assertEqual(exported["ewma_lock_handoff_count"], 0.0)
        self.assertEqual(exported["ewma_committing_count"], 0.0)

class AgentWorkloadTests(unittest.TestCase):
    def test_ycsb_is_deterministic_serializable_and_executable(self):
        workload = YCSBAgentWorkload(
            YCSBConfig(
                record_count=8,
                field_count=2,
                requests_per_task=2,
                candidates_per_task=3,
                read_weight=0.25,
                update_weight=0.75,
            )
        )
        first = workload.generate_tasks(3, seed=17)
        second = workload.generate_tasks(3, seed=17)
        self.assertEqual(
            [task.to_dict() for task in first],
            [task.to_dict() for task in second],
        )
        json.dumps([task.to_dict() for task in first])
        manifest = workload.manifest().to_dict()
        self.assertEqual(manifest["benchmark_family"], "YCSB")
        self.assertEqual(manifest["source_system"], "DBx1000")
        self.assertIn("third_party/dbx1000/benchmarks/ycsb_wl.cpp", manifest["source_files"])
        self.assertIn("ranked K candidate plans", manifest["agent_adaptations"])

        manager = AgentTransactionManager()
        register_workload(manager, workload)
        results = [execute_task(manager, task) for task in first]
        self.assertTrue(all(result.committed for result in results))
        self.assertTrue(all(len(task.candidates) == 3 for task in first))

    def test_faithful_workload_layers_use_single_candidate(self):
        ycsb = YCSBFaithfulAgentWorkload(
            YCSBConfig(record_count=4, field_count=2, candidates_per_task=3)
        )
        ycsb_task = ycsb.generate_tasks(1, seed=1)[0]
        self.assertEqual(ycsb.name, "agent-ycsb-faithful")
        self.assertEqual(ycsb.manifest().to_dict()["workload_layer"], "faithful")
        self.assertEqual(len(ycsb_task.candidates), 1)

        tpcc = TPCCFaithfulAgentWorkload(
            TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=4,
                transaction_mix=(
                    ("new_order", 0.5),
                    ("payment", 0.5),
                    ("delivery", 0.5),
                ),
                candidates_per_task=3,
            )
        )
        task = tpcc.generate_tasks(1, seed=2)[0]
        self.assertIn(task.task_type, {"new_order", "payment"})
        self.assertEqual(len(task.candidates), 1)
        self.assertEqual(tpcc.manifest().to_dict()["workload_layer"], "faithful")

    def test_ycsb_rejects_an_impossible_request_width(self):
        with self.assertRaises(ValueError):
            YCSBConfig(
                record_count=1,
                field_count=1,
                requests_per_task=2,
            )

    def test_tpcc_new_order_preserves_stock_and_exposes_candidates(self):
        workload = TPCCAgentWorkload(
            TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=3,
                items=6,
                initial_stock=20,
                order_lines=2,
                candidates_per_task=2,
                transaction_mix=(("new_order", 1.0),),
            )
        )
        tasks = workload.generate_tasks(4, seed=4)
        json.dumps([task.to_dict() for task in tasks])
        manifest = workload.manifest().to_dict()
        self.assertEqual(manifest["benchmark_family"], "TPC-C")
        self.assertEqual(manifest["source_system"], "DBx1000")
        self.assertIn("third_party/dbx1000/benchmarks/tpcc_txn.cpp", manifest["source_files"])
        self.assertIn("stock lower-bound constraint for new-order", manifest["preserved_semantics"])
        self.assertTrue(all(task.task_type == "new_order" for task in tasks))
        self.assertTrue(all(len(task.candidates) == 2 for task in tasks))

        manager = AgentTransactionManager()
        register_workload(manager, workload)
        results = [execute_task(manager, task) for task in tasks]
        self.assertTrue(all(result.committed for result in results))

        stock_values = [
            int(value)
            for key, value in manager.values().items()
            if key.startswith("tpcc:stock:") and key.endswith(":quantity")
        ]
        self.assertTrue(stock_values)
        self.assertTrue(all(value >= 0 for value in stock_values))

    def test_tpcc_transaction_families_generate_independently(self):
        for transaction_type in (
            "payment",
            "order_status",
            "delivery",
            "stock_level",
        ):
            workload = TPCCAgentWorkload(
                TPCCConfig(
                    warehouses=1,
                    districts_per_warehouse=1,
                    customers_per_district=3,
                    items=4,
                    initial_stock=10,
                    order_lines=2,
                    candidates_per_task=2,
                    transaction_mix=((transaction_type, 1.0),),
                )
            )
            task = workload.generate_tasks(1, seed=9)[0]
            self.assertEqual(task.task_type, transaction_type)
            self.assertTrue(task.candidates)
            json.dumps(task.to_dict())


class StrategyMatrixEvaluationTests(unittest.TestCase):
    def test_tpcc_contention_window_distinguishes_semantic_and_strict_cc(self):
        workload = TPCCAgentWorkload(
            TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=2,
                initial_stock=100,
                order_lines=1,
                candidates_per_task=1,
                transaction_mix=(("new_order", 1.0),),
            )
        )
        summaries = {
            summary.strategy: summary
            for summary in run_strategy_matrix(
                workload,
                ("semantic", "adaptive", "occ", "2pl"),
                task_count=4,
                seed=3,
                contention_window=4,
            )
        }
        json.dumps([summary.to_dict() for summary in summaries.values()])

        self.assertEqual(summaries["semantic"].committed, 4)
        self.assertEqual(
            summaries["semantic"].workload_manifest["benchmark_family"], "TPC-C"
        )
        self.assertEqual(
            summaries["semantic"].workload_manifest["source_system"], "DBx1000"
        )
        self.assertGreaterEqual(summaries["semantic"].action_counts.get("merge", 0), 1)
        self.assertEqual(summaries["adaptive"].selected_cc_counts, {"semantic": 4})
        self.assertEqual(summaries["adaptive"].committed, 4)

        self.assertEqual(summaries["occ"].committed, 1)
        self.assertEqual(summaries["occ"].aborted, 3)
        self.assertEqual(summaries["occ"].action_counts["regenerate_required"], 3)
        self.assertEqual(summaries["2pl"].committed, 1)
        self.assertEqual(summaries["2pl"].aborted, 3)

    def test_ycsb_strategy_matrix_is_deterministic_and_serializable(self):
        workload = YCSBAgentWorkload(
            YCSBConfig(
                record_count=4,
                field_count=2,
                requests_per_task=2,
                candidates_per_task=2,
                read_weight=0.5,
                update_weight=0.5,
            )
        )
        first = run_strategy_matrix(
            workload,
            ("semantic", "occ"),
            task_count=3,
            seed=11,
            contention_window=2,
        )
        second = run_strategy_matrix(
            workload,
            ("semantic", "occ"),
            task_count=3,
            seed=11,
            contention_window=2,
        )
        first_rows = [
            {k: v for k, v in summary.to_dict().items() if k != "elapsed_s"}
            for summary in first
        ]
        second_rows = [
            {k: v for k, v in summary.to_dict().items() if k != "elapsed_s"}
            for summary in second
        ]
        self.assertEqual(first_rows, second_rows)
        json.dumps([summary.to_dict() for summary in first])


if __name__ == "__main__":
    unittest.main()

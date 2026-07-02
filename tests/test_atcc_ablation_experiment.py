import json
import tempfile
import unittest
from pathlib import Path

from agent.evaluation.atcc_ablation_experiment import (
    STATIC_OPERATION_WIDE_OVERWRITE_THRESHOLD,
    STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD,
    STATIC_TRANSACTION_WIDE_WRITE_THRESHOLD,
    StaticOperationATCCPolicy,
    StaticTransactionATCCModule,
    _freeze_transaction_policy_learning,
    _operation_policy_for_variant,
    _transaction_policy_for_variant,
    _variant_spec,
    main,
)
from agent.runtime import ATCCRuntimeStats
from agent.runtime.atcc import TransactionAwareATCCModule
from agent.runtime.adaptive import OperationPolicyProfile


class ATCCAblationExperimentTests(unittest.TestCase):
    def test_static_ycsb_operation_policy_uses_generic_thresholds(self):
        policy = StaticOperationATCCPolicy("ycsb", priority_enabled=False)
        hot = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=4,
            retry_count=0,
            agent_phase="commit",
            hotspot_record_count=2,
        )
        repeated = OperationPolicyProfile(
            object_id="ycsb:record:7:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=1,
            retry_count=0,
            operation_count_for_object=2,
            agent_phase="commit",
            hotspot_record_count=2,
        )

        hot_decision, repeated_decision = policy.select_profiles((hot, repeated))

        self.assertEqual("optimistic", hot_decision.policy)
        self.assertEqual("pessimistic", repeated_decision.policy)
        self.assertEqual(0, hot_decision.atcc_priority)
        self.assertIn("priority=0", hot_decision.atcc_state_key)

    def test_conservative_static_operation_wide_overwrite_threshold(self):
        policy = StaticOperationATCCPolicy("ycsb", priority_enabled=False)
        below = OperationPolicyProfile(
            object_id="ycsb:record:10:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=STATIC_OPERATION_WIDE_OVERWRITE_THRESHOLD - 1,
            retry_count=0,
            agent_phase="commit",
        )
        at_threshold = OperationPolicyProfile(
            object_id="ycsb:record:11:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=STATIC_OPERATION_WIDE_OVERWRITE_THRESHOLD,
            retry_count=0,
            agent_phase="commit",
        )

        below_decision, at_decision = policy.select_profiles((below, at_threshold))

        self.assertEqual("optimistic", below_decision.policy)
        self.assertEqual("pessimistic", at_decision.policy)

    def test_static_tpcc_operation_policy_does_not_oracle_lock_counter_name(self):
        policy = StaticOperationATCCPolicy("tpcc", priority_enabled=True)
        counter = OperationPolicyProfile(
            object_id="tpcc:district:1:1:next_order_id",
            access_kind="write",
            intent_name="overwrite",
            task_type="new_order",
            workload="agent-tpcc-semantic",
            total_writes=10,
            agent_interval_s=0.050,
            agent_phase="commit",
        )
        retry_counter = OperationPolicyProfile(
            object_id="tpcc:district:1:1:next_order_id",
            access_kind="write",
            intent_name="overwrite",
            task_type="new_order",
            workload="agent-tpcc-semantic",
            total_writes=10,
            retry_count=1,
            agent_interval_s=0.050,
            agent_phase="commit",
        )

        counter_decision, retry_decision = policy.select_profiles((counter, retry_counter))

        self.assertEqual("optimistic", counter_decision.policy)
        self.assertEqual("pessimistic", retry_decision.policy)
        self.assertGreater(retry_decision.atcc_priority, 0)

    def test_static_transaction_policy_selects_write_set_after_retry(self):
        module = StaticTransactionATCCModule("ycsb", priority_enabled=False)
        profiles = (
            OperationPolicyProfile(
                object_id="ycsb:record:0:field:0",
                access_kind="write",
                intent_name="overwrite",
                task_type="read-update",
                workload="agent-ycsb-semantic",
                retry_count=1,
                total_writes=2,
                agent_phase="commit",
                hotspot_record_count=2,
            ),
            OperationPolicyProfile(
                object_id="ycsb:record:4:field:0",
                access_kind="write",
                intent_name="overwrite",
                task_type="read-update",
                workload="agent-ycsb-semantic",
                retry_count=1,
                total_writes=2,
                agent_phase="commit",
                hotspot_record_count=2,
            ),
        )

        decision = module.select_transaction(profiles)

        self.assertEqual("lock-write-set", decision.action)
        self.assertEqual(0, decision.priority)
        self.assertIn("priority=0", decision.state_key)
        self.assertEqual(
            ("ycsb:record:0:field:0", "ycsb:record:4:field:0"),
            decision.prelock_targets,
        )

    def test_conservative_static_transaction_wide_write_threshold(self):
        module = StaticTransactionATCCModule("tpcc", priority_enabled=False)

        below_threshold = tuple(
            OperationPolicyProfile(
                object_id=f"tpcc:stock:0:{index}:quantity",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                retry_count=0,
                total_writes=STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD - 1,
                agent_phase="commit",
            )
            for index in range(STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD - 1)
        )
        at_threshold = tuple(
            OperationPolicyProfile(
                object_id=f"tpcc:stock:0:{index}:quantity",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                retry_count=0,
                total_writes=STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD,
                agent_phase="commit",
            )
            for index in range(STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD)
        )

        self.assertEqual("occ", module.select_transaction(below_threshold).action)
        self.assertEqual("lock-write-set", module.select_transaction(at_threshold).action)

    def test_threshold32_static_preset_preserves_previous_transaction_threshold(self):
        module = StaticTransactionATCCModule(
            "tpcc",
            priority_enabled=False,
            static_preset="threshold32",
        )

        below_threshold = tuple(
            OperationPolicyProfile(
                object_id=f"tpcc:stock:0:{index}:quantity",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                retry_count=0,
                total_writes=STATIC_TRANSACTION_WIDE_WRITE_THRESHOLD - 1,
                agent_phase="commit",
            )
            for index in range(STATIC_TRANSACTION_WIDE_WRITE_THRESHOLD - 1)
        )
        at_threshold = tuple(
            OperationPolicyProfile(
                object_id=f"tpcc:stock:0:{index}:quantity",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                retry_count=0,
                total_writes=STATIC_TRANSACTION_WIDE_WRITE_THRESHOLD,
                agent_phase="commit",
            )
            for index in range(STATIC_TRANSACTION_WIDE_WRITE_THRESHOLD)
        )

        self.assertEqual("occ", module.select_transaction(below_threshold).action)
        self.assertEqual("lock-write-set", module.select_transaction(at_threshold).action)

    def test_threshold32_static_preset_preserves_previous_operation_threshold(self):
        policy = StaticOperationATCCPolicy(
            "ycsb",
            priority_enabled=False,
            static_preset="threshold32",
        )
        wide = OperationPolicyProfile(
            object_id="ycsb:record:20:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=12,
            retry_count=0,
            agent_phase="commit",
        )

        (decision,) = policy.select_profiles((wide,))

        self.assertEqual("pessimistic", decision.policy)

    def test_no_priority_dynamic_variants_zero_priority_score(self):
        op_policy = _operation_policy_for_variant(
            "ycsb",
            _variant_spec("op-dynamic"),
        )
        tx_policy = _transaction_policy_for_variant(
            "tpcc",
            _variant_spec("tx-dynamic"),
        )

        self.assertEqual(0, op_policy.atcc_module.priority_score(
            profiles=(),
            retry_count=10,
            agent_interval_s=1.0,
            hot_read_ratio=1.0,
            hot_write_ratio=1.0,
            global_abort_rate=1.0,
        ))
        self.assertEqual(0, tx_policy._priority_score(
            operation_count=100,
            retry_count=10,
            agent_interval_s=1.0,
            hot_read_count=10,
            hot_write_count=10,
            global_abort_rate=1.0,
        ))

    def test_priority_variants_cap_positive_priority_scores(self):
        op_policy = _operation_policy_for_variant(
            "ycsb",
            _variant_spec("op-dynamic-priority"),
            priority_cap=1,
        )
        tx_policy = _transaction_policy_for_variant(
            "tpcc",
            _variant_spec("tx-dynamic-priority"),
            priority_cap=1,
        )

        self.assertEqual(1, op_policy.atcc_module.priority_score(
            profiles=(object(),) * 100,
            retry_count=10,
            agent_interval_s=1.0,
            hot_read_ratio=1.0,
            hot_write_ratio=1.0,
            global_abort_rate=1.0,
        ))
        self.assertEqual(1, tx_policy._priority_score(
            operation_count=100,
            retry_count=10,
            agent_interval_s=1.0,
            hot_read_count=10,
            hot_write_count=10,
            global_abort_rate=1.0,
        ))

    def test_dynamic_transaction_priority_waits_for_retry_or_pressure(self):
        tx_policy = _transaction_policy_for_variant(
            "tpcc",
            _variant_spec("tx-dynamic-priority"),
            priority_cap=1,
        )

        self.assertEqual(0, tx_policy._priority_score(
            operation_count=100,
            retry_count=0,
            agent_interval_s=1.0,
            hot_read_count=0,
            hot_write_count=10,
            global_abort_rate=0.0,
        ))
        self.assertEqual(1, tx_policy._priority_score(
            operation_count=100,
            retry_count=1,
            agent_interval_s=1.0,
            hot_read_count=0,
            hot_write_count=10,
            global_abort_rate=0.0,
        ))

    def test_frozen_transaction_policy_keeps_q_table_read_only(self):
        module = TransactionAwareATCCModule.ycsb()
        state_key = "scope=transaction|phase=commit"
        module.learner.update(state_key, "lock-hot-writes", 1.0)
        before = module.learner.to_dict()["updates"]

        _freeze_transaction_policy_learning(module)
        module.learner.update(state_key, "occ", 100.0)

        after = module.learner.to_dict()["updates"]
        self.assertEqual(before, after)
        self.assertEqual(0.0, module.learner.to_dict()["epsilon"])

    def test_transaction_artifact_load_resets_runtime_stats_for_test(self):
        trained = TransactionAwareATCCModule.ycsb()
        trained.runtime_stats.observe(
            committed=False,
            rejected=False,
            conflict_abort=True,
            lock_wait_s=1.0,
            latency_s=1.0,
        )
        loaded = _transaction_policy_for_variant(
            "ycsb",
            _variant_spec("tx-dynamic-priority"),
            learned_artifact={"transaction_atcc_module": trained.to_dict()},
            priority_cap=1,
            freeze_learning=True,
        )

        self.assertEqual(0, loaded.runtime_stats.observations)
        self.assertEqual(0.0, loaded.runtime_stats.ewma_abort_rate)

    def test_dynamic_transaction_ablation_uses_compact_state(self):
        module = _transaction_policy_for_variant(
            "tpcc",
            _variant_spec("tx-dynamic-priority"),
            priority_cap=1,
        )
        profiles = tuple(
            OperationPolicyProfile(
                object_id=f"tpcc:stock:0:{index}:quantity",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                retry_count=0,
                total_writes=20,
                agent_interval_s=0.025,
                agent_phase="commit",
            )
            for index in range(20)
        )

        decision = module.select_transaction(
            profiles,
            runtime_stats=ATCCRuntimeStats(),
        )

        self.assertIn("scope=transaction-ablation-dynamic", decision.state_key)
        self.assertIn("pressure=cold", decision.state_key)
        self.assertNotIn("globalObs=", decision.state_key)
        self.assertNotIn("globalLockWait=", decision.state_key)

    def test_dynamic_transaction_ablation_prior_locks_wide_retry(self):
        module = _transaction_policy_for_variant(
            "tpcc",
            _variant_spec("tx-dynamic"),
        )
        profiles = tuple(
            OperationPolicyProfile(
                object_id=f"tpcc:stock:0:{index}:quantity",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                retry_count=1,
                total_writes=20,
                agent_interval_s=0.050,
                agent_phase="commit",
            )
            for index in range(20)
        )

        decision = module.select_transaction(
            profiles,
            runtime_stats=ATCCRuntimeStats(),
        )

        self.assertEqual("lock-write-set", decision.action)
        self.assertEqual(0, decision.priority)

    def test_dynamic_transaction_ablation_overrides_occ_on_wide_retry(self):
        module = _transaction_policy_for_variant(
            "tpcc",
            _variant_spec("tx-dynamic"),
        )
        profiles = tuple(
            OperationPolicyProfile(
                object_id=f"tpcc:stock:0:{index}:quantity",
                access_kind="write",
                intent_name="delta",
                task_type="new_order",
                workload="agent-tpcc-semantic",
                retry_count=1,
                total_writes=20,
                agent_interval_s=0.050,
                agent_phase="commit",
            )
            for index in range(20)
        )
        first = module.select_transaction(
            profiles,
            runtime_stats=ATCCRuntimeStats(),
        )
        module.learner.update(first.state_key, "occ", 100.0)

        decision = module.select_transaction(
            profiles,
            runtime_stats=ATCCRuntimeStats(),
        )

        self.assertEqual("lock-write-set", decision.action)
        self.assertEqual(
            tuple(sorted(f"tpcc:stock:0:{index}:quantity" for index in range(20))),
            decision.prelock_targets,
        )

    def test_cli_writes_smoke_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "ablation"
            code = main(
                (
                    "--workload",
                    "ycsb",
                    "--profile",
                    "high",
                    "--variants",
                    "op-static",
                    "--seeds",
                    "1",
                    "--task-count",
                    "2",
                    "--workers",
                    "1",
                    "--agent-slots",
                    "0",
                    "--planning-delay-ms",
                    "0",
                    "--latency-distribution",
                    "fixed",
                    "--background-workers",
                    "0",
                    "--no-baselines",
                    "--output-dir",
                    str(output_dir),
                )
            )

            self.assertEqual(0, code)
            self.assertTrue((output_dir / "atcc_ablation.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "atcc_ablation_metrics.csv").exists())
            self.assertTrue((output_dir / "atcc_ablation_ratios.csv").exists())
            self.assertTrue((output_dir / "atcc_ablation_report.md").exists())
            report = json.loads(
                (output_dir / "atcc_ablation.json").read_text(encoding="utf-8")
            )
            self.assertEqual("conservative", report["static_preset"])
            self.assertEqual(
                STATIC_TRANSACTION_CONSERVATIVE_WIDE_WRITE_THRESHOLD,
                report["static_transaction_wide_write_threshold"],
            )
            markdown = (output_dir / "atcc_ablation_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("conflict aborts", markdown)

    def test_cli_trains_and_writes_dynamic_policy_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "ablation"
            code = main(
                (
                    "--workload",
                    "ycsb",
                    "--profile",
                    "high",
                    "--variants",
                    "op-dynamic",
                    "--seeds",
                    "1",
                    "--task-count",
                    "1",
                    "--train-seeds",
                    "2",
                    "--train-rounds",
                    "1",
                    "--train-task-count",
                    "1",
                    "--workers",
                    "1",
                    "--agent-slots",
                    "0",
                    "--planning-delay-ms",
                    "0",
                    "--latency-distribution",
                    "fixed",
                    "--background-workers",
                    "0",
                    "--no-baselines",
                    "--output-dir",
                    str(output_dir),
                )
            )

            self.assertEqual(0, code)
            self.assertTrue((output_dir / "atcc_ablation_policy_artifacts.json").exists())

    def test_cli_loads_pretrained_dynamic_policy_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp) / "train"
            test_dir = Path(tmp) / "test"
            code = main(
                (
                    "--workload",
                    "ycsb",
                    "--profile",
                    "high",
                    "--variants",
                    "op-dynamic",
                    "--seeds",
                    "1",
                    "--task-count",
                    "1",
                    "--train-seeds",
                    "2",
                    "--train-rounds",
                    "1",
                    "--train-task-count",
                    "1",
                    "--workers",
                    "1",
                    "--agent-slots",
                    "0",
                    "--planning-delay-ms",
                    "0",
                    "--latency-distribution",
                    "fixed",
                    "--background-workers",
                    "0",
                    "--no-baselines",
                    "--output-dir",
                    str(train_dir),
                )
            )
            self.assertEqual(0, code)

            artifact_path = train_dir / "atcc_ablation_policy_artifacts.json"
            code = main(
                (
                    "--workload",
                    "ycsb",
                    "--profile",
                    "high",
                    "--variants",
                    "op-dynamic",
                    "--seeds",
                    "3",
                    "--task-count",
                    "1",
                    "--train-rounds",
                    "0",
                    "--workers",
                    "1",
                    "--agent-slots",
                    "0",
                    "--planning-delay-ms",
                    "0",
                    "--latency-distribution",
                    "fixed",
                    "--background-workers",
                    "0",
                    "--no-baselines",
                    "--pretrained-artifacts",
                    str(artifact_path),
                    "--output-dir",
                    str(test_dir),
                )
            )

            self.assertEqual(0, code)
            report = json.loads(
                (test_dir / "atcc_ablation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(str(artifact_path), report["pretrained_artifacts_path"])
            self.assertIn("ycsb:high:op-dynamic", report["pretrained_artifact_keys"])
            self.assertEqual([], report["training_runs"])


if __name__ == "__main__":
    unittest.main()

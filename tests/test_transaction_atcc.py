import unittest

from agent.runtime import (
    AgentTransactionManager,
    ATCCRuntimeStats,
    TransactionAwareATCCModule,
)
from agent.evaluation.atcc_retry_experiment import _transaction_atcc_policy
from agent.runtime.adaptive import OperationPolicyProfile
from agent.workloads.base import AgentCandidate, AgentOperation, AgentTask


class TransactionAwareATCCTests(unittest.TestCase):
    def test_low_risk_cold_transaction_uses_occ_fast_path(self):
        module = TransactionAwareATCCModule.ycsb()
        profiles = (
            OperationPolicyProfile(
                object_id="ycsb:record:42:field:0",
                access_kind="read",
                intent_name="read",
                task_type="read-update",
                workload="agent-ycsb-semantic",
                retry_count=0,
                agent_phase="commit",
                agent_interval_s=0.020,
                hotspot_record_count=2,
                total_writes=1,
            ),
            OperationPolicyProfile(
                object_id="ycsb:record:43:field:0",
                access_kind="write",
                intent_name="overwrite",
                task_type="read-update",
                workload="agent-ycsb-semantic",
                retry_count=0,
                agent_phase="commit",
                agent_interval_s=0.020,
                hotspot_record_count=2,
                total_writes=1,
            ),
        )

        decision = module.select_transaction(
            profiles,
            runtime_stats=ATCCRuntimeStats(),
        )

        self.assertEqual("occ", decision.action)
        self.assertTrue(decision.fast_path)
        self.assertEqual((), decision.prelock_targets)
        self.assertIn("phase=commit", decision.state_key)

    def test_hot_ycsb_write_commit_locks_only_hot_write_set(self):
        module = TransactionAwareATCCModule.ycsb()
        profiles = (
            OperationPolicyProfile(
                object_id="ycsb:record:0:field:0",
                access_kind="write",
                intent_name="overwrite",
                task_type="read-update",
                workload="agent-ycsb-semantic",
                retry_count=0,
                agent_phase="commit",
                agent_interval_s=0.080,
                hotspot_record_count=2,
                total_writes=2,
            ),
            OperationPolicyProfile(
                object_id="ycsb:record:9:field:0",
                access_kind="write",
                intent_name="overwrite",
                task_type="read-update",
                workload="agent-ycsb-semantic",
                retry_count=0,
                agent_phase="commit",
                agent_interval_s=0.080,
                hotspot_record_count=2,
                total_writes=2,
            ),
            OperationPolicyProfile(
                object_id="ycsb:record:0:field:1",
                access_kind="read",
                intent_name="read",
                task_type="read-update",
                workload="agent-ycsb-semantic",
                retry_count=0,
                agent_phase="commit",
                agent_interval_s=0.080,
                hotspot_record_count=2,
                total_writes=2,
            ),
        )

        decision = module.select_transaction(
            profiles,
            runtime_stats=ATCCRuntimeStats(),
        )

        self.assertEqual("lock-hot-writes", decision.action)
        self.assertFalse(decision.fast_path)
        self.assertEqual(("ycsb:record:0:field:0",), decision.prelock_targets)
        self.assertEqual(("ycsb:record:0:field:1",), decision.hot_read_set)
        self.assertEqual(("ycsb:record:0:field:0",), decision.hot_write_set)
        self.assertEqual(("ycsb:record:9:field:0",), decision.cold_write_set)

    def test_registry_builds_transaction_atcc_pre_snapshot_plan(self):
        manager = AgentTransactionManager()
        task = AgentTask(
            task_id="hot-ycsb",
            workload="agent-ycsb-semantic",
            task_type="read-update",
            request="hot ycsb write",
            context={
                "agent_phase": "commit",
                "hot_record_count": 2,
                "hotspot_access_probability": 1.0,
            },
            candidates=(
                AgentCandidate(
                    "candidate",
                    1.0,
                    (
                        AgentOperation.overwrite("ycsb:record:0:field:0", "v"),
                        AgentOperation.overwrite("ycsb:record:9:field:0", "v"),
                        AgentOperation.read("ycsb:record:0:field:1"),
                    ),
                ),
            ),
        )

        targets, decisions = manager.cc_registry.pre_snapshot_operation_plan(
            "transaction-atcc-strict",
            task.candidates,
            metadata={
                "workload": task.workload,
                "task_type": task.task_type,
                "context": task.context,
                "agent_phase": "commit",
            },
        )

        self.assertEqual(["ycsb:record:0:field:0"], targets)
        self.assertTrue(decisions)
        self.assertTrue(
            all(decision.rule == "transaction-atcc-lock-hot-writes" for decision in decisions)
        )
        self.assertEqual(
            {"pessimistic", "optimistic"},
            {decision.policy for decision in decisions},
        )

    def test_manager_accepts_transaction_atcc_policy_plugin(self):
        manager = AgentTransactionManager(
            transaction_atcc_policy=TransactionAwareATCCModule.tpcc()
        )

        metadata = manager.cc_strategy("transaction-atcc-strict")

        self.assertEqual("transaction_policy_table", metadata["selector"])
        self.assertEqual(
            "tpcc-transaction-aware-atcc",
            metadata["transaction_policy_table"]["name"],
        )

    def test_retry_runner_selects_workload_specific_transaction_policy(self):
        self.assertEqual(
            "tpcc-transaction-aware-atcc",
            _transaction_atcc_policy("tpcc").name,
        )
        self.assertEqual(
            "ycsb-transaction-aware-atcc",
            _transaction_atcc_policy("ycsb").name,
        )


if __name__ == "__main__":
    unittest.main()

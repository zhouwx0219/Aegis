import dataclasses
import unittest

from agent.evaluation.atcc_retry_experiment import _operation_policy
from agent.runtime import OperationPolicyTable
from agent.runtime.adaptive import (
    OperationPolicyProfile,
    operation_object_class,
    operation_profile_key,
)


class YCSBStrictTunedATCCTests(unittest.TestCase):
    def test_low_risk_first_attempt_stays_optimistic(self):
        policy = dataclasses.replace(
            OperationPolicyTable.ycsb_strict_tuned_atcc(),
            atcc_cold_occ_fast_path=True,
        )
        profile = OperationPolicyProfile(
            object_id="ycsb:record:42:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=1,
            retry_count=0,
            agent_phase="commit",
            agent_interval_s=0.050,
        )

        decisions = policy.select_profiles((profile,))

        self.assertEqual(["optimistic"], [decision.policy for decision in decisions])
        self.assertEqual("phase-atcc-fastpath-occ", decisions[0].rule)
        self.assertEqual("occ", decisions[0].atcc_action)
        self.assertEqual("fastpath-occ", decisions[0].atcc_phase)
        self.assertIn("fastpath=occ", decisions[0].atcc_state_key)

    def test_low_risk_fast_path_is_disabled_after_retry(self):
        policy = dataclasses.replace(
            OperationPolicyTable.ycsb_strict_tuned_atcc(),
            atcc_cold_occ_fast_path=True,
        )
        profile = OperationPolicyProfile(
            object_id="ycsb:record:42:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=1,
            retry_count=1,
            agent_phase="commit",
            agent_interval_s=0.050,
        )

        decisions = policy.select_profiles((profile,))

        self.assertEqual(["optimistic"], [decision.policy for decision in decisions])
        self.assertNotEqual("phase-atcc-fastpath-occ", decisions[0].rule)
        self.assertNotEqual("", decisions[0].atcc_action)
        self.assertNotEqual("", decisions[0].atcc_state_key)

    def test_wide_first_attempt_write_task_stays_out_of_fast_path(self):
        policy = OperationPolicyTable.ycsb_strict_tuned_atcc()
        profile = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=8,
            retry_count=0,
            agent_phase="commit",
            agent_interval_s=0.050,
        )

        decisions = policy.select_profiles((profile,))

        self.assertNotEqual("phase-atcc-fastpath-occ", decisions[0].rule)
        self.assertEqual("occ", decisions[0].atcc_action)
        self.assertEqual("optimistic", decisions[0].policy)
        self.assertNotEqual("", decisions[0].atcc_state_key)

    def test_high_retry_ycsb_write_pressure_locks_write_set_not_read_write_set(self):
        policy = OperationPolicyTable.ycsb_strict_tuned_atcc()
        write_profile = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=8,
            retry_count=3,
            agent_phase="commit",
            agent_interval_s=0.100,
        )
        read_profile = OperationPolicyProfile(
            object_id="ycsb:record:0:field:1",
            access_kind="read",
            intent_name="read",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=8,
            retry_count=3,
            agent_phase="commit",
            agent_interval_s=0.100,
        )

        decisions = policy.select_profiles((write_profile, read_profile))

        by_access = {decision.access_kind: decision for decision in decisions}
        self.assertEqual("lock-write-set", by_access["write"].atcc_action)
        self.assertEqual("pessimistic", by_access["write"].policy)
        self.assertEqual("lock-write-set", by_access["read"].atcc_action)
        self.assertEqual("optimistic", by_access["read"].policy)

    def test_retry_hot_write_locks_hot_writes_before_full_read_write_set(self):
        policy = OperationPolicyTable.ycsb_strict_tuned_atcc()
        profile = OperationPolicyProfile(
            object_id="ycsb:record:0:field:0",
            access_kind="write",
            intent_name="overwrite",
            task_type="read-update",
            workload="agent-ycsb-semantic",
            total_writes=12,
            retry_count=1,
            agent_phase="commit",
            agent_interval_s=0.080,
        )
        object_class = operation_object_class(profile.object_id)
        policy.telemetry.observe(
            (
                operation_profile_key(profile, object_class=object_class),
                operation_profile_key(
                    profile,
                    object_class=object_class,
                    exact_object=True,
                ),
            ),
            policy="optimistic",
            conflict_abort=True,
            committed=False,
            rejected=False,
            lock_wait_s=0.0,
        )

        decisions = policy.select_profiles((profile,))

        self.assertEqual("pessimistic", decisions[0].policy)
        self.assertEqual("lock-hot-writes", decisions[0].atcc_action)

    def test_runner_accepts_ycsb_strict_tuned_policy_variant(self):
        policy = _operation_policy("ycsb", "ycsb-strict-tuned")

        self.assertEqual("ycsb-strict-tuned-operation-atcc-table", policy.name)


if __name__ == "__main__":
    unittest.main()

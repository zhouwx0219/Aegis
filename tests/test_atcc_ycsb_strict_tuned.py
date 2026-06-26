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
        policy = OperationPolicyTable.ycsb_strict_tuned_atcc()
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
        self.assertEqual("occ", decisions[0].atcc_action)

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

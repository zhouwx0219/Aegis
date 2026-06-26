import unittest

from agent.evaluation.atcc_policy_training import train_phase_atcc_policy
from agent.runtime import OperationPolicyTable
from agent.workloads import YCSBConfig, build_agent_workload


class ATCCPolicyTrainingTests(unittest.TestCase):
    def test_training_artifact_contains_policy_table_and_statistics(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=4,
                field_count=1,
                requests_per_task=2,
                candidates_per_task=2,
                read_weight=0.0,
                update_weight=1.0,
                zipf_theta=0.0,
            ),
        )

        artifact = train_phase_atcc_policy(
            workload,
            workload_kind="ycsb",
            workload_config={"record_count": 4},
            episodes=1,
            task_count=3,
            seed=11,
            workers=1,
            agent_slots=0,
            agent_admission_mode="before-begin",
            planning_delay_s=0.0,
            latency_distribution="fixed",
            latency_cv=0.8,
            latency_max_s=0.0,
            max_attempts=2,
            tokens_per_operation=10.0,
            object_lock_scheduler="bounded-priority",
            object_lock_priority_burst=3,
            prelock_wait_budget_s=0.007,
            prelock_wait_budget_mode="object",
            prelock_lease_mode="defer-until-after-planning",
            atcc_lock_wait_cost_per_s=123.0,
            atcc_lock_action_cost=0.07,
            atcc_lock_queue_depth_cost=0.11,
            atcc_lock_handoff_cost=0.13,
            atcc_committing_count_cost=0.17,
        )

        table = artifact["operation_policy_table"]
        stats = artifact["stats"]
        learner = table["atcc_module"]["learner"]

        self.assertEqual(
            artifact["training_method"],
            "offline-simulation-tabular-q-learning",
        )
        self.assertEqual(artifact["artifact_version"], 2)
        self.assertEqual(artifact["atcc_state_schema"]["version"], 2)
        self.assertIn("class", artifact["atcc_state_schema"]["dimensions"])
        self.assertGreater(learner["updates"], 0)
        self.assertGreater(stats["atcc_state_count"], 0)
        self.assertEqual(stats["atcc_state_schema_version"], 2)
        self.assertTrue(stats["atcc_state_has_object_class"])
        self.assertGreater(stats["telemetry_observation_count"], 0)
        self.assertTrue(table["telemetry"])
        self.assertIn("atcc_runtime_stats", table)
        self.assertGreater(table["atcc_runtime_stats"]["observations"], 0)
        self.assertGreater(stats["atcc_runtime_observation_count"], 0)
        self.assertEqual(artifact["training_config"]["agent_admission_mode"], "before-begin")
        self.assertEqual(artifact["training_config"]["object_lock_scheduler"], "bounded-priority")
        self.assertEqual(artifact["training_config"]["object_lock_priority_burst"], 3)
        self.assertEqual(artifact["training_config"]["prelock_wait_budget_s"], 0.007)
        self.assertEqual(artifact["training_config"]["prelock_wait_budget_mode"], "object")
        self.assertEqual(artifact["training_config"]["prelock_lease_mode"], "defer-until-after-planning")
        self.assertEqual(artifact["runs"][0]["agent_admission_mode"], "before-begin")
        self.assertEqual(artifact["runs"][0]["object_lock_scheduler"], "bounded-priority")
        self.assertEqual(artifact["runs"][0]["prelock_wait_budget_mode"], "object")
        self.assertEqual(artifact["training_config"]["atcc_lock_wait_cost_per_s"], 123.0)
        self.assertEqual(artifact["training_config"]["atcc_lock_action_cost"], 0.07)
        self.assertEqual(artifact["training_config"]["atcc_lock_queue_depth_cost"], 0.11)
        self.assertEqual(artifact["training_config"]["atcc_lock_handoff_cost"], 0.13)
        self.assertEqual(artifact["training_config"]["atcc_committing_count_cost"], 0.17)
        self.assertEqual(table["atcc_module"]["lock_wait_cost_per_s"], 123.0)
        self.assertEqual(table["atcc_module"]["lock_action_cost"], 0.07)
        self.assertEqual(table["atcc_module"]["lock_queue_depth_cost"], 0.11)
        self.assertEqual(table["atcc_module"]["lock_handoff_cost"], 0.13)
        self.assertEqual(table["atcc_module"]["committing_count_cost"], 0.17)

        loaded = OperationPolicyTable.ycsb_phase_rl_atcc().with_learned_state(
            artifact,
            policy_epsilon=0.0,
        )
        loaded_table = loaded.to_dict()
        self.assertEqual(
            loaded_table["atcc_module"]["learner"]["updates"],
            learner["updates"],
        )
        self.assertEqual(loaded_table["atcc_module"]["learner"]["epsilon"], 0.0)
        self.assertEqual(loaded_table["atcc_module"]["learner"]["min_epsilon"], 0.0)
        self.assertEqual(loaded_table["telemetry"], table["telemetry"])
        self.assertEqual(loaded_table["atcc_runtime_stats"]["observations"], 0)
        loaded_with_runtime_stats = OperationPolicyTable.ycsb_phase_rl_atcc().with_learned_state(
            artifact,
            policy_epsilon=0.0,
            load_runtime_stats=True,
        )
        self.assertEqual(
            loaded_with_runtime_stats.to_dict()["atcc_runtime_stats"],
            table["atcc_runtime_stats"],
        )
        self.assertEqual(loaded_table["atcc_module"]["lock_wait_cost_per_s"], 123.0)
        self.assertEqual(loaded_table["atcc_module"]["lock_action_cost"], 0.07)
        self.assertEqual(loaded_table["atcc_module"]["lock_queue_depth_cost"], 0.11)
        self.assertEqual(loaded_table["atcc_module"]["lock_handoff_cost"], 0.13)
        self.assertEqual(loaded_table["atcc_module"]["committing_count_cost"], 0.17)


if __name__ == "__main__":
    unittest.main()

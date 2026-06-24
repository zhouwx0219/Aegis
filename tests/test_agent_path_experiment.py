import unittest

from agent.evaluation.agent_path_experiment import (
    learn_new_order_operation_policy,
    run_agent_path_matrix,
)
from agent.runtime import AgentTransactionManager, OperationPolicyTable
from agent.workloads import TPCCAgentWorkload, TPCCConfig


class AgentPathExperimentTests(unittest.TestCase):
    def test_operation_atcc_training_selects_hot_counter_threshold(self):
        result = learn_new_order_operation_policy(
            thresholds=(1, 2, 4),
            workload_config=TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=4,
                order_lines=1,
                candidates_per_task=2,
                transaction_mix=(("new_order", 1.0),),
            ),
            train_seeds=(1,),
            task_count=2,
            contention_window=2,
            hot_counter_miss_cost=5.0,
        )

        self.assertEqual(
            result["policy_family"],
            "tpcc-new-order-operation-atcc-hot-counter-threshold",
        )
        self.assertEqual(result["best_hot_object_threshold"], 2)
        best = [
            row
            for row in result["thresholds"]
            if row["threshold"] == result["best_hot_object_threshold"]
        ][0]
        self.assertGreater(best["hot_counter_pessimistic"], 0)

    def test_pre_snapshot_operation_switch_beats_occ_under_live_contention(self):
        workload = TPCCAgentWorkload(
            TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=2,
                items=16,
                order_lines=2,
                candidates_per_task=2,
                transaction_mix=(("new_order", 1.0),),
            )
        )
        summaries = {
            summary.strategy: summary
            for summary in run_agent_path_matrix(
                workload,
                ("occ", "2pl-pre", "adaptive-op-strict"),
                paths=("astra",),
                task_count=20,
                seed=7,
                contention_window=4,
                execution_mode="concurrent",
                planning_delay_s=0.002,
                manager_factory=lambda: AgentTransactionManager(
                    operation_policy=OperationPolicyTable.tpcc_new_order(
                        hot_object_threshold=2
                    )
                ),
            )
        }

        self.assertLess(summaries["occ"].committed_tasks, 20)
        self.assertEqual(summaries["2pl-pre"].committed_tasks, 20)
        self.assertEqual(summaries["adaptive-op-strict"].committed_tasks, 20)
        self.assertEqual(
            summaries["adaptive-op-strict"].operation_policy_counts["pessimistic"],
            20,
        )
        self.assertGreater(
            summaries["adaptive-op-strict"].operation_policy_counts["optimistic"],
            0,
        )

    def test_traditional_k_counts_loser_aborts_while_astra_does_not(self):
        workload = TPCCAgentWorkload(
            TPCCConfig(
                warehouses=1,
                districts_per_warehouse=1,
                customers_per_district=1,
                items=4,
                order_lines=1,
                candidates_per_task=3,
                transaction_mix=(("new_order", 1.0),),
            )
        )
        summaries = {
            (summary.path, summary.strategy): summary
            for summary in run_agent_path_matrix(
                workload,
                ("semantic",),
                paths=("traditional-k", "astra"),
                task_count=2,
                seed=1,
                contention_window=2,
            )
        }
        traditional = summaries[("traditional-k", "semantic")]
        astra = summaries[("astra", "semantic")]

        self.assertEqual(traditional.task_count, 2)
        self.assertEqual(traditional.physical_transactions, 6)
        self.assertEqual(traditional.loser_aborts, 4)
        self.assertEqual(astra.task_count, 2)
        self.assertEqual(astra.physical_transactions, 2)
        self.assertEqual(astra.loser_aborts, 0)


if __name__ == "__main__":
    unittest.main()

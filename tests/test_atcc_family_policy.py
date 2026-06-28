import unittest

from agent.runtime.hybrid import ATCCFamilyPolicyTable
from agent.workloads import TPCCConfig, YCSBConfig, build_agent_workload
from agent.workloads.base import AgentCandidate, AgentOperation, AgentTask


class ATCCFamilyPolicyTests(unittest.TestCase):
    def test_ycsb_read_heavy_hotspot_uses_tictoc_full(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=128,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=3,
                read_weight=0.90,
                update_weight=0.10,
                zipf_theta=0.7,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.50,
            ),
        )
        task = workload.generate_tasks(1, seed=920104)[0]

        decision = ATCCFamilyPolicyTable.default().select_task(
            task,
            workload_kind="ycsb",
        )

        self.assertEqual(decision.selected_strategy, "tictoc-full")
        self.assertEqual(decision.rule, "ycsb-read-heavy-tictoc")
        self.assertLessEqual(decision.signals["write_ratio"], 0.25)

    def test_ycsb_write_hotspot_uses_operation_atcc(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=64,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=3,
                read_weight=0.50,
                update_weight=0.50,
                zipf_theta=0.99,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.75,
            ),
        )
        task = workload.generate_tasks(1, seed=920104)[0]

        decision = ATCCFamilyPolicyTable.default().select_task(
            task,
            workload_kind="ycsb",
        )

        self.assertEqual(decision.selected_strategy, "adaptive-op-strict")
        self.assertEqual(decision.rule, "ycsb-hot-write-atcc")
        self.assertGreaterEqual(decision.signals["write_ratio"], 0.30)

    def test_ycsb_medium_hotspot_write_outlier_stays_read_heavy(self):
        task = AgentTask(
            task_id="medium-write-outlier",
            workload="agent-ycsb-semantic",
            task_type="read-update",
            request="medium hotspot write outlier",
            context={
                "hotspot_fraction": 0.10,
                "hotspot_access_probability": 0.50,
                "hot_record_count": 2,
            },
            candidates=(
                AgentCandidate(
                    "candidate",
                    1.0,
                    (
                        AgentOperation.overwrite("ycsb:record:0:field:0", "v"),
                        AgentOperation.overwrite("ycsb:record:0:field:1", "v"),
                        AgentOperation.read("ycsb:record:0:field:2"),
                        AgentOperation.read("ycsb:record:3:field:0"),
                    ),
                ),
            ),
        )

        decision = ATCCFamilyPolicyTable.default().select_task(
            task,
            workload_kind="ycsb",
        )

        self.assertEqual(decision.selected_strategy, "tictoc-full")
        self.assertEqual(decision.rule, "fallback-traditional")

    def test_ycsb_high_hotspot_write_outlier_uses_operation_atcc(self):
        task = AgentTask(
            task_id="high-hotspot-write-outlier",
            workload="agent-ycsb-semantic",
            task_type="read-update",
            request="high hotspot task with a single hot write",
            context={
                "hotspot_fraction": 0.10,
                "hotspot_access_probability": 0.75,
                "hot_record_count": 2,
            },
            candidates=(
                AgentCandidate(
                    "candidate",
                    1.0,
                    (
                        AgentOperation.overwrite("ycsb:record:0:field:0", "v"),
                        AgentOperation.read("ycsb:record:0:field:1"),
                        AgentOperation.read("ycsb:record:0:field:2"),
                        AgentOperation.read("ycsb:record:3:field:0"),
                    ),
                ),
            ),
        )

        decision = ATCCFamilyPolicyTable.default().select_task(
            task,
            workload_kind="ycsb",
        )

        self.assertEqual(decision.selected_strategy, "adaptive-op-strict")
        self.assertEqual(decision.rule, "ycsb-high-hotspot-write-atcc")
        self.assertLess(decision.signals["write_ratio"], 0.30)

    def test_family_policy_can_load_read_heavy_strategy_from_artifact(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=128,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=3,
                read_weight=0.90,
                update_weight=0.10,
                zipf_theta=0.7,
                hotspot_fraction=0.10,
                hotspot_access_probability=0.50,
            ),
        )
        task = workload.generate_tasks(1, seed=920104)[0]
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "read_heavy_strategy": "mvcc-full",
                "hot_write_strategy": "adaptive-op-strict",
                "fallback_strategy": "tictoc-full",
            }
        )

        decision = policy.select_task(task, workload_kind="ycsb")

        self.assertEqual(decision.selected_strategy, "mvcc-full")
        self.assertEqual(policy.to_dict()["read_heavy_strategy"], "mvcc-full")

    def test_family_policy_can_load_cold_read_heavy_strategy_from_artifact(self):
        workload = build_agent_workload(
            "ycsb",
            "semantic",
            ycsb_config=YCSBConfig(
                record_count=512,
                field_count=10,
                requests_per_task=10,
                candidates_per_task=3,
                read_weight=0.95,
                update_weight=0.05,
                zipf_theta=0.0,
                hotspot_fraction=0.0,
                hotspot_access_probability=0.0,
            ),
        )
        task = workload.generate_tasks(1, seed=920104)[0]
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "read_heavy_strategy": "mvcc-full",
                "cold_read_heavy_strategy": "occ",
                "hot_write_strategy": "adaptive-op-strict",
                "fallback_strategy": "tictoc-full",
            }
        )

        decision = policy.select_task(task, workload_kind="ycsb")

        self.assertEqual(decision.selected_strategy, "occ")
        self.assertEqual(decision.rule, "ycsb-cold-read-heavy")

    def test_non_ycsb_workload_uses_configured_fallback_strategy(self):
        task = AgentTask(
            task_id="tpcc-new-order",
            workload="agent-tpcc-semantic",
            task_type="new_order",
            request="tpcc high conflict new order",
            candidates=(
                AgentCandidate(
                    "candidate",
                    1.0,
                    (
                        AgentOperation.delta(
                            "tpcc:district:0:0:next_order_id",
                            1,
                        ),
                    ),
                ),
            ),
        )
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "fallback_strategy": "adaptive-op-strict",
            }
        )

        decision = policy.select_task(task, workload_kind="tpcc")

        self.assertEqual(decision.selected_strategy, "adaptive-op-strict")
        self.assertEqual(decision.rule, "fallback-traditional")

    def test_tpcc_window_routes_low_contention_to_configured_fast_path(self):
        workload = build_agent_workload(
            "tpcc",
            "semantic",
            tpcc_config=TPCCConfig(
                warehouses=8,
                districts_per_warehouse=5,
                customers_per_district=100,
                items=500,
                order_lines=5,
                transaction_mix=(("new_order", 1.0),),
            ),
        )
        tasks = workload.generate_tasks(40, seed=920104)
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "fallback_strategy": "adaptive-op-strict",
                "tpcc_low_contention_strategy": "occ",
                "tpcc_low_contention_min_distinct_order_counters": 8,
            }
        ).resolve_for_task_window(tasks, workload_kind="tpcc")

        decisions = {
            policy.select_task(task, workload_kind="tpcc").selected_strategy
            for task in tasks
        }

        self.assertEqual(decisions, {"occ"})

    def test_tpcc_window_keeps_high_contention_on_operation_atcc(self):
        workload = build_agent_workload(
            "tpcc",
            "semantic",
            tpcc_config=TPCCConfig(
                warehouses=1,
                districts_per_warehouse=2,
                customers_per_district=40,
                items=100,
                order_lines=10,
                transaction_mix=(("new_order", 1.0),),
            ),
        )
        tasks = workload.generate_tasks(40, seed=920104)
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "fallback_strategy": "adaptive-op-strict",
                "tpcc_low_contention_strategy": "occ",
                "tpcc_low_contention_min_distinct_order_counters": 8,
            }
        ).resolve_for_task_window(tasks, workload_kind="tpcc")

        decisions = {
            policy.select_task(task, workload_kind="tpcc").selected_strategy
            for task in tasks
        }

        self.assertEqual(decisions, {"adaptive-op-strict"})

    def test_adaptive_cold_read_heavy_strategy_uses_tictoc_below_write_threshold(self):
        task = AgentTask(
            task_id="cold-low-write",
            workload="agent-ycsb-semantic",
            task_type="read-update",
            request="cold low-write read-heavy task",
            context={
                "hotspot_fraction": 0.0,
                "hotspot_access_probability": 0.0,
                "hot_record_count": 0,
            },
            candidates=(
                AgentCandidate(
                    "candidate",
                    1.0,
                    (
                        AgentOperation.overwrite("ycsb:record:1:field:0", "v"),
                        AgentOperation.read("ycsb:record:1:field:1"),
                        AgentOperation.read("ycsb:record:1:field:2"),
                        AgentOperation.read("ycsb:record:1:field:3"),
                    ),
                ),
            ),
        )
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "cold_read_heavy_strategy": "adaptive-mvcc-tictoc",
                "adaptive_cold_mvcc_write_ratio_threshold": 0.30,
            }
        )

        decision = policy.select_task(task, workload_kind="ycsb")

        self.assertEqual(decision.selected_strategy, "tictoc-full")
        self.assertEqual(decision.rule, "ycsb-cold-read-heavy-adaptive-tictoc")

    def test_adaptive_cold_read_heavy_strategy_uses_mvcc_at_write_threshold(self):
        task = AgentTask(
            task_id="cold-higher-write",
            workload="agent-ycsb-semantic",
            task_type="read-update",
            request="cold higher-write read-heavy task",
            context={
                "hotspot_fraction": 0.0,
                "hotspot_access_probability": 0.0,
                "hot_record_count": 0,
            },
            candidates=(
                AgentCandidate(
                    "candidate",
                    1.0,
                    (
                        AgentOperation.overwrite("ycsb:record:1:field:0", "v"),
                        AgentOperation.read("ycsb:record:1:field:1"),
                        AgentOperation.read("ycsb:record:1:field:2"),
                        AgentOperation.read("ycsb:record:1:field:3"),
                    ),
                ),
            ),
        )
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "cold_read_heavy_strategy": "adaptive-mvcc-tictoc",
                "adaptive_cold_mvcc_write_ratio_threshold": 0.25,
            }
        )

        decision = policy.select_task(task, workload_kind="ycsb")

        self.assertEqual(decision.selected_strategy, "mvcc-full")
        self.assertEqual(decision.rule, "ycsb-cold-read-heavy-adaptive-mvcc")

    def test_adaptive_hotspot_read_heavy_strategy_uses_tictoc_below_write_threshold(self):
        task = AgentTask(
            task_id="hotspot-low-write",
            workload="agent-ycsb-semantic",
            task_type="read-update",
            request="hotspot low-write read-heavy task",
            context={
                "hotspot_fraction": 0.10,
                "hotspot_access_probability": 0.50,
                "hot_record_count": 2,
            },
            candidates=(
                AgentCandidate(
                    "candidate",
                    1.0,
                    (
                        AgentOperation.overwrite("ycsb:record:0:field:0", "v"),
                        AgentOperation.read("ycsb:record:0:field:1"),
                        AgentOperation.read("ycsb:record:0:field:2"),
                        AgentOperation.read("ycsb:record:3:field:0"),
                    ),
                ),
            ),
        )
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "read_heavy_strategy": "adaptive-mvcc-tictoc",
                "adaptive_hotspot_mvcc_write_ratio_threshold": 0.30,
            }
        )

        decision = policy.select_task(task, workload_kind="ycsb")

        self.assertEqual(decision.selected_strategy, "tictoc-full")
        self.assertEqual(decision.rule, "ycsb-read-heavy-adaptive-tictoc")

    def test_adaptive_hotspot_read_heavy_strategy_uses_mvcc_at_write_threshold(self):
        task = AgentTask(
            task_id="hotspot-higher-write",
            workload="agent-ycsb-semantic",
            task_type="read-update",
            request="hotspot higher-write read-heavy task",
            context={
                "hotspot_fraction": 0.10,
                "hotspot_access_probability": 0.50,
                "hot_record_count": 2,
            },
            candidates=(
                AgentCandidate(
                    "candidate",
                    1.0,
                    (
                        AgentOperation.overwrite("ycsb:record:0:field:0", "v"),
                        AgentOperation.read("ycsb:record:0:field:1"),
                        AgentOperation.read("ycsb:record:0:field:2"),
                        AgentOperation.read("ycsb:record:3:field:0"),
                    ),
                ),
            ),
        )
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "read_heavy_strategy": "adaptive-mvcc-tictoc",
                "adaptive_hotspot_mvcc_write_ratio_threshold": 0.25,
            }
        )

        decision = policy.select_task(task, workload_kind="ycsb")

        self.assertEqual(decision.selected_strategy, "mvcc-full")
        self.assertEqual(decision.rule, "ycsb-read-heavy-adaptive-mvcc")

    def test_adaptive_read_heavy_thresholds_round_trip(self):
        policy = ATCCFamilyPolicyTable.from_dict(
            {
                "cold_read_heavy_strategy": "adaptive-mvcc-tictoc",
                "read_heavy_strategy": "adaptive-mvcc-tictoc",
                "adaptive_cold_mvcc_write_ratio_threshold": 0.07,
                "adaptive_hotspot_mvcc_write_ratio_threshold": 0.13,
            }
        )

        data = policy.to_dict()

        self.assertEqual(data["cold_read_heavy_strategy"], "adaptive-mvcc-tictoc")
        self.assertEqual(data["read_heavy_strategy"], "adaptive-mvcc-tictoc")
        self.assertEqual(data["adaptive_cold_mvcc_write_ratio_threshold"], 0.07)
        self.assertEqual(data["adaptive_hotspot_mvcc_write_ratio_threshold"], 0.13)

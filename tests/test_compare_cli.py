import dataclasses
import contextlib
import io
import json
import random
import threading
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from agent.cc import ConcurrencyControlRegistry, LockConflict, ReservationTable
from agent.benchmarks.mixed import (
    ATCCAdmissionConflict,
    MixedBenchmarkConfig,
    MixedCounters,
    TPCCReplayAdmission,
    admission_failure_reason,
    apply_atcc_experiment_overrides,
    begin_planned_transaction,
    can_defer_transaction_begin,
    can_defer_read_heavy_transaction_begin,
    mixed_transaction_metadata,
    observe_atcc_admission_conflict,
    run_agent_with_deferred_commit_reservation,
    run_agent_with_deferred_commit_lock,
    run_agent_with_deferred_read_optimistic,
    run_agent_with_deferred_read_reservation,
    run_agent_attempt,
    run_mixed_benchmark,
    paper_native_optimistic_fast_path,
    should_serialize_ycsb_observed_replay,
    should_serialize_tpcc_mixed_replay,
    should_use_paper_deferred_replay,
    should_use_ycsb_observed_deferred_replay,
    should_use_ycsb_warmed_low_write_replay,
    ycsb_observed_replay_ready,
    ycsb_observed_native_ready,
    tpcc_replay_admission,
    tpcc_replay_gate_pressure,
)
from agent.benchmarks.phases import PlannedPhase, PlannedTask, ReasoningProfile, plan_task_phases
from agent.cli import compare, matrix, mixed, train_atcc
from agent.cc.atcc.common import ATCCDecision
from agent.cc.atcc.actions import LOCK_HOT_BEFORE_COMMIT, action_spec
from agent.cc.atcc.dynamic import DynamicATCC
from agent.cc.atcc.features import ATCCFeatures, extract_task_features
from agent.cc.atcc.policy import ATCCActionStats, ATCCPolicyTable, ATCCPolicyRow
from agent.cc.atcc.reward import ATCCRewardConfig
from scripts.train_paper_atcc_coordinated import (
    behavior_probabilities,
    behavior_probability,
    coordinated_behavior_distribution,
    normalized_system_scores,
)
from scripts.unified_trace.generate_castdas_trace import (
    paper_trace_workload,
    resolve_background_trace_length,
)
from scripts.unified_trace.run_castdas_trace_fair import (
    FixedCountRunCoordinator,
    TPCCReplayPressureAdmission,
    cached_retry_scheduler_cooldown_s,
    continuous_background_rows,
    high_contention_cached_retry,
    paper_agent_admission_cap,
    paper_background_admission_cap,
    paper_background_batch_size,
    result_row,
    run_background_row,
    run_rows,
    should_reuse_atcc_retry_plan,
)
from scripts.unified_trace.run_dbx1000_trace import write_dbx1000_trace
from scripts.unified_trace.run_unified_trace_matrix import summarize as summarize_unified_trace
from agent.runtime import AgentTransactionManager
from agent.runtime import CompiledPhasePolicy, CompiledPolicyEntry, LockClass
from agent.workloads import AgentOperation, AgentTask, build_workload


class CompareCliTests(unittest.TestCase):
    def test_ycsb_replay_requires_repeated_observed_contention(self):
        manager = AgentTransactionManager()
        self.assertFalse(ycsb_observed_replay_ready(manager))

        for _ in range(3):
            manager.hotness_tracker.observe_contention(
                "row", "validation-failure"
            )

        self.assertTrue(ycsb_observed_replay_ready(manager))

    def test_ycsb_native_path_requires_stable_observed_warmup(self):
        manager = AgentTransactionManager()
        task = build_workload("ycsb", "high", "paper").generate_tasks(
            1, seed=920104
        )[0]
        self.assertFalse(
            ycsb_observed_native_ready(
                manager, task, retry_count=0, background_workers=0
            )
        )

        for index in range(32):
            manager.hotness_tracker.observe_access(f"cold-{index}")

        self.assertTrue(
            ycsb_observed_native_ready(
                manager, task, retry_count=0, background_workers=0
            )
        )
        self.assertFalse(
            ycsb_observed_native_ready(
                manager, task, retry_count=1, background_workers=0
            )
        )

    def test_retry_plan_cache_covers_deterministic_high_contention_traces(self):
        self.assertTrue(
            high_contention_cached_retry(
                MixedBenchmarkConfig(workload="tpcc", level="high", background_workers=0)
            )
        )
        self.assertTrue(
            high_contention_cached_retry(
                MixedBenchmarkConfig(workload="ycsb", level="high", background_workers=8)
            )
        )
        self.assertFalse(
            high_contention_cached_retry(
                MixedBenchmarkConfig(workload="tpcc", level="low", background_workers=0)
            )
        )
        self.assertTrue(
            high_contention_cached_retry(
                MixedBenchmarkConfig(workload="ycsb", level="high", background_workers=0)
            )
        )
        self.assertTrue(
            high_contention_cached_retry(
                MixedBenchmarkConfig(
                    workload="ycsb",
                    level="medium",
                    clients=40,
                    background_workers=0,
                )
            )
        )
        self.assertFalse(
            high_contention_cached_retry(
                MixedBenchmarkConfig(
                    workload="ycsb",
                    level="medium",
                    clients=24,
                    background_workers=0,
                )
            )
        )

    def test_cached_retry_cooldown_desynchronizes_all_high_contention_replays(self):
        rng = random.Random(7)
        all_agent = cached_retry_scheduler_cooldown_s(
            MixedBenchmarkConfig(
                workload="ycsb", level="high", clients=40, background_workers=0
            ),
            rng,
        )
        mixed = cached_retry_scheduler_cooldown_s(
            MixedBenchmarkConfig(
                workload="ycsb", level="high", clients=40, background_workers=8
            ),
            rng,
        )
        medium = cached_retry_scheduler_cooldown_s(
            MixedBenchmarkConfig(
                workload="ycsb", level="medium", clients=40, background_workers=0
            ),
            rng,
        )
        self.assertGreaterEqual(all_agent, 0.5)
        self.assertLessEqual(all_agent, 1.0)
        self.assertGreaterEqual(mixed, 0.005)
        self.assertLessEqual(mixed, 0.020)
        self.assertGreaterEqual(medium, 0.005)
        self.assertLessEqual(medium, 0.020)

    def test_paper_agent_admission_decouples_ycsb_and_caps_saturated_tpcc(self):
        manager = AgentTransactionManager()
        self.assertEqual(
            40,
            paper_agent_admission_cap(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="ycsb", level="high", clients=40, background_workers=0
                ),
                agent_worker_count=40,
            ),
        )
        self.assertEqual(
            26,
            paper_agent_admission_cap(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="ycsb", level="high", clients=32, background_workers=6
                ),
                agent_worker_count=26,
            ),
        )
        self.assertEqual(
            32,
            paper_agent_admission_cap(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="ycsb", level="high", clients=40, background_workers=8
                ),
                agent_worker_count=32,
            ),
        )
        self.assertEqual(
            24,
            paper_agent_admission_cap(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="ycsb", level="high", clients=24, background_workers=0
                ),
                agent_worker_count=24,
            ),
        )
        self.assertEqual(
            32,
            paper_agent_admission_cap(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="tpcc", level="high", clients=40, background_workers=8
                ),
                agent_worker_count=32,
            ),
        )
        self.assertEqual(
            13,
            paper_agent_admission_cap(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="tpcc", level="high", clients=40, background_workers=0
                ),
                agent_worker_count=40,
            ),
        )
        self.assertEqual(
            40,
            paper_agent_admission_cap(
                manager,
                "bamboo",
                MixedBenchmarkConfig(
                    workload="ycsb", level="high", clients=40, background_workers=0
                ),
                agent_worker_count=40,
            ),
        )
        self.assertEqual(
            5,
            paper_background_admission_cap(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="tpcc", level="high", clients=40, background_workers=8
                ),
                background_worker_count=8,
            ),
        )
        self.assertEqual(
            8,
            paper_background_admission_cap(
                manager,
                "bamboo",
                MixedBenchmarkConfig(
                    workload="tpcc", level="high", clients=40, background_workers=8
                ),
                background_worker_count=8,
            ),
        )

    def test_paper_background_batches_only_disjoint_ycsb_low_and_medium(self):
        manager = AgentTransactionManager()
        self.assertEqual(
            40,
            paper_background_batch_size(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="ycsb", level="low", clients=40, background_workers=8
                ),
            ),
        )
        self.assertEqual(
            20,
            paper_background_batch_size(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="ycsb", level="medium", clients=40, background_workers=8
                ),
            ),
        )
        self.assertEqual(
            1,
            paper_background_batch_size(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="ycsb", level="high", clients=40, background_workers=8
                ),
            ),
        )
        self.assertEqual(
            1,
            paper_background_batch_size(
                manager,
                "silo",
                MixedBenchmarkConfig(workload="ycsb", level="low", background_workers=8),
            ),
        )

    def test_paper_deferred_replay_targets_tpcc_high_and_mixed_ycsb(self):
        read = AgentOperation.read("row")
        write = AgentOperation.write("row", "1")

        def plan(workload, level):
            return PlannedTask(
                AgentTask(
                    task_id=f"{workload}-{level}",
                    workload=workload,
                    task_type="update",
                    operations=(read, write),
                    context={"level": level},
                ),
                (
                    PlannedPhase("explore", (read,), 0, (1,)),
                    PlannedPhase("commit", (write,), 0, (1,)),
                ),
            )

        self.assertTrue(
            should_use_paper_deferred_replay(
                plan("tpcc", "high"), background_workers=0
            )
        )
        self.assertTrue(
            should_use_paper_deferred_replay(
                plan("ycsb", "medium"), background_workers=2
            )
        )
        self.assertFalse(
            should_use_paper_deferred_replay(
                plan("ycsb", "high"), background_workers=2
            )
        )
        self.assertFalse(
            should_use_paper_deferred_replay(
                plan("ycsb", "medium"), background_workers=0
            )
        )
        commit_only_ycsb = PlannedTask(
            AgentTask(
                task_id="ycsb-high-commit-only",
                workload="ycsb",
                task_type="update",
                operations=(write,),
                context={
                    "level": "high",
                    "access_distribution": "zipfian",
                    "ycsb_zipf_theta": 0.99,
                    "write_ratio": 0.9,
                },
            ),
            (PlannedPhase("commit", (write,), 0, (1,)),),
        )
        self.assertTrue(
            should_use_paper_deferred_replay(
                commit_only_ycsb, background_workers=0
            )
        )
        self.assertTrue(
            should_serialize_tpcc_mixed_replay(
                plan("tpcc", "high").task,
                background_workers=8,
            )
        )
        self.assertTrue(
            should_serialize_tpcc_mixed_replay(
                plan("tpcc", "high").task,
                background_workers=0,
            )
        )

    def test_ycsb_p0_native_and_p1_observed_deferred_shapes(self):
        def ycsb_task(**context):
            return AgentTask(
                task_id="ycsb-shape",
                workload="ycsb",
                task_type="read-update",
                operations=(
                    AgentOperation.read("row"),
                    AgentOperation.write("row", "1"),
                ),
                context={"level": "high", **context},
            )

        theta_half = ycsb_task(
            access_distribution="zipfian",
            ycsb_zipf_theta=0.5,
            write_ratio=0.5,
        )
        low_write = ycsb_task(
            access_distribution="zipfian",
            ycsb_zipf_theta=0.99,
            write_ratio=0.1,
        )
        hotset = ycsb_task(
            access_distribution="hotspot",
            ycsb_hotset_size=32,
            ycsb_hotspot_access_probability=0.8,
            write_ratio=0.5,
        )
        theta_moderate = ycsb_task(
            access_distribution="zipfian",
            ycsb_zipf_theta=0.8,
            write_ratio=0.5,
        )
        broad_hotset = ycsb_task(
            access_distribution="hotspot",
            ycsb_hotset_size=2048,
            ycsb_hotspot_access_probability=0.8,
            write_ratio=0.5,
        )
        medium_hotset = ycsb_task(
            access_distribution="hotspot",
            ycsb_hotset_size=512,
            ycsb_hotspot_access_probability=0.8,
            write_ratio=0.5,
        )
        self.assertEqual(
            "paper-ycsb-low-risk-native-silo",
            paper_native_optimistic_fast_path(
                theta_half, retry_count=0, background_workers=0
            ),
        )
        self.assertEqual(
            "",
            paper_native_optimistic_fast_path(
                theta_moderate, retry_count=0, background_workers=0
            ),
        )
        self.assertEqual(
            "",
            paper_native_optimistic_fast_path(
                broad_hotset, retry_count=0, background_workers=0
            ),
        )
        self.assertEqual(
            "paper-ycsb-low-risk-native-silo",
            paper_native_optimistic_fast_path(
                low_write, retry_count=0, background_workers=0
            ),
        )
        self.assertEqual(
            "",
            paper_native_optimistic_fast_path(
                low_write, retry_count=1, background_workers=0
            ),
        )
        self.assertTrue(
            should_use_ycsb_observed_deferred_replay(
                hotset, background_workers=0
            )
        )
        self.assertTrue(
            should_use_ycsb_observed_deferred_replay(
                medium_hotset, background_workers=0
            )
        )
        self.assertTrue(
            should_use_ycsb_observed_deferred_replay(
                theta_moderate, background_workers=0
            )
        )
        self.assertTrue(
            should_use_ycsb_observed_deferred_replay(
                broad_hotset, background_workers=0
            )
        )
        self.assertFalse(
            should_use_ycsb_observed_deferred_replay(
                hotset, background_workers=8
            )
        )
        manager = AgentTransactionManager()
        self.assertFalse(
            should_use_ycsb_warmed_low_write_replay(
                manager, low_write, background_workers=0
            )
        )
        self.assertFalse(
            should_serialize_ycsb_observed_replay(manager, retry_count=0)
        )
        manager.hotness_tracker.observe_access("row")
        self.assertTrue(
            should_use_ycsb_warmed_low_write_replay(
                manager, low_write, background_workers=0
            )
        )
        self.assertFalse(
            should_use_ycsb_warmed_low_write_replay(
                manager, theta_half, background_workers=0
            )
        )
        self.assertTrue(
            should_serialize_ycsb_observed_replay(manager, retry_count=0)
        )
        self.assertTrue(
            should_serialize_ycsb_observed_replay(manager, retry_count=1)
        )

    def test_tpcc_replay_admission_is_exact_per_root_and_reports_queue(self):
        manager = AgentTransactionManager()
        task = AgentTask(
            task_id="tpcc-high-root",
            workload="tpcc",
            task_type="payment",
            operations=(),
            context={"level": "high", "warehouse": 7, "district": 3},
        )
        admission = tpcc_replay_admission(manager, task)
        self.assertEqual(admission, tpcc_replay_admission(manager, task))
        self.assertEqual(
            "tpcc:warehouse:7:ytd:commit-replay",
            admission.key,
        )
        new_order = AgentTask(
            task_id="tpcc-high-new-order",
            workload="tpcc",
            task_type="new_order",
            operations=(),
            context={"level": "high", "warehouse": 7, "district": 3},
        )
        self.assertEqual(
            "tpcc:district:7:3:next_order_id:commit-replay",
            tpcc_replay_admission(manager, new_order).key,
        )
        admission.acquire()
        admission.acquire()
        acquired = threading.Event()

        def wait_for_replay():
            admission.acquire()
            acquired.set()
            admission.release()

        waiter = threading.Thread(target=wait_for_replay)
        waiter.start()
        deadline = time.monotonic() + 1.0
        while (
            tpcc_replay_gate_pressure(manager)["waiters"] < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        pressure = tpcc_replay_gate_pressure(manager)
        self.assertEqual(2, pressure["active"])
        self.assertEqual(1, pressure["waiters"])
        self.assertEqual(1, pressure["warehouses"])
        self.assertEqual(2, pressure["roots"])
        admission.release()
        waiter.join(1.0)
        self.assertTrue(acquired.is_set())
        admission.release()
        self.assertEqual(0, tpcc_replay_gate_pressure(manager)["active"])

    def test_tpcc_replay_admission_prefers_priority_and_keeps_fifo_ties(self):
        admission = TPCCReplayAdmission("root", capacity=1)
        admission.acquire()
        order = []

        def wait_for_replay(label, priority):
            admission.acquire(priority=priority)
            order.append(label)
            admission.release()

        low = threading.Thread(target=wait_for_replay, args=("low", 1))
        high = threading.Thread(target=wait_for_replay, args=("high", 9))
        low.start()
        deadline = time.monotonic() + 1.0
        while admission.pressure()[0] < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        high.start()
        deadline = time.monotonic() + 1.0
        while admission.pressure()[0] < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        admission.release()
        high.join(1.0)
        low.join(1.0)
        self.assertEqual(["high", "low"], order)

    def test_tpcc_activity_window_shrinks_from_observed_replay_queue(self):
        manager = AgentTransactionManager()
        admission = TPCCReplayPressureAdmission(
            manager,
            full_limit=40,
            pressure_limit=12,
        )
        admission._limit = 24
        admission._last_adjusted = 0.0
        with mock.patch(
            "scripts.unified_trace.run_castdas_trace_fair.tpcc_replay_gate_pressure",
            return_value={"waiters": 8, "active": 1, "warehouses": 1, "max_waiters": 8},
        ):
            admission._adjust_limit()
        self.assertEqual(20, admission._limit)

    def test_retry_plan_cache_requires_atcc_conflict(self):
        atcc = SimpleNamespace(
            cc_registry=SimpleNamespace(
                resolve=lambda _cc: SimpleNamespace(family="paper-atcc")
            )
        )
        baseline = SimpleNamespace(
            cc_registry=SimpleNamespace(resolve=lambda _cc: SimpleNamespace(family="occ"))
        )
        config = MixedBenchmarkConfig(
            workload="tpcc", level="high", background_workers=0
        )

        self.assertTrue(
            should_reuse_atcc_retry_plan(
                atcc,
                "paper-atcc",
                config,
                {"committed": False, "failure_reason": "version-conflict"},
            )
        )
        self.assertFalse(
            should_reuse_atcc_retry_plan(
                baseline,
                "occ",
                config,
                {"committed": False, "failure_reason": "version-conflict"},
            )
        )
        self.assertFalse(
            should_reuse_atcc_retry_plan(
                atcc,
                "paper-atcc",
                config,
                {"committed": True, "failure_reason": "none"},
            )
        )

    def test_timed_agent_task_tps_excludes_post_window_drain(self):
        counters = MixedCounters()
        counters.completed_agent_tasks = 4
        counters.agent_commits = 4
        counters.steady_agent_commits = 2
        counters.measurement_window_s = 1.0
        counters.agent_drain_s = 4.0

        row = result_row(
            {},
            "occ",
            counters,
            elapsed_s=5.0,
            tokens_per_operation=2703,
            rows=[],
            manager=AgentTransactionManager(),
        )

        self.assertEqual(2.0, row["agent_task_tps"])
        self.assertEqual(2.0, row["agent_tps"])
        self.assertEqual(0.8, row["agent_drain_task_tps"])

    def test_token_attribution_separates_retry_cache_from_reasoning(self):
        counters = MixedCounters(
            completed_agent_tasks=2,
            agent_reasoning_operation_units=20,
            agent_initial_reasoning_operation_units=16,
            agent_retry_reasoning_operation_units=4,
            agent_retry_cache_saved_operation_units=8,
            agent_initial_reasoning_invocations=2,
            agent_retry_reasoning_invocations=1,
            agent_cached_retry_replays=2,
        )
        row = result_row(
            {},
            "paper-atcc",
            counters,
            elapsed_s=1.0,
            tokens_per_operation=10,
            rows=[],
            manager=AgentTransactionManager(),
        )

        self.assertEqual(200, row["agent_total_tokens"])
        self.assertEqual(160, row["agent_initial_reasoning_tokens"])
        self.assertEqual(40, row["agent_retry_reasoning_tokens"])
        self.assertEqual(80, row["agent_retry_cache_saved_tokens"])
        self.assertEqual(280, row["agent_counterfactual_no_cache_tokens"])
        self.assertEqual(140, row["agent_avg_tokens_without_retry_cache"])
        self.assertAlmostEqual(80 / 280, row["agent_retry_cache_savings_ratio"])

    def test_dbx1000_trace_exports_operation_level_agent_delays(self):
        ops = [
            {"kind": "read", "key": 1, "phase": "explore", "delay_ms": 7},
            {"kind": "read", "key": 2, "phase": "refine", "delay_ms": 11},
            {"kind": "write", "key": 3, "phase": "commit", "delay_ms": 13},
        ]
        rows = [
            {
                "worker_id": "0",
                "sequence": "0",
                "client_type": "agent",
                "workload": "ycsb",
                "task_type": "read-update",
                "ops_json": json.dumps(ops),
                "explore_delay_ms": "3",
                "refine_delay_ms": "4",
                "commit_delay_ms": "5",
                "retry_delay_ms": "17",
                "context_json": "{}",
                "tpcc_warehouses": "",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.tsv"
            stats = write_dbx1000_trace(path, rows)
            fields = path.read_text(encoding="utf-8").strip().split("\t")

        self.assertEqual("25", fields[3])
        self.assertEqual("17", fields[4])
        self.assertEqual("18", fields[5])
        self.assertEqual(43, stats["total_agent_delay_ms"])

    def test_planned_transaction_uses_replay_generation_id(self):
        task = AgentTask(
            task_id="logical-task",
            workload="ycsb",
            task_type="read",
            operations=(AgentOperation.read("row"),),
            context={},
        )
        planned = PlannedTask(
            task=task,
            phases=(PlannedPhase("explore", task.operations, 0),),
            retry_delay_ms=0,
        )
        manager = SimpleNamespace(begin=mock.Mock(return_value="txn"))

        result = begin_planned_transaction(
            manager,
            planned,
            {"strategy": "paper-atcc", "transaction_id": "logical-task:g7"},
        )

        self.assertEqual("txn", result)
        self.assertEqual("logical-task:g7", manager.begin.call_args.args[0])

    def test_unified_summary_compares_atcc_with_best_traditional_agent_tps(self):
        common = {
            "workload": "ycsb",
            "workload_variant": "ycsb_high_z099",
            "level": "high",
            "client_mix": "agent80_backend20",
            "clients": "16",
            "agent_ratio": "0.8",
            "source_system": "cast-das-trace-fair",
            "system": "cast-das",
            "status": "ok",
        }
        rows = [
            {
                **common,
                "cc": "occ",
                "cc_label": "OCC",
                "cc_family": "traditional",
                "agent_tps": "6",
                "total_tps": "120",
                "agent_p99_latency_ms": "80",
                "agent_task_completion_rate": "0.7",
                "agent_attempt_abort_rate": "0.3",
            },
            {
                **common,
                "cc": "silo",
                "cc_label": "Silo",
                "cc_family": "traditional",
                "agent_tps": "8",
                "total_tps": "100",
                "agent_p99_latency_ms": "100",
                "agent_task_completion_rate": "0.8",
                "agent_attempt_abort_rate": "0.2",
            },
            {
                **common,
                "cc": "paper-atcc",
                "cc_label": "ATCC",
                "cc_family": "atcc",
                "agent_tps": "10",
                "total_tps": "90",
                "agent_p99_latency_ms": "50",
                "agent_task_completion_rate": "1.0",
                "agent_attempt_abort_rate": "0.1",
            },
        ]

        summary = summarize_unified_trace(rows, run_id="best-traditional")
        atcc = next(row for row in summary if row["cc"] == "paper-atcc")

        self.assertEqual("silo", atcc["best_traditional_cc"])
        self.assertEqual("1.25", atcc["atcc_vs_best_traditional_agent_tps_speedup"])
        self.assertEqual("0.9", atcc["atcc_vs_best_traditional_total_tps_speedup"])
        self.assertEqual("0.2", atcc["atcc_vs_best_traditional_completion_rate_delta"])
        self.assertEqual("-0.1", atcc["atcc_vs_best_traditional_abort_rate_delta"])
        self.assertEqual("0.5", atcc["atcc_vs_best_traditional_p99_reduction"])

    def test_background_trace_pool_is_independent_from_agent_task_count(self):
        self.assertEqual(128, resolve_background_trace_length(1, 128))
        self.assertEqual(4, resolve_background_trace_length(4, 0))

    def test_fixed_count_background_cycles_until_every_agent_worker_finishes(self):
        coordination = FixedCountRunCoordinator(2)
        rows = [{"sequence": 0}, {"sequence": 1}]
        selected = continuous_background_rows(rows, coordination)

        self.assertEqual([0, 1, 0], [next(selected)["sequence"] for _ in range(3)])
        coordination.agent_worker_done()
        self.assertEqual(1, coordination.remaining_agent_workers)
        self.assertEqual(1, next(selected)["sequence"])
        coordination.agent_worker_done()

        self.assertEqual(0, coordination.remaining_agent_workers)
        with self.assertRaises(StopIteration):
            next(selected)

    def test_fixed_count_runner_treats_backend_rows_as_background(self):
        rows = [
            {"worker_id": 0, "sequence": 0, "client_type": "agent", "seed": 7},
            {"worker_id": 1, "sequence": 0, "client_type": "backend", "seed": 7},
        ]
        config = MixedBenchmarkConfig(clients=2, agent_ratio=0.5)
        background_calls = []

        def run_agent(*_args):
            time.sleep(0.01)

        def run_backend(*_args):
            background_calls.append(1)
            time.sleep(0.001)

        with mock.patch(
            "scripts.unified_trace.run_castdas_trace_fair.run_agent_row", run_agent
        ), mock.patch(
            "scripts.unified_trace.run_castdas_trace_fair.run_background_row", run_backend
        ):
            run_rows(SimpleNamespace(), "occ", config, rows, 1)

        self.assertGreater(len(background_calls), 1)

    def test_fixed_trace_ycsb_uses_paper_scale_row_objects(self):
        workload = paper_trace_workload(build_workload("ycsb", "high", "paper"))
        tasks = workload.generate_tasks(8, seed=920104)

        self.assertEqual(1_000_000, workload.config.record_count)
        self.assertEqual(1, workload.config.field_count)
        self.assertTrue(
            all(operation.object_id.endswith(":field:0") for task in tasks for operation in task.operations)
        )
        self.assertTrue(
            all(task.context["record_count"] == 1_000_000 for task in tasks)
        )

    def test_coordinated_operation_stay_has_exact_behavior_probability(self):
        probabilities = behavior_probabilities(((0, 0), (0, 1), (1, 3)))

        self.assertEqual(0.5, behavior_probability(probabilities, "commit", 0, 1))
        self.assertEqual(1.0, behavior_probability(probabilities, "commit", 1, 1))

    def test_coordinated_retry_distribution_expands_inherited_mask(self):
        paths = ((0, 0), (0, 1), (0, 3), (0, 5), (1, 5), (3, 7))

        refine = coordinated_behavior_distribution(
            paths, "refine", 4, 0, stochastic=True
        )
        commit = coordinated_behavior_distribution(
            paths, "commit", 4, 0, stochastic=True
        )
        repeated = coordinated_behavior_distribution(
            paths, "commit", 5, 0, stochastic=False
        )

        self.assertAlmostEqual(4.0 / 6.0, refine[4])
        self.assertAlmostEqual(1.0 / 6.0, refine[5])
        self.assertAlmostEqual(1.0 / 6.0, refine[7])
        self.assertAlmostEqual(0.25, commit[4])
        self.assertAlmostEqual(0.50, commit[5])
        self.assertAlmostEqual(0.25, commit[7])
        self.assertEqual(1.0, repeated[5])

    def test_coordinated_mixed_reward_prioritizes_agent_tps_and_uses_background_tps(self):
        runs = [
            {
                "result": {
                    "agent_task_tps": 100,
                    "total_tps": 100,
                    "agent_attempt_abort_rate": 0,
                    "agent_p99_latency_ms": 100,
                    "background_tps": 0,
                    "background_commit_rate": 1,
                }
            },
            {
                "result": {
                    "agent_task_tps": 0,
                    "total_tps": 100,
                    "agent_attempt_abort_rate": 0,
                    "agent_p99_latency_ms": 100,
                    "background_tps": 100,
                    "background_commit_rate": 0,
                }
            },
        ]

        scores = normalized_system_scores(runs, mixed=True)

        self.assertAlmostEqual(0.35, scores[0] - scores[1])

    def test_registry_exposes_distinct_paper_atcc_runtime(self):
        registry = ConcurrencyControlRegistry()
        strict = registry.resolve("paper-atcc")
        optimized = registry.resolve("paper-atcc-opt")
        oracle = registry.resolve("paper-atcc-oracle")
        self.assertEqual("paper-atcc", strict.family)
        self.assertEqual("paper-atcc", optimized.family)
        self.assertEqual("paper-atcc", oracle.family)
        self.assertFalse(strict.plan(SimpleNamespace()).metadata.get("paper_atcc_optimized", False))
        self.assertTrue(optimized.plan(SimpleNamespace()).metadata["paper_atcc_optimized"])
        self.assertNotIn("paper-atcc-opt", registry.expand("all"))
        self.assertNotIn("paper-atcc-oracle", registry.expand("all"))

    def test_paper_main_hides_future_access_set_and_oracle_retains_it(self):
        task = AgentTask(
            task_id="online-access",
            workload="ycsb",
            task_type="read-write",
            operations=(
                AgentOperation.read("read-row"),
                AgentOperation.write("write-row", "1"),
            ),
            context={"level": "high"},
        )
        planned = PlannedTask(
            task,
            (PlannedPhase("explore", task.operations),),
        )

        online = mixed_transaction_metadata(
            planned,
            retry_count=0,
            background_workers=8,
            strategy="paper-atcc",
        )
        oracle = mixed_transaction_metadata(
            planned,
            retry_count=0,
            background_workers=8,
            strategy="paper-atcc-oracle",
        )

        self.assertEqual("online_observed", online["access_set_visibility"])
        self.assertEqual([], online["planned_write_targets"])
        self.assertEqual("full_trace_oracle", oracle["access_set_visibility"])
        self.assertEqual(["write-row"], oracle["planned_write_targets"])

        manager = mock.Mock()
        manager.begin.return_value = object()
        begin_planned_transaction(manager, planned, online)
        self.assertIsNone(manager.begin.call_args.kwargs["snapshot_object_ids"])
        begin_planned_transaction(manager, planned, oracle)
        self.assertEqual(
            ("read-row", "write-row"),
            manager.begin.call_args.kwargs["snapshot_object_ids"],
        )

    def test_background_row_defers_conflict_to_next_trace_cycle(self):
        first = mock.Mock()
        first.commit.return_value = SimpleNamespace(committed=False)
        second = mock.Mock()
        second.commit.return_value = SimpleNamespace(committed=True)
        manager = SimpleNamespace(
            cc_registry=SimpleNamespace(
                resolve=lambda _cc: SimpleNamespace(family="paper-atcc")
            ),
            atcc_locks=SimpleNamespace(
                background_pre_admission_block=lambda _targets, **_kwargs: None
            ),
            reservations=SimpleNamespace(
                write_guard=lambda *_args, **_kwargs: contextlib.nullcontext()
            ),
            note_background_abort=mock.Mock(),
            begin=mock.Mock(side_effect=(first, second)),
        )
        task = AgentTask(
            task_id="background-row",
            workload="ycsb",
            task_type="update",
            operations=(AgentOperation.write("row", "1"),),
            context={},
        )
        row = {
            "trace_id": "trace",
            "worker_id": 3,
            "sequence": 7,
            "workload": "ycsb",
            "task_type": "update",
            "_context": {},
            "_task": task,
        }
        counters = MixedCounters()
        sleeps = []

        with mock.patch(
            "scripts.unified_trace.run_castdas_trace_fair.sleep_for_reasoning",
            side_effect=lambda value: sleeps.append(value),
        ):
            run_background_row(
                manager,
                "paper-atcc",
                MixedBenchmarkConfig(
                    workload="ycsb",
                    level="high",
                    cc="paper-atcc",
                    background_retry_backoff_min_ms=10,
                    background_retry_backoff_max_ms=30,
                ),
                row,
                random.Random(920106),
                threading.Lock(),
                counters,
            )

        self.assertEqual(1, manager.begin.call_count)
        first.write.assert_called_once_with("row", "1")
        second.write.assert_not_called()
        self.assertEqual(1, counters.background_attempts)
        self.assertEqual(0, counters.background_commits)
        self.assertEqual(1, counters.background_aborts)
        self.assertEqual(1, counters.background_retries)
        self.assertEqual([], sleeps)

    def test_mixed_paper_atcc_uses_phase_policy_actions(self):
        policy = CompiledPhasePolicy(
            [
                CompiledPolicyEntry(phase="refine", action=int(LockClass.COLD_READ)),
                CompiledPolicyEntry(
                    phase="commit",
                    action=int(LockClass.COLD_READ | LockClass.COLD_WRITE),
                ),
            ],
            generation=1,
        )
        report = run_mixed_benchmark(
            MixedBenchmarkConfig(
                workload="ycsb",
                level="low",
                cc="paper-atcc",
                duration_s=0.05,
                clients=2,
                agent_ratio=1.0,
                reasoning_profile="none",
                retry_until_commit=True,
                max_attempts_per_task=2,
                paper_policy=policy,
            )
        )

        row = report["cc_results"][0]
        self.assertGreater(row["agent_attempts"], 0)
        self.assertTrue(any(action.startswith("paper-action-") for action in row["action_counts"]))

    def test_transaction_manager_can_disable_trace_materialization(self):
        manager = AgentTransactionManager(record_traces=False)
        manager.register_object("row", "0", kind="row")
        txn = manager.begin("no-trace")
        txn.write("row", "1")

        result = txn.commit("occ")

        self.assertTrue(result.committed)
        self.assertEqual([], manager.traces())
        self.assertEqual("1", manager.value_of("row"))

    def test_agentic_reasoning_cost_is_independent_of_conflict_level(self):
        profile = ReasoningProfile("agentic", 2.0)
        low_values = [
            profile.delay_ms(level="low", phase=phase, task_id="low-task", attempt=0)
            for phase in ("explore", "refine", "commit")
        ]
        high_values = [
            profile.delay_ms(level="high", phase=phase, task_id="high-task", attempt=0)
            for phase in ("explore", "refine", "commit")
        ]

        self.assertGreaterEqual(sum(low_values), 120)
        self.assertGreaterEqual(sum(high_values), 120)
        self.assertLessEqual(max(low_values), 100)
        self.assertLessEqual(max(high_values), 100)

    def test_agent_phase_split_preserves_sampled_read_order(self):
        reads = tuple(
            AgentOperation.read(f"ycsb:record:{record}:field:0")
            for record in (1, 50, 2, 40)
        )
        task = AgentTask(
            task_id="phase-order",
            workload="ycsb",
            task_type="read-update",
            operations=reads + (AgentOperation.write("ycsb:record:0:field:0", "1"),),
            context={"level": "high", "zipf_theta": 0.99},
        )

        planned = plan_task_phases(
            task,
            attempt=0,
            profile=ReasoningProfile("none", 1.0),
        )
        phases = {phase.name: phase for phase in planned.phases}
        explore_ranks = [int(op.object_id.split(":")[2]) for op in phases["explore"].operations]
        refine_ranks = [int(op.object_id.split(":")[2]) for op in phases["refine"].operations]

        self.assertEqual([1, 50], explore_ranks)
        self.assertEqual([2, 40], refine_ranks)

    def test_paper_reasoning_uses_operation_and_retry_delays(self):
        profile = ReasoningProfile("agentic", 1.0)

        operation_delay = profile.operation_delay_ms(
            level="high",
            phase="refine",
            task_id="task",
            attempt=0,
            operation_index=3,
        )
        retry_delay = profile.retry_delay_ms(level="high", task_id="task", attempt=1)

        self.assertGreaterEqual(operation_delay, 1)
        self.assertLessEqual(operation_delay, 20)
        self.assertGreaterEqual(retry_delay, 500)
        self.assertLessEqual(retry_delay, 5000)

    def test_paper_tpcc_reasoning_delay_is_charged_per_logical_row_tool_call(self):
        task = AgentTask(
            task_id="tpcc-tool-call-delay",
            workload="tpcc",
            task_type="payment",
            operations=(
                AgentOperation.read(
                    "tpcc:customer:1:2:3:balance", phase="explore"
                ),
                AgentOperation.read(
                    "tpcc:customer:1:2:3:status", phase="explore"
                ),
                AgentOperation.read("tpcc:warehouse:1:ytd", phase="explore"),
                AgentOperation.write(
                    "tpcc:customer:1:2:3:balance", "1", phase="commit"
                ),
                AgentOperation.write(
                    "tpcc:customer:1:2:3:payment_count", "1", phase="commit"
                ),
            ),
            context={"level": "low", "profile": "paper"},
        )

        planned = plan_task_phases(
            task,
            attempt=0,
            profile=ReasoningProfile("agentic", 1.0),
        )
        phases = {phase.name: phase for phase in planned.phases}

        self.assertGreater(phases["explore"].operation_delays_ms[0], 0)
        self.assertEqual(0, phases["explore"].operation_delays_ms[1])
        self.assertGreater(phases["explore"].operation_delays_ms[2], 0)
        self.assertGreater(phases["commit"].operation_delays_ms[0], 0)
        self.assertEqual(0, phases["commit"].operation_delays_ms[1])

    def test_reservation_attributes_background_blocking_to_owner(self):
        reservations = ReservationTable()
        owner = SimpleNamespace(started_at=time.perf_counter(), background_blocked_checks=0)

        with reservations.reserve(("hot",), owner=owner, wait=False):
            with self.assertRaises(LockConflict):
                with reservations.write_guard(("hot",), owner=SimpleNamespace(), wait=False):
                    pass

        self.assertEqual(1, owner.background_blocked_checks)

    def test_atcc_reward_penalizes_measured_background_loss(self):
        config = ATCCRewardConfig(background_tps_loss_weight=5.0)
        without_blocking = config.reward(
            committed=True,
            elapsed_ms=100.0,
            lock_wait_ms=0.0,
            wasted_reasoning_ms=0.0,
            background_tps_loss=0.0,
        )
        with_blocking = config.reward(
            committed=True,
            elapsed_ms=100.0,
            lock_wait_ms=0.0,
            wasted_reasoning_ms=0.0,
            background_tps_loss=2.0,
        )

        self.assertEqual(10.0, without_blocking - with_blocking)

    def test_atcc_admission_failure_reason_uses_selected_lock_phase_and_scope(self):
        cause = LockConflict("timeout", ("hot",))
        cases = (
            ("reserve-hot", "reservation-timeout"),
            ("lock-before-commit", "full-commit-lock-timeout"),
            ("lock-hot-before-commit", "hot-commit-lock-timeout"),
            ("lock-write-set", "begin-lock-timeout"),
        )
        for action, expected in cases:
            conflict = ATCCAdmissionConflict(
                cause,
                decision=ATCCDecision(action=action, targets=("hot",)),
                wait_s=0.01,
                background_workers=8,
            )
            self.assertEqual(expected, admission_failure_reason(conflict, action))

    def test_reservation_waiter_blocks_later_background_writer(self):
        reservations = ReservationTable()
        first_writer = SimpleNamespace()
        agent = SimpleNamespace(started_at=time.perf_counter() - 1.0)
        acquired = threading.Event()

        def reserve_agent() -> None:
            with reservations.reserve(
                ("hot",),
                owner=agent,
                wait=True,
                timeout_s=1.0,
                priority=9,
            ):
                acquired.set()

        with reservations.write_guard(("hot",), owner=first_writer, wait=False):
            thread = threading.Thread(target=reserve_agent)
            thread.start()
            deadline = time.perf_counter() + 1.0
            while not reservations._waiters and time.perf_counter() < deadline:
                time.sleep(0.001)
            self.assertTrue(reservations._waiters)
            with self.assertRaises(LockConflict):
                with reservations.write_guard(("hot",), owner=SimpleNamespace(), wait=False):
                    pass
            self.assertFalse(acquired.is_set())

        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired.is_set())
        diagnostics = reservations.snapshot_diagnostics()
        self.assertEqual(1, diagnostics["reservation_waiter_count"])
        self.assertEqual([1], diagnostics["reservation_waiter_target_sizes"])
        self.assertGreater(diagnostics["background_writer_waiter_blocked_checks"], 0)

    def test_reservation_pressure_marks_inconsistent_front_queue_convoy(self):
        reservations = ReservationTable()
        first_writer = SimpleNamespace()
        second_writer = SimpleNamespace()
        first_agent = SimpleNamespace(started_at=time.perf_counter() - 2.0)
        second_agent = SimpleNamespace(started_at=time.perf_counter() - 1.0)

        def reserve_first() -> None:
            with reservations.reserve(("a", "b"), owner=first_agent, wait=True, timeout_s=1.0):
                pass

        def reserve_second() -> None:
            with reservations.reserve(("b", "c"), owner=second_agent, wait=True, timeout_s=1.0):
                pass

        with reservations.write_guard(("a",), owner=first_writer, wait=False):
            with reservations.write_guard(("c",), owner=second_writer, wait=False):
                first_thread = threading.Thread(target=reserve_first)
                second_thread = threading.Thread(target=reserve_second)
                first_thread.start()
                second_thread.start()
                deadline = time.perf_counter() + 1.0
                while len(reservations._waiters) < 3 and time.perf_counter() < deadline:
                    time.sleep(0.001)
                pressure = reservations.snapshot_pressure(("a", "b", "c"))
                self.assertTrue(pressure["reservation_convoy_active"])
                self.assertEqual(3, pressure["reservation_convoy_queue_target_count"])
                self.assertEqual(2, pressure["reservation_convoy_front_waiter_count"])

        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())

    def test_registry_resolves_all_strategies(self):
        registry = ConcurrencyControlRegistry()
        expected = [
            "occ",
            "2pl-nowait",
            "2pl-wait-die",
            "mvcc",
            "silo",
            "tictoc",
            "bamboo",
            "polaris",
            "paper-atcc",
        ]

        self.assertEqual(expected, registry.expand("all"))
        for name in expected:
            self.assertEqual(name, registry.resolve(name).name)
        with self.assertRaises(ValueError):
            registry.resolve("semantic")

    def test_runtime_polaris_uses_retry_priority(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        txn = manager.begin(
            "polaris-priority",
            {"retry_count": 2, "context": {"retry_count": 2}},
            snapshot_object_ids=("row",),
        )
        txn.write("row", "1")

        plan = manager.cc_registry.resolve("polaris").plan(txn)

        self.assertEqual("polaris", plan.strategy)
        self.assertEqual("polaris", plan.family)
        self.assertEqual(("row",), plan.lock_targets)
        self.assertEqual("exclusive", plan.metadata["lock_table"])
        self.assertGreater(plan.metadata["priority"], 0)

    def test_runtime_bamboo_uses_short_write_set_lock(self):
        manager = AgentTransactionManager()
        for object_id in ("read", "write"):
            manager.register_object(object_id, "0", kind="row")
        txn = manager.begin("bamboo-plan", snapshot_object_ids=("read", "write"))
        txn.read("read")
        txn.write("write", "1")

        plan = manager.cc_registry.resolve("bamboo").plan(txn)

        self.assertEqual("bamboo", plan.strategy)
        self.assertEqual("bamboo", plan.family)
        self.assertEqual(("write",), plan.lock_targets)
        self.assertTrue(plan.validate_reads)
        self.assertTrue(plan.validate_writes)
        self.assertTrue(plan.metadata["bamboo_early_retire"])

    def test_mvcc_and_tictoc_adapters_enforce_serializable_read_validation(self):
        manager = AgentTransactionManager()
        manager.register_object("read", "0", kind="row")
        manager.register_object("write", "0", kind="row")

        for strategy in ("mvcc", "tictoc"):
            txn = manager.begin(f"{strategy}-stale")
            txn.read("read")
            txn.write("write", strategy)
            updater = manager.begin(f"{strategy}-updater")
            updater.write("read", f"changed-{strategy}")
            self.assertTrue(updater.commit("occ").committed)

            result = txn.commit(strategy)

            self.assertFalse(result.committed)
            self.assertIn("read", result.conflict_object_ids)

    def test_compare_cli_emits_report(self):
        stdout = io.StringIO()
        status = compare.main(
            [
                "--workload",
                "ycsb",
                "--level",
                "low",
                "--cc",
                "occ,dynamic-atcc",
                "--tasks",
                "4",
                "--workers",
                "2",
                "--reasoning-profile",
                "light",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        self.assertEqual("cc-benchmark", report["mode"])
        self.assertEqual("ycsb", report["workload"])
        self.assertEqual("low", report["level"])
        self.assertEqual(["occ", "dynamic-atcc"], report["strategies"])
        self.assertEqual(2, len(report["cc_results"]))
        self.assertIn("task_commit_rate", report["cc_results"][0])
        self.assertIn("attempt_commit_rate", report["cc_results"][0])
        self.assertIn("commit_rate", report["cc_results"][0])
        self.assertIn("throughput", report["cc_results"][0])
        self.assertIn("abort_count", report["cc_results"][0])
        self.assertIn("p95_latency_ms", report["cc_results"][0])
        self.assertIn("avg_phase_count", report["cc_results"][0])
        self.assertIn("avg_reasoning_delay_ms", report["cc_results"][0])
        self.assertIn("wasted_reasoning_ms", report["cc_results"][0])
        self.assertIn("wasted_elapsed_ms", report["cc_results"][0])
        self.assertIn("avg_lock_hold_ms", report["cc_results"][0])
        self.assertIn("early_abort_count", report["cc_results"][0])
        self.assertIn("skipped_reasoning_ms", report["cc_results"][0])
        self.assertIn("action_counts", report["cc_results"][0])
        self.assertGreater(report["cc_results"][0]["avg_phase_count"], 0)
        self.assertGreaterEqual(report["cc_results"][0]["avg_reasoning_delay_ms"], 0)

    def test_reasoning_profile_none_disables_delay(self):
        report = compare.run_compare(
            workload="ycsb",
            level="high",
            cc="occ",
            tasks=4,
            workers=2,
            retries=0,
            reasoning_profile="none",
            reasoning_scale=1.0,
            seed=920104,
        )

        self.assertEqual("none", report["reasoning_profile"])
        self.assertEqual(0, report["cc_results"][0]["total_reasoning_delay_ms"])
        self.assertEqual(0.0, report["cc_results"][0]["avg_reasoning_delay_ms"])

    def test_paper_workload_profile_matches_agentic_shape(self):
        ycsb_low = build_workload("ycsb", "low", "paper")
        ycsb_medium = build_workload("ycsb", "medium", "paper")
        ycsb = build_workload("ycsb", "high", "paper")
        tpcc = build_workload("tpcc", "low", "paper")
        ycsb_task = ycsb.generate_tasks(1, seed=920104)[0]
        tpcc_tasks = tpcc.generate_tasks(16, seed=920104)

        self.assertEqual(1_000_000, ycsb.config.logical_record_count)
        self.assertEqual(10, ycsb.config.operations_per_task)
        self.assertEqual(0.95, ycsb_low.config.read_weight)
        self.assertEqual(0.05, ycsb_low.config.update_weight)
        self.assertEqual(0.0, ycsb_low.config.zipf_theta)
        self.assertEqual(0.90, ycsb_medium.config.read_weight)
        self.assertEqual(0.10, ycsb_medium.config.update_weight)
        self.assertEqual(0.10, ycsb_medium.config.hotspot_fraction)
        self.assertEqual(0.50, ycsb_medium.config.hotspot_access_probability)
        self.assertEqual(0.50, ycsb.config.read_weight)
        self.assertEqual(0.50, ycsb.config.update_weight)
        self.assertEqual(0.10, ycsb.config.hotspot_fraction)
        self.assertEqual(0.75, ycsb.config.hotspot_access_probability)
        self.assertEqual(64, ycsb.config.record_count)
        self.assertEqual(48, tpcc.config.warehouses)
        self.assertEqual(5, tpcc.config.order_lines)
        self.assertEqual(10, len(ycsb_task.operations))
        self.assertEqual(1_000_000, ycsb_task.context["logical_record_count"])
        self.assertIn("payment", {task.task_type for task in tpcc_tasks})
        self.assertIn("new_order", {task.task_type for task in tpcc_tasks})
        self.assertEqual({"payment", "new_order"}, {name for name, _weight in tpcc.config.transaction_mix})

    def test_fixed_trace_tpcc_uses_standard_logical_scale_and_real_phases(self):
        workload = paper_trace_workload(build_workload("tpcc", "high", "paper"))
        tasks = workload.generate_tasks(32, seed=920104)

        self.assertEqual(10, workload.config.districts_per_warehouse)
        self.assertEqual(3_000, workload.config.customers_per_district)
        self.assertEqual(100_000, workload.config.items)
        self.assertTrue(workload.config.trace_mode)
        self.assertTrue(any(operation.kind == "read" for task in tasks for operation in task.operations))
        self.assertTrue(any(operation.kind == "write" for task in tasks for operation in task.operations))
        self.assertEqual(
            {"explore", "refine", "commit"},
            {
                str(operation.metadata.get("phase"))
                for task in tasks
                for operation in task.operations
            },
        )

    def test_ycsb_zipfian_override_switches_sampling_mode(self):
        default_medium = build_workload("ycsb", "medium", "paper")
        zipf_medium = build_workload("ycsb", "medium", "paper", ycsb_zipf_theta=0.8)
        task = zipf_medium.generate_tasks(1, seed=920104)[0]

        self.assertEqual(0.7, default_medium.config.zipf_theta)
        self.assertEqual("hotspot", default_medium.config.access_distribution)
        self.assertEqual(0.8, zipf_medium.config.zipf_theta)
        self.assertEqual("zipfian", zipf_medium.config.access_distribution)
        self.assertEqual(0.8, task.context["zipf_theta"])
        self.assertEqual("zipfian", task.context["access_distribution"])

    def test_concurrent_benchmark_exposes_occ_conflicts(self):
        report = compare.run_compare(
            workload="tpcc",
            level="high",
            cc="occ",
            tasks=8,
            workers=8,
            retries=0,
            reasoning_profile="agentic",
            reasoning_scale=1.0,
            seed=920104,
        )

        self.assertEqual("cc-benchmark", report["mode"])
        self.assertEqual(["occ"], report["strategies"])
        self.assertGreater(report["cc_results"][0]["abort_count"], 0)
        self.assertLess(report["cc_results"][0]["task_commit_rate"], 1.0)

    def test_retries_can_recover_concurrent_conflicts(self):
        without_retry = compare.run_compare(
            workload="tpcc",
            level="high",
            cc="occ",
            tasks=8,
            workers=8,
            retries=0,
            reasoning_profile="agentic",
            reasoning_scale=1.0,
            seed=920104,
        )
        with_retry = compare.run_compare(
            workload="tpcc",
            level="high",
            cc="occ",
            tasks=8,
            workers=8,
            retries=1,
            reasoning_profile="agentic",
            reasoning_scale=1.0,
            seed=920104,
        )

        self.assertGreaterEqual(
            with_retry["cc_results"][0]["task_commit_rate"],
            without_retry["cc_results"][0]["task_commit_rate"],
        )
        self.assertGreater(with_retry["cc_results"][0]["retry_count"], 0)

    def test_train_atcc_cli_writes_policy_artifact(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "atcc_policy.json"
            status = train_atcc.main(
                [
                    "--workload",
                    "tpcc",
                    "--level",
                    "high",
                    "--episodes",
                    "1",
                    "--tasks",
                    "8",
                    "--workers",
                    "8",
                    "--min-visits",
                    "2",
                    "--output",
                    str(output),
                ],
                stdout=stdout,
            )

            self.assertEqual(0, status)
            report = json.loads(stdout.getvalue())
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("train-atcc", report["mode"])
            self.assertEqual(str(output), report["output"])
            self.assertEqual("cast-das-atcc-policy", artifact["artifact_type"])
            self.assertEqual(2, artifact["version"])
            self.assertEqual(2, artifact["min_visits"])
            self.assertIn("protect_cost_threshold_ms", artifact)
            self.assertIn("reward_config", artifact)
            self.assertIn("rows", artifact)
            self.assertTrue(
                any("agent_cost=" in key and "contention=" in key for key in artifact["rows"])
            )
            self.assertTrue(
                any("actions" in row for row in artifact["rows"].values())
            )

    def test_trained_dynamic_atcc_can_reduce_high_conflict_aborts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "atcc_policy.json"
            train_atcc.main(
                [
                    "--workload",
                    "tpcc",
                    "--level",
                    "high",
                    "--episodes",
                    "2",
                    "--tasks",
                    "8",
                    "--workers",
                    "8",
                    "--min-visits",
                    "2",
                    "--output",
                    str(output),
                ],
                stdout=io.StringIO(),
            )
            report = compare.run_compare(
                workload="tpcc",
                level="high",
                cc="dynamic-atcc",
                tasks=8,
                workers=8,
                retries=0,
                reasoning_profile="agentic",
                reasoning_scale=1.0,
                seed=920104,
                policy=output,
            )

            self.assertGreaterEqual(report["cc_results"][0]["task_commit_rate"], 0.5)

    def test_dynamic_atcc_uses_transaction_start_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "atcc_policy.json"
            train_atcc.main(
                [
                    "--workload",
                    "tpcc",
                    "--level",
                    "high",
                    "--episodes",
                    "1",
                    "--tasks",
                    "8",
                    "--workers",
                    "8",
                    "--min-visits",
                    "2",
                    "--output",
                    str(output),
                ],
                stdout=io.StringIO(),
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))

            for row in artifact["rows"].values():
                if row["protect_visits"] > 0:
                    self.assertEqual(0, row["protect_aborts"])

    def test_compare_defaults_to_frozen_policy_when_policy_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "atcc_policy.json"
            train_atcc.main(
                [
                    "--workload",
                    "tpcc",
                    "--level",
                    "high",
                    "--episodes",
                    "1",
                    "--tasks",
                    "8",
                    "--workers",
                    "8",
                    "--min-visits",
                    "1",
                    "--output",
                    str(output),
                ],
                stdout=io.StringIO(),
            )
            before = json.loads(output.read_text(encoding="utf-8"))
            report = compare.run_compare(
                workload="tpcc",
                level="high",
                cc="dynamic-atcc",
                tasks=4,
                workers=4,
                retries=0,
                reasoning_profile="agentic",
                reasoning_scale=1.0,
                seed=930104,
                policy=output,
            )
            after = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual("eval", report["policy_mode"])
            self.assertEqual(before, after)
            self.assertIn("action_counts", report["cc_results"][0])

    def test_mixed_cli_emits_agent_and_background_metrics(self):
        stdout = io.StringIO()
        status = mixed.main(
            [
                "--workload",
                "tpcc",
                "--level",
                "high",
                "--cc",
                "occ,dynamic-atcc",
                "--duration",
                "0.2",
                "--agents",
                "1",
                "--background",
                "1",
                "--reasoning-profile",
                "light",
                "--reasoning-scale",
                "0.1",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        self.assertEqual("mixed-starvation", report["mode"])
        self.assertEqual(["occ", "dynamic-atcc"], report["strategies"])
        self.assertEqual(2, len(report["cc_results"]))
        self.assertEqual(0, report["clients"])
        self.assertEqual(2703, report["tokens_per_operation"])
        self.assertFalse(report["retry_until_commit"])
        self.assertIn("agent_tps", report["cc_results"][0])
        self.assertIn("agent_task_tps", report["cc_results"][0])
        self.assertIn("agent_task_completion_rate", report["cc_results"][0])
        self.assertIn("agent_abort_rate", report["cc_results"][0])
        self.assertIn("agent_attempt_abort_rate", report["cc_results"][0])
        self.assertIn("agent_p9999_latency_ms", report["cc_results"][0])
        self.assertIn("agent_avg_tokens", report["cc_results"][0])
        self.assertIn("agent_total_tokens", report["cc_results"][0])
        self.assertIn("background_tps", report["cc_results"][0])
        self.assertIn("background_retries", report["cc_results"][0])
        self.assertIn("agent_commit_rate", report["cc_results"][0])
        self.assertIn("guard_wait_ms", report["cc_results"][0])
        self.assertIn("agent_task_guard_wait_ms_p95", report["cc_results"][0])
        self.assertIn("reservation_all_or_nothing_failed_grant_checks", report["cc_results"][0])
        self.assertIn("reservation_front_queue_wait_ms", report["cc_results"][0])
        self.assertIn("background_writer_waiter_blocked_checks", report["cc_results"][0])
        self.assertIn("reserve_read_write_set_hot_target_count_p95", report["cc_results"][0])

    def test_mixed_clients_derives_paper_agent_background_split(self):
        stdout = io.StringIO()
        status = mixed.main(
            [
                "--workload",
                "ycsb",
                "--level",
                "low",
                "--cc",
                "occ",
                "--duration",
                "0.1",
                "--clients",
                "10",
                "--agent-ratio",
                "0.8",
                "--retry-until-commit",
                "--max-attempts-per-task",
                "2",
                "--agent-retry-backoff-ms",
                "1,1",
                "--background-retry-backoff-ms",
                "1,1",
                "--reasoning-profile",
                "none",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        row = report["cc_results"][0]
        self.assertEqual(10, report["clients"])
        self.assertEqual(0.8, report["agent_ratio"])
        self.assertEqual(8, report["agent_workers"])
        self.assertEqual(2, report["background_workers"])
        self.assertTrue(report["retry_until_commit"])
        self.assertEqual([1, 1], report["agent_retry_backoff_ms"])
        self.assertEqual([1, 1], report["background_retry_backoff_ms"])
        self.assertIn("agent_p9999_latency_ms", row)
        self.assertIn("agent_avg_tokens", row)

    def test_mixed_clients_supports_all_agent_split(self):
        stdout = io.StringIO()
        status = mixed.main(
            [
                "--workload",
                "ycsb",
                "--level",
                "low",
                "--cc",
                "occ",
                "--duration",
                "0.1",
                "--clients",
                "4",
                "--agent-ratio",
                "1.0",
                "--reasoning-profile",
                "none",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        row = report["cc_results"][0]
        self.assertEqual(4, report["clients"])
        self.assertEqual(1.0, report["agent_ratio"])
        self.assertEqual(4, report["agent_workers"])
        self.assertEqual(0, report["background_workers"])
        self.assertEqual(0, row["background_attempts"])
        self.assertEqual(0, row["background_tps"])

    def test_mixed_procedure_background_uses_workload_tasks(self):
        stdout = io.StringIO()
        status = mixed.main(
            [
                "--workload",
                "tpcc",
                "--level",
                "medium",
                "--workload-profile",
                "paper",
                "--background-mode",
                "procedure",
                "--cc",
                "occ",
                "--duration",
                "0.2",
                "--agents",
                "1",
                "--background",
                "1",
                "--reasoning-profile",
                "none",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        self.assertEqual("paper", report["workload_profile"])
        self.assertEqual("procedure", report["background_mode"])
        self.assertGreater(report["cc_results"][0]["background_attempts"], 0)
        self.assertGreater(report["cc_results"][0]["background_commits"], 0)
        self.assertGreater(report["cc_results"][0]["background_tps"], 0)

    def test_mixed_train_atcc_cli_writes_reservation_action_stats(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "mixed_atcc_policy.json"
            status = train_atcc.main(
                [
                    "--benchmark",
                    "mixed",
                    "--workload",
                    "tpcc",
                    "--level",
                    "high",
                    "--episodes",
                    "1",
                    "--duration",
                    "2.0",
                    "--clients",
                    "5",
                    "--agent-ratio",
                    "0.8",
                    "--retry-until-commit",
                    "--max-attempts-per-task",
                    "2",
                    "--agent-retry-backoff-ms",
                    "1,1",
                    "--background-retry-backoff-ms",
                    "1,1",
                    "--min-visits",
                    "1",
                    "--reasoning-profile",
                    "none",
                    "--output",
                    str(output),
                ],
                stdout=stdout,
            )

            self.assertEqual(0, status)
            report = json.loads(stdout.getvalue())
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("mixed", report["benchmark"])
            self.assertEqual(5, report["clients"])
            self.assertEqual(4, report["agents"])
            self.assertEqual(1, report["background"])
            self.assertTrue(report["retry_until_commit"])
            self.assertEqual(2703, report["tokens_per_operation"])
            self.assertIn("agent_p9999_latency_ms", report["episodes_detail"][0])
            self.assertIn("agent_avg_tokens", report["episodes_detail"][0])
            self.assertIn("agent_task_tps", report["episodes_detail"][0])
            self.assertEqual("cast-das-atcc-policy", artifact["artifact_type"])
            self.assertEqual(
                [
                    "occ",
                    "write-validate",
                    "reserve-hot",
                    "reserve-hot-rw",
                    "reserve-hot-rw-k",
                    "reserve-read-write-set",
                    "lock-before-commit",
                    "lock-hot-before-commit",
                    "retry-protect",
                ],
                artifact["trainable_actions"],
            )
            self.assertTrue(
                any(
                    "reserve-hot" in row.get("actions", {})
                    for row in artifact["rows"].values()
                )
            )

    def test_dynamic_atcc_uses_read_aware_reservation_for_hot_reads(self):
        atcc = DynamicATCC(policy=ATCCPolicyTable(min_visits=5))
        features = ATCCFeatures(
            workload="ycsb",
            task_type="read-update",
            level="high",
            read_count=6,
            write_count=4,
            hot_write_count=2,
            retry_count=0,
            hot_targets=("ycsb:record:0:field:0", "ycsb:record:1:field:0"),
            hot_read_targets=("ycsb:record:0:field:1", "ycsb:record:1:field:2"),
            write_targets=(
                "ycsb:record:0:field:0",
                "ycsb:record:1:field:0",
                "ycsb:record:2:field:0",
                "ycsb:record:3:field:0",
            ),
            read_targets=(
                "ycsb:record:0:field:1",
                "ycsb:record:1:field:2",
                "ycsb:record:2:field:1",
                "ycsb:record:3:field:1",
                "ycsb:record:4:field:1",
                "ycsb:record:5:field:1",
            ),
            phase_count=3,
            reasoning_delay_ms=180,
        )

        decision = atcc.decide(features)

        self.assertEqual("reserve-read-write-set", decision.action)
        self.assertEqual("read-write-set", decision.lock_scope)
        self.assertEqual("hot-set-reservation", decision.metadata["execution_path"])
        self.assertIn("ycsb:record:0:field:1", decision.targets)
        self.assertIn("ycsb:record:0:field:0", decision.targets)
        self.assertIn(
            decision.metadata["selected_action_source"],
            {"policy", "runtime-hot-read-protect"},
        )

    def test_dynamic_atcc_bounds_full_set_after_reservation_convoy(self):
        atcc = DynamicATCC(policy=ATCCPolicyTable(min_visits=5), hot_rw_k_target_limit=1)
        features = ATCCFeatures(
            workload="ycsb",
            task_type="read-update",
            level="high",
            read_count=6,
            write_count=4,
            hot_write_count=2,
            retry_count=0,
            hot_targets=("ycsb:record:0:field:0", "ycsb:record:1:field:0"),
            hot_read_targets=("ycsb:record:0:field:1", "ycsb:record:1:field:2"),
            write_targets=(
                "ycsb:record:0:field:0",
                "ycsb:record:1:field:0",
                "ycsb:record:2:field:0",
                "ycsb:record:3:field:0",
            ),
            read_targets=(
                "ycsb:record:0:field:1",
                "ycsb:record:1:field:2",
                "ycsb:record:2:field:1",
                "ycsb:record:3:field:1",
                "ycsb:record:4:field:1",
                "ycsb:record:5:field:1",
            ),
            phase_count=3,
            reasoning_delay_ms=180,
            background_workers=6,
            reservation_queue_lengths={"ycsb:record:0:field:0": 4},
            reservation_waiter_count_current=4,
        )

        no_convoy_decision = atcc.decide(features)
        self.assertEqual("reserve-read-write-set", no_convoy_decision.action)
        self.assertFalse(no_convoy_decision.metadata["post_policy_override"])
        self.assertTrue(no_convoy_decision.metadata["background_pressure_background_high"])
        self.assertTrue(no_convoy_decision.metadata["background_pressure_queue_high"])
        self.assertFalse(no_convoy_decision.metadata["reservation_convoy_active"])
        self.assertFalse(no_convoy_decision.metadata["bp_mode_entered"])
        self.assertEqual(0, no_convoy_decision.metadata["bp_mode_windows"])

        features = dataclasses.replace(
            features,
            reservation_convoy_active=True,
            reservation_convoy_queue_target_count=3,
            reservation_convoy_front_waiter_count=2,
            reservation_convoy_pressure=3,
        )
        decision = atcc.decide(features)

        self.assertEqual("reserve-hot-rw-k", decision.action)
        self.assertEqual("hot-rw-k", decision.lock_scope)
        self.assertEqual(1, len(decision.targets))
        self.assertTrue(
            set(decision.targets).issubset(
                {
                    "ycsb:record:0:field:0",
                    "ycsb:record:1:field:0",
                    "ycsb:record:0:field:1",
                    "ycsb:record:1:field:2",
                }
            )
        )
        self.assertEqual("reserve-read-write-set", decision.metadata["pre_override_action"])
        self.assertTrue(decision.metadata["post_policy_override"])
        self.assertEqual(1, decision.metadata["post_policy_override_target_limit"])
        self.assertIn("post-policy-background-pressure", decision.metadata["selected_action_source"])
        self.assertEqual("hot-set-reservation-k", decision.metadata["execution_path"])
        self.assertTrue(decision.metadata["bp_mode_active"])
        self.assertTrue(decision.metadata["bp_mode_entered"])
        self.assertEqual(1, decision.metadata["bp_mode_windows"])
        self.assertEqual(3, decision.metadata["bp_mode_min_windows"])
        self.assertTrue(decision.metadata["reservation_convoy_active"])
        self.assertEqual(3, decision.metadata["reservation_convoy_pressure"])

        retry_decision = atcc.decide(dataclasses.replace(features, retry_count=1))
        self.assertEqual("reserve-hot-rw-k", retry_decision.action)
        self.assertTrue(retry_decision.metadata["post_policy_override"])
        self.assertTrue(retry_decision.metadata["bp_mode_active"])
        self.assertEqual(2, retry_decision.metadata["bp_mode_windows"])
        recovered_features = dataclasses.replace(
            features,
            background_workers=0,
            reservation_queue_lengths={},
            reservation_waiter_count_current=0,
            reservation_convoy_active=False,
            reservation_convoy_queue_target_count=0,
            reservation_convoy_front_waiter_count=0,
            reservation_convoy_pressure=0,
        )
        medium_pressure_decision = atcc.decide(recovered_features)
        self.assertEqual("reserve-hot-rw-k", medium_pressure_decision.action)
        self.assertTrue(medium_pressure_decision.metadata["post_policy_override"])
        self.assertTrue(medium_pressure_decision.metadata["bp_mode_active"])
        self.assertTrue(medium_pressure_decision.metadata["bp_mode_exited_after_decision"])
        self.assertEqual(3, medium_pressure_decision.metadata["bp_mode_windows"])
        recovered_decision = atcc.decide(recovered_features)
        self.assertEqual("reserve-read-write-set", recovered_decision.action)
        self.assertFalse(recovered_decision.metadata["post_policy_override"])
        self.assertFalse(recovered_decision.metadata["bp_mode_active"])

    def test_dynamic_atcc_uses_queue_pressure_aware_hot_rw_k_targets(self):
        atcc = DynamicATCC(policy=ATCCPolicyTable(min_visits=5), hot_rw_k_target_limit=2)
        features = ATCCFeatures(
            workload="ycsb",
            task_type="read-update",
            level="high",
            read_count=6,
            write_count=4,
            hot_write_count=3,
            retry_count=0,
            hot_targets=(
                "ycsb:record:0:field:0",
                "ycsb:record:1:field:0",
                "ycsb:record:2:field:0",
            ),
            hot_read_targets=(
                "ycsb:record:0:field:1",
                "ycsb:record:1:field:1",
            ),
            write_targets=(
                "ycsb:record:0:field:0",
                "ycsb:record:1:field:0",
                "ycsb:record:2:field:0",
                "ycsb:record:3:field:0",
            ),
            read_targets=(
                "ycsb:record:0:field:1",
                "ycsb:record:1:field:1",
                "ycsb:record:2:field:1",
                "ycsb:record:3:field:1",
                "ycsb:record:4:field:1",
                "ycsb:record:5:field:1",
            ),
            phase_count=3,
            reasoning_delay_ms=180,
            background_workers=2,
            reservation_queue_lengths={"ycsb:record:0:field:0": 4},
            reservation_waiter_count_current=2,
            reservation_convoy_active=True,
            reservation_convoy_queue_target_count=2,
            reservation_convoy_front_waiter_count=2,
            reservation_convoy_pressure=2,
            target_selection_seed=12345,
        )

        decision = atcc.decide(features)

        self.assertEqual("reserve-hot-rw-k", decision.action)
        self.assertEqual(2, len(decision.targets))
        self.assertNotIn("ycsb:record:0:field:0", decision.targets)
        self.assertEqual(2, decision.metadata["post_policy_override_target_limit"])
        self.assertTrue(decision.metadata["background_pressure_queue_high"])
        self.assertEqual(4, decision.metadata["reservation_queue_pressure"])
        self.assertTrue(decision.metadata["reservation_convoy_active"])

    def test_dynamic_atcc_trained_hot_commit_lock_uses_hot_scope(self):
        policy = ATCCPolicyTable(min_visits=1)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="new_order",
            level="high",
            read_count=0,
            write_count=6,
            hot_write_count=2,
            retry_count=0,
            hot_targets=("district-hot", "stock-hot"),
            write_targets=("district-hot", "stock-hot", "order-row"),
            background_workers=8,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action=LOCK_HOT_BEFORE_COMMIT))
        row.visits = 3
        row.actions[LOCK_HOT_BEFORE_COMMIT] = ATCCActionStats(visits=3, commits=3, avg_reward=10.0)

        decision = DynamicATCC(policy=policy).decide(features)

        self.assertEqual(LOCK_HOT_BEFORE_COMMIT, decision.action)
        self.assertEqual("hot", decision.lock_scope)
        self.assertEqual("before-commit", decision.lock_phase)
        self.assertEqual(set(features.hot_targets), set(decision.targets))
        self.assertTrue(action_spec(decision.action).locks_before_commit)

    def test_deferred_commit_admission_conflict_retains_selected_action(self):
        manager = AgentTransactionManager()
        hot_target = "tpcc:warehouse:0:district:0:next_order_id"
        manager.register_object(hot_target, "0", kind="row")
        task = AgentTask(
            task_id="commit-admission-conflict",
            workload="tpcc",
            task_type="payment",
            operations=(AgentOperation.write(hot_target, "1"),),
            context={"level": "high"},
        )
        planned = plan_task_phases(task, attempt=0, profile=ReasoningProfile("agentic", 0.0))
        strategy = manager.cc_registry.resolve("dynamic-atcc")
        features = extract_task_features(
            task,
            retry_count=0,
            agentic={
                "phase_count": planned.phase_count,
                "reasoning_delay_ms": planned.total_reasoning_delay_ms,
                "retry_delay_ms": planned.retry_delay_ms,
                "background_workers": 1,
                "reservation_owner_targets": (hot_target,),
            },
        )
        strategy.policy.min_visits = 1
        row = strategy.policy.rows.setdefault(
            features.state_key,
            ATCCPolicyRow(action=LOCK_HOT_BEFORE_COMMIT),
        )
        row.visits = 1
        row.actions[LOCK_HOT_BEFORE_COMMIT] = ATCCActionStats(visits=1, commits=1, avg_reward=1.0)

        with manager.reservations.reserve((hot_target,), owner=object(), wait=False):
            with self.assertRaises(ATCCAdmissionConflict) as raised:
                run_agent_attempt(
                    manager,
                    planned,
                    "dynamic-atcc",
                    ttl_s=0.001,
                    jitter_ms=0,
                    retry_count=0,
                    background_workers=1,
                    config=MixedBenchmarkConfig(),
                )

        self.assertIn(raised.exception.action, {"lock-before-commit", LOCK_HOT_BEFORE_COMMIT})
        self.assertNotEqual("dynamic-atcc", raised.exception.action)
        self.assertEqual(raised.exception.action, raised.exception.decision.action)

    def test_dynamic_atcc_shrinks_full_commit_lock_under_convoy_pressure(self):
        policy = ATCCPolicyTable(min_visits=1)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="new_order",
            level="high",
            read_count=0,
            write_count=6,
            hot_write_count=2,
            retry_count=0,
            hot_targets=("district-hot",),
            write_targets=("district-hot", "order-row", "stock-row", "history-row"),
            background_workers=8,
            reservation_queue_lengths={"district-hot": 4, "order-row": 2},
            reservation_waiter_count_current=4,
            reservation_convoy_active=True,
            reservation_convoy_queue_target_count=2,
            reservation_convoy_front_waiter_count=2,
            reservation_convoy_pressure=2,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="lock-before-commit"))
        row.visits = 3
        row.actions["lock-before-commit"] = ATCCActionStats(visits=3, commits=3, avg_reward=10.0)

        decision = DynamicATCC(policy=policy).decide(features)

        self.assertEqual(LOCK_HOT_BEFORE_COMMIT, decision.action)
        self.assertEqual(features.hot_targets, decision.targets)
        self.assertTrue(decision.metadata["post_policy_override"])

    def test_atcc_state_key_separates_all_agent_and_mixed_clients(self):
        features = ATCCFeatures(
            workload="ycsb",
            task_type="read-update",
            level="high",
            read_count=6,
            write_count=4,
            hot_write_count=2,
            retry_count=0,
            background_workers=0,
        )

        self.assertIn("client_mix=all_agent", features.state_key)
        mixed = dataclasses.replace(features, background_workers=8)
        self.assertIn("client_mix=mixed", mixed.state_key)
        self.assertNotEqual(features.state_key, mixed.state_key)

    def test_atcc_state_key_separates_live_reservation_pressure(self):
        clear = ATCCFeatures(
            workload="tpcc",
            task_type="new_order",
            level="high",
            read_count=0,
            write_count=6,
            hot_write_count=2,
            retry_count=0,
            hot_targets=("district-hot", "stock-hot"),
            write_targets=("district-hot", "stock-hot", "order-row"),
            background_workers=8,
        )
        occupied = dataclasses.replace(clear, reservation_owner_targets=("district-hot",))
        queued = dataclasses.replace(
            occupied,
            reservation_queue_lengths={"district-hot": 2},
            reservation_waiter_count_current=2,
        )
        convoy = dataclasses.replace(
            queued,
            reservation_convoy_active=True,
            reservation_convoy_pressure=2,
        )

        self.assertIn("reservation=clear", clear.state_key)
        self.assertIn("reservation=occupied", occupied.state_key)
        self.assertIn("reservation=queued", queued.state_key)
        self.assertIn("reservation=convoy", convoy.state_key)
        self.assertEqual(4, len({clear.state_key, occupied.state_key, queued.state_key, convoy.state_key}))

    def test_atcc_state_key_separates_previous_failure_reason(self):
        first = ATCCFeatures(
            workload="tpcc",
            task_type="payment",
            level="high",
            read_count=0,
            write_count=3,
            hot_write_count=2,
            retry_count=1,
        )
        reservation = dataclasses.replace(first, previous_failure_reason="reservation-timeout")
        version = dataclasses.replace(first, previous_failure_reason="version-conflict")
        lock = dataclasses.replace(first, previous_failure_reason="full-commit-lock-timeout")

        self.assertIn("last_failure=none", first.state_key)
        self.assertIn("last_failure=reservation-timeout", reservation.state_key)
        self.assertIn("last_failure=version-conflict", version.state_key)
        self.assertIn("last_failure=full-commit-lock-timeout", lock.state_key)
        self.assertEqual(4, len({first.state_key, reservation.state_key, version.state_key, lock.state_key}))

    def test_dynamic_atcc_trusts_trained_optimistic_retry_after_admission_timeout(self):
        policy = ATCCPolicyTable(min_visits=2)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="payment",
            level="high",
            read_count=0,
            write_count=3,
            hot_write_count=2,
            retry_count=1,
            hot_targets=("warehouse-hot", "district-hot"),
            write_targets=("warehouse-hot", "district-hot", "history-row"),
            background_workers=8,
            previous_failure_reason="reservation-timeout",
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="write-validate"))
        row.visits = 10
        row.actions["write-validate"] = ATCCActionStats(
            visits=10,
            commits=9,
            aborts=1,
            avg_reward=20.0,
        )

        decision = DynamicATCC(policy=policy).decide(features)

        self.assertEqual("write-validate", decision.action)
        self.assertEqual("policy-trained", decision.metadata["selected_action_source"])

    def test_mixed_atcc_experiment_overrides_apply_guardrail_and_full_fallback(self):
        task = AgentTask(
            task_id="task-override",
            workload="ycsb",
            task_type="read-update",
            operations=(
                AgentOperation("read", "ycsb:record:0:field:1"),
                AgentOperation("write", "ycsb:record:0:field:0", "v"),
                AgentOperation("write", "ycsb:record:1:field:0", "v"),
            ),
            context={"level": "high", "hot_record_count": 2},
        )
        features = ATCCFeatures(
            workload="ycsb",
            task_type="read-update",
            level="high",
            read_count=1,
            write_count=2,
            hot_write_count=2,
            retry_count=0,
            hot_targets=("ycsb:record:0:field:0", "ycsb:record:1:field:0"),
            hot_read_targets=("ycsb:record:0:field:1",),
            write_targets=("ycsb:record:0:field:0", "ycsb:record:1:field:0"),
            read_targets=("ycsb:record:0:field:1",),
            reservation_queue_lengths={"ycsb:record:0:field:0": 3},
        )
        decision = ATCCDecision(
            action="reserve-hot-rw-k",
            targets=("ycsb:record:0:field:0", "ycsb:record:1:field:0"),
            lock_scope="hot-rw-k",
            lock_phase="reserve",
            metadata={},
        )

        guarded = apply_atcc_experiment_overrides(
            task,
            decision,
            features,
            MixedBenchmarkConfig(atcc_agent_guardrail=True),
            retry_count=0,
        )
        self.assertEqual(("ycsb:record:1:field:0",), guarded.targets)
        self.assertEqual("agent-guardrail", guarded.metadata["mixed_experiment_override"])

        full = apply_atcc_experiment_overrides(
            task,
            decision,
            features,
            MixedBenchmarkConfig(atcc_full_reservation_fallback_ratio=1.0),
            retry_count=0,
        )
        self.assertEqual("reserve-read-write-set", full.action)
        self.assertEqual("read-write-set", full.lock_scope)
        self.assertEqual(
            {
                "ycsb:record:0:field:0",
                "ycsb:record:1:field:0",
                "ycsb:record:0:field:1",
            },
            set(full.targets),
        )
        self.assertEqual("full-reservation-fallback", full.metadata["mixed_experiment_override"])

    def test_dynamic_atcc_prefers_optimistic_first_attempt_for_medium_reads(self):
        policy = ATCCPolicyTable(min_visits=1)
        atcc = DynamicATCC(policy=policy)
        features = ATCCFeatures(
            workload="ycsb",
            task_type="read-update",
            level="medium",
            read_count=8,
            write_count=2,
            hot_write_count=1,
            retry_count=0,
            hot_targets=("ycsb:record:0:field:0",),
            hot_read_targets=("ycsb:record:0:field:1", "ycsb:record:1:field:1"),
            write_targets=("ycsb:record:0:field:0", "ycsb:record:1:field:0"),
            read_targets=(
                "ycsb:record:0:field:1",
                "ycsb:record:1:field:1",
                "ycsb:record:2:field:1",
                "ycsb:record:3:field:1",
                "ycsb:record:4:field:1",
                "ycsb:record:5:field:1",
                "ycsb:record:6:field:1",
                "ycsb:record:7:field:1",
            ),
            reasoning_delay_ms=120,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="reserve-read-write-set"))
        row.visits = 1
        row.actions["reserve-read-write-set"] = ATCCActionStats(
            visits=1,
            commits=1,
            avg_reward=75.0,
        )

        decision = atcc.decide(features)

        self.assertEqual("write-validate", decision.action)
        self.assertEqual("none", decision.lock_scope)
        self.assertEqual("snapshot-write-validate", decision.metadata["execution_path"])
        self.assertEqual("runtime-medium-read-write-validate", decision.metadata["selected_action_source"])

    def test_dynamic_atcc_pure_policy_keeps_policy_action_for_medium_reads(self):
        policy = ATCCPolicyTable(min_visits=1)
        atcc = DynamicATCC(policy=policy, runtime_guards_enabled=False)
        features = ATCCFeatures(
            workload="ycsb",
            task_type="read-update",
            level="medium",
            read_count=8,
            write_count=2,
            hot_write_count=1,
            retry_count=0,
            hot_targets=("ycsb:record:0:field:0",),
            hot_read_targets=("ycsb:record:0:field:1", "ycsb:record:1:field:1"),
            write_targets=("ycsb:record:0:field:0", "ycsb:record:1:field:0"),
            read_targets=(
                "ycsb:record:0:field:1",
                "ycsb:record:1:field:1",
                "ycsb:record:2:field:1",
                "ycsb:record:3:field:1",
                "ycsb:record:4:field:1",
                "ycsb:record:5:field:1",
                "ycsb:record:6:field:1",
                "ycsb:record:7:field:1",
            ),
            reasoning_delay_ms=120,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="reserve-read-write-set"))
        row.visits = 1
        row.actions["reserve-read-write-set"] = ATCCActionStats(
            visits=1,
            commits=1,
            avg_reward=75.0,
        )

        decision = atcc.decide(features)

        self.assertEqual("reserve-read-write-set", decision.action)
        self.assertEqual("read-write-set", decision.lock_scope)
        self.assertEqual("policy", decision.metadata["selected_action_source"])
        self.assertFalse(decision.metadata["runtime_guards_enabled"])

    def test_dynamic_atcc_pure_policy_uses_only_learned_priority(self):
        policy = ATCCPolicyTable(min_visits=1, low_conflict_occ_guard=False)
        atcc = DynamicATCC(policy=policy, runtime_guards_enabled=False)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="new_order",
            level="high",
            read_count=4,
            write_count=8,
            hot_write_count=4,
            retry_count=3,
            hot_targets=("tpcc:district:0:0:next_order_id",),
            write_targets=("tpcc:district:0:0:next_order_id",),
            reasoning_delay_ms=500,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="lock-before-commit"))
        row.visits = 1
        row.priority = 2

        decision = atcc.decide(features)

        self.assertEqual(2, decision.priority)
        self.assertEqual("policy-row", decision.metadata["priority_reason"])

    def test_atcc_policy_learns_and_round_trips_admission_yield(self):
        policy = ATCCPolicyTable(
            min_visits=1,
            low_conflict_occ_guard=False,
            sparse_state_risk_prior=False,
            admission_yield_candidates_ms=(0, 2),
        )
        state_key = "workload=ycsb|level=high|client_mix=mixed"
        for yield_ms, elapsed_ms in ((0, 80.0), (2, 20.0)):
            policy.observe(
                state_key,
                action="occ",
                committed=True,
                elapsed_ms=elapsed_ms,
                admission_yield_ms=yield_ms,
            )

        self.assertEqual(2, policy.admission_yield_for(state_key))
        restored = ATCCPolicyTable.from_dict(policy.to_dict())
        self.assertEqual(2, restored.admission_yield_for(state_key))

    def test_mixed_pure_policy_does_not_use_low_conflict_fast_path(self):
        policy = ATCCPolicyTable(min_visits=1, low_conflict_occ_guard=False)
        policy.set_mode("eval")
        result = run_mixed_benchmark(
            MixedBenchmarkConfig(
                workload="ycsb",
                level="low",
                cc="dynamic-atcc",
                duration_s=0.05,
                clients=2,
                agent_ratio=1.0,
                reasoning_profile="none",
                retry_until_commit=True,
                max_attempts_per_task=2,
                policy=policy,
                policy_mode="eval",
                atcc_pure_policy=True,
            )
        )

        row = result["cc_results"][0]
        actions = row["action_counts"]
        self.assertNotIn("low-conflict-optimistic", actions)

    def test_dynamic_atcc_write_validate_skips_read_validation_only(self):
        manager = AgentTransactionManager()
        for object_id in ("read-row", "write-row"):
            manager.register_object(object_id, "0", kind="row")
        preplan = {
            "action": "write-validate",
            "targets": (),
            "priority": 0,
            "state_key": "test-state",
            "reason": "test",
            "lock_scope": "none",
            "lock_phase": "none",
            "metadata": {},
        }

        txn = manager.begin("read-stale", {"retry_count": 0, "atcc_preplan": preplan})
        txn.read("read-row")
        txn.write("write-row", "1")
        background = manager.begin("bg-read")
        background.write("read-row", "changed")
        self.assertTrue(background.commit("occ").committed)

        result = txn.commit("dynamic-atcc")

        self.assertTrue(result.committed)
        trace = manager.traces()[-1]
        validate_events = [event for event in trace["events"] if event["kind"] == "validate"]
        self.assertFalse(validate_events[-1]["detail"]["validate_reads"])
        self.assertTrue(validate_events[-1]["detail"]["validate_writes"])

        conflict_txn = manager.begin("write-stale", {"retry_count": 0, "atcc_preplan": preplan})
        conflict_txn.write("write-row", "2")
        background = manager.begin("bg-write")
        background.write("write-row", "changed-again")
        self.assertTrue(background.commit("occ").committed)

        conflict_result = conflict_txn.commit("dynamic-atcc")

        self.assertFalse(conflict_result.committed)
        self.assertEqual(("write-row",), conflict_result.conflict_object_ids)

    def test_mixed_deferred_read_reservation_replays_reads_after_begin(self):
        manager = AgentTransactionManager()
        for object_id in ("a", "b", "c"):
            manager.register_object(object_id, "0", kind="row")
        task = AgentTask(
            task_id="read-heavy",
            workload="ycsb",
            task_type="read-update",
            operations=(
                AgentOperation.read("a"),
                AgentOperation.read("b"),
                AgentOperation.write("c", "1"),
            ),
            context={"level": "medium"},
        )
        planned = PlannedTask(
            task=task,
            phases=(
                PlannedPhase("explore", (task.operations[0],), reasoning_delay_ms=0),
                PlannedPhase("refine", (task.operations[1],), reasoning_delay_ms=0),
                PlannedPhase("commit", (task.operations[2],), reasoning_delay_ms=1),
            ),
        )
        decision = dataclasses.make_dataclass(
            "Decision",
            [
                ("action", str),
                ("targets", tuple),
                ("priority", int),
                ("state_key", str),
                ("reason", str),
                ("lock_scope", str),
                ("lock_phase", str),
                ("metadata", dict),
            ],
        )(
            "reserve-read-write-set",
            ("a", "b", "c"),
            0,
            "test-state",
            "test",
            "read-write-set",
            "reserve",
            {},
        )

        self.assertTrue(can_defer_read_heavy_transaction_begin(planned))
        result, action, _wait_s = run_agent_with_deferred_read_reservation(
            manager,
            planned,
            "dynamic-atcc",
            mixed_transaction_metadata(
                planned,
                retry_count=0,
                background_workers=1,
                strategy="dynamic-atcc",
                decision=decision,
            ),
            decision=decision,
            background_workers=1,
            ttl_s=1.0,
        )

        self.assertTrue(result["committed"])
        self.assertEqual("reserve-read-write-set", action)
        trace = manager.traces()[-1]
        self.assertEqual({"a", "b"}, set(trace["read_set"]))
        self.assertEqual({"c"}, set(trace["write_set"]))
        self.assertTrue(trace["metadata"]["atcc_runtime"]["deferred_read_begin"])
        self.assertEqual(1.0, trace["metadata"]["atcc_runtime"]["deferred_commit_reasoning_ms"])

    def test_mixed_deferred_commit_lock_replays_writes_after_reasoning(self):
        manager = AgentTransactionManager()
        manager.register_object("c", "0", kind="row")
        task = AgentTask(
            task_id="write-heavy",
            workload="tpcc",
            task_type="new_order",
            operations=(AgentOperation.write("c", "1"),),
            context={"level": "medium"},
        )
        planned = PlannedTask(
            task=task,
            phases=(
                PlannedPhase("plan", (), reasoning_delay_ms=1),
                PlannedPhase("commit", task.operations, reasoning_delay_ms=1),
            ),
        )
        decision = dataclasses.make_dataclass(
            "LockDecision",
            [
                ("action", str),
                ("targets", tuple),
                ("priority", int),
                ("state_key", str),
                ("reason", str),
                ("lock_scope", str),
                ("lock_phase", str),
                ("metadata", dict),
            ],
        )("lock-before-commit", ("c",), 0, "test-state", "test", "write-set", "before-commit", {})

        self.assertTrue(can_defer_transaction_begin(planned))
        result, action, _wait_s = run_agent_with_deferred_commit_lock(
            manager,
            planned,
            "dynamic-atcc",
            mixed_transaction_metadata(
                planned,
                retry_count=0,
                background_workers=1,
                strategy="dynamic-atcc",
                decision=decision,
            ),
            decision=decision,
            background_workers=1,
            ttl_s=1.0,
        )

        self.assertTrue(result["committed"])
        self.assertEqual("lock-before-commit", action)
        trace = manager.traces()[-1]
        self.assertEqual({"c"}, set(trace["write_set"]))
        runtime = trace["metadata"]["atcc_runtime"]
        self.assertEqual(1.0, runtime["deferred_before_begin_ms"])
        self.assertEqual(1.0, runtime["deferred_commit_reasoning_ms"])
        self.assertLess(runtime["background_aborts"], 0.1)

    def test_mixed_deferred_commit_reservation_admits_before_reasoning(self):
        manager = AgentTransactionManager()
        manager.register_object("c", "0", kind="row")
        task = AgentTask(
            task_id="write-heavy",
            workload="tpcc",
            task_type="new_order",
            operations=(AgentOperation.write("c", "1"),),
            context={"level": "high"},
        )
        planned = PlannedTask(
            task=task,
            phases=(
                PlannedPhase("plan", (), reasoning_delay_ms=50),
                PlannedPhase("commit", task.operations, reasoning_delay_ms=50),
            ),
        )
        decision = dataclasses.make_dataclass(
            "ReservationDecision",
            [
                ("action", str),
                ("targets", tuple),
                ("priority", int),
                ("state_key", str),
                ("reason", str),
                ("lock_scope", str),
                ("lock_phase", str),
                ("metadata", dict),
            ],
        )("reserve-hot", ("c",), 0, "test-state", "test", "hot", "reserve", {})

        self.assertTrue(can_defer_transaction_begin(planned))
        writer = SimpleNamespace()
        started = time.perf_counter()
        with manager.reservations.write_guard(("c",), owner=writer, wait=False):
            with self.assertRaises(ATCCAdmissionConflict) as raised:
                run_agent_with_deferred_commit_reservation(
                    manager,
                    planned,
                    "dynamic-atcc",
                    mixed_transaction_metadata(
                        planned,
                        retry_count=0,
                        background_workers=1,
                        strategy="dynamic-atcc",
                        decision=decision,
                    ),
                    decision=decision,
                    background_workers=1,
                    ttl_s=0.001,
                )
        action, wait_s, diagnostics = observe_atcc_admission_conflict(
            manager,
            "dynamic-atcc",
            raised.exception,
        )
        self.assertEqual("reserve-hot", action)
        self.assertGreaterEqual(wait_s, 0.0)
        self.assertEqual("reserve-hot", diagnostics["action"])
        row = manager.cc_registry.resolve("dynamic-atcc").policy.rows["test-state"]
        self.assertEqual(1, row.aborts)
        self.assertEqual(1, row.actions["reserve-hot"].aborts)
        self.assertLess((time.perf_counter() - started) * 1000.0, 50.0)

        result, action, _wait_s = run_agent_with_deferred_commit_reservation(
            manager,
            planned,
            "dynamic-atcc",
            mixed_transaction_metadata(
                planned,
                retry_count=0,
                background_workers=1,
                strategy="dynamic-atcc",
                decision=decision,
            ),
            decision=decision,
            background_workers=1,
            ttl_s=1.0,
        )

        self.assertTrue(result["committed"])
        self.assertEqual("reserve-hot", action)
        runtime = manager.traces()[-1]["metadata"]["atcc_runtime"]
        self.assertTrue(runtime["reservation_before_reasoning"])
        self.assertEqual(50.0, runtime["deferred_before_begin_ms"])
        self.assertEqual(50.0, runtime["deferred_commit_reasoning_ms"])

    def test_mixed_deferred_read_optimistic_replays_reads_after_begin(self):
        manager = AgentTransactionManager()
        for object_id in ("a", "b", "c"):
            manager.register_object(object_id, "0", kind="row")
        task = AgentTask(
            task_id="read-heavy-occ",
            workload="ycsb",
            task_type="read-update",
            operations=(
                AgentOperation.read("a"),
                AgentOperation.read("b"),
                AgentOperation.write("c", "1"),
            ),
            context={"level": "medium"},
        )
        planned = PlannedTask(
            task=task,
            phases=(
                PlannedPhase("explore", (task.operations[0],), reasoning_delay_ms=0),
                PlannedPhase("refine", (task.operations[1],), reasoning_delay_ms=0),
                PlannedPhase("commit", (task.operations[2],), reasoning_delay_ms=0),
            ),
        )
        decision = dataclasses.make_dataclass(
            "OccDecision",
            [
                ("action", str),
                ("targets", tuple),
                ("priority", int),
                ("state_key", str),
                ("reason", str),
                ("lock_scope", str),
                ("lock_phase", str),
                ("metadata", dict),
            ],
        )("occ", (), 0, "test-state", "test", "none", "none", {})

        result, action, wait_s = run_agent_with_deferred_read_optimistic(
            manager,
            planned,
            "dynamic-atcc",
            mixed_transaction_metadata(
                planned,
                retry_count=0,
                background_workers=1,
                strategy="dynamic-atcc",
                decision=decision,
            ),
            decision=decision,
        )

        self.assertTrue(result["committed"])
        self.assertEqual("occ", action)
        self.assertEqual(0.0, wait_s)
        trace = manager.traces()[-1]
        self.assertEqual({"a", "b"}, set(trace["read_set"]))
        self.assertEqual({"c"}, set(trace["write_set"]))
        self.assertTrue(trace["metadata"]["atcc_runtime"]["deferred_read_begin"])

    def test_mixed_low_conflict_atcc_runtime_fast_path_bypasses_preplan(self):
        manager = AgentTransactionManager()
        for object_id in ("a", "b"):
            manager.register_object(object_id, "0", kind="row")
        task = AgentTask(
            task_id="low-fast-path",
            workload="tpcc",
            task_type="read-update",
            operations=(
                AgentOperation.read("a"),
                AgentOperation.write("b", "1"),
            ),
            context={"level": "low"},
        )
        planned = PlannedTask(
            task=task,
            phases=(
                PlannedPhase("explore", (task.operations[0],), reasoning_delay_ms=0),
                PlannedPhase("commit", (task.operations[1],), reasoning_delay_ms=0),
            ),
        )

        result, action, wait_s, diagnostics = run_agent_attempt(
            manager,
            planned,
            "dynamic-atcc",
            ttl_s=1.0,
            jitter_ms=0,
            retry_count=0,
            background_workers=0,
            config=MixedBenchmarkConfig(),
        )

        self.assertTrue(result["committed"])
        self.assertEqual("occ", action)
        self.assertEqual(0.0, wait_s)
        self.assertEqual("low-conflict-optimistic", diagnostics["runtime_fast_path"])
        trace = manager.traces()[-1]
        self.assertEqual("occ", trace["result"]["strategy"])
        self.assertEqual("low-conflict-optimistic", trace["metadata"]["atcc_runtime_fast_path"])
        self.assertNotIn("atcc_preplan", trace["metadata"])

        retry_result, retry_action, _retry_wait_s, retry_diagnostics = run_agent_attempt(
            manager,
            planned,
            "dynamic-atcc",
            ttl_s=1.0,
            jitter_ms=0,
            retry_count=1,
            background_workers=0,
            config=MixedBenchmarkConfig(),
        )

        self.assertTrue(retry_result["committed"])
        self.assertEqual("occ", retry_action)
        self.assertEqual("low-conflict-optimistic", retry_diagnostics["runtime_fast_path"])
        retry_trace = manager.traces()[-1]
        self.assertEqual("low-conflict-optimistic", retry_trace["metadata"]["atcc_runtime_fast_path"])
        self.assertNotIn("atcc_preplan", retry_trace["metadata"])

    def test_mixed_ycsb_low_conflict_uses_optimistic_silo_fast_path(self):
        manager = AgentTransactionManager()
        for object_id in ("a", "b"):
            manager.register_object(object_id, "0", kind="row")
        task = AgentTask(
            task_id="ycsb-low-optimistic",
            workload="ycsb",
            task_type="read-update",
            operations=(
                AgentOperation.read("a"),
                AgentOperation.write("b", "1"),
            ),
            context={"level": "low"},
        )
        planned = PlannedTask(
            task=task,
            phases=(
                PlannedPhase("explore", (task.operations[0],), reasoning_delay_ms=0),
                PlannedPhase("commit", (task.operations[1],), reasoning_delay_ms=0),
            ),
        )

        result, action, wait_s, diagnostics = run_agent_attempt(
            manager,
            planned,
            "dynamic-atcc",
            ttl_s=1.0,
            jitter_ms=0,
            retry_count=0,
            background_workers=0,
            config=MixedBenchmarkConfig(),
        )

        self.assertTrue(result["committed"])
        self.assertEqual("occ", action)
        self.assertEqual(0.0, wait_s)
        self.assertEqual("low-conflict-optimistic", diagnostics["runtime_fast_path"])
        trace = manager.traces()[-1]
        self.assertEqual("silo", trace["result"]["strategy"])
        self.assertEqual("low-conflict-optimistic", trace["metadata"]["atcc_runtime_fast_path"])
        self.assertEqual("silo", trace["metadata"]["atcc_runtime_fast_path_commit_strategy"])
        self.assertNotIn("atcc_preplan", trace["metadata"])

    def test_paper_atcc_low_first_attempt_uses_native_optimistic_runtime(self):
        manager = AgentTransactionManager()
        root = "tpcc:district:7:1:next_order_id"
        for object_id in (root, "b"):
            manager.register_object(object_id, "0", kind="row")
        task = AgentTask(
            task_id="paper-low-native",
            workload="tpcc",
            task_type="read-update",
            operations=(
                AgentOperation.read(root),
                AgentOperation.write(root, "1"),
            ),
            context={"level": "low"},
        )
        planned = PlannedTask(
            task=task,
            phases=(
                PlannedPhase("explore", (task.operations[0],), reasoning_delay_ms=0),
                PlannedPhase("commit", (task.operations[1],), reasoning_delay_ms=0),
            ),
        )

        result, action, wait_s, diagnostics = run_agent_attempt(
            manager,
            planned,
            "paper-atcc",
            ttl_s=1.0,
            jitter_ms=0,
            retry_count=0,
            background_workers=0,
            config=MixedBenchmarkConfig(atcc_pure_policy=True),
            transaction_id="paper-low-logical",
        )

        self.assertTrue(result["committed"])
        self.assertEqual("paper-action-0", action)
        self.assertEqual(0.0, wait_s)
        self.assertEqual("paper-low-native-occ", diagnostics["runtime_fast_path"])
        self.assertFalse(manager.paper_versioning_enabled)
        trace = manager.traces()[-1]
        self.assertEqual("occ", trace["result"]["strategy"])
        self.assertEqual("paper-atcc", trace["metadata"]["atcc_reported_strategy"])
        self.assertTrue(trace["metadata"]["paper_atcc_retry_feedback"])
        self.assertEqual("paper-low-logical", trace["task_id"])

    def test_paper_atcc_medium_all_agent_first_attempt_uses_native_optimistic(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        task = AgentTask(
            task_id="paper-medium-all-agent-native",
            workload="ycsb",
            task_type="read-update",
            operations=(AgentOperation.read("row"),),
            context={"level": "medium"},
        )
        planned = PlannedTask(
            task=task,
            phases=(PlannedPhase("explore", task.operations),),
        )

        result, action, _wait_s, diagnostics = run_agent_attempt(
            manager,
            planned,
            "paper-atcc",
            ttl_s=1.0,
            jitter_ms=0,
            retry_count=0,
            background_workers=0,
            config=MixedBenchmarkConfig(atcc_pure_policy=True),
        )

        self.assertTrue(result["committed"])
        self.assertEqual("paper-action-0", action)
        self.assertEqual(
            "paper-medium-all-agent-native-optimistic",
            diagnostics["runtime_fast_path"],
        )
        self.assertFalse(manager.paper_versioning_enabled)

    def test_paper_atcc_tpcc_low_mixed_uses_native_optimistic_runtime(self):
        manager = AgentTransactionManager()
        manager.register_object("row", "0", kind="row")
        task = AgentTask(
            task_id="paper-tpcc-low-mixed",
            workload="tpcc",
            task_type="payment",
            operations=(AgentOperation.read("row"),),
            context={"level": "low"},
        )
        planned = PlannedTask(
            task=task,
            phases=(PlannedPhase("explore", task.operations),),
        )

        result, _action, _wait_s, diagnostics = run_agent_attempt(
            manager,
            planned,
            "paper-atcc",
            ttl_s=1.0,
            jitter_ms=0,
            retry_count=0,
            background_workers=1,
            config=MixedBenchmarkConfig(atcc_pure_policy=True),
        )

        self.assertTrue(result["committed"])
        self.assertEqual("paper-low-native-occ", diagnostics["runtime_fast_path"])
        self.assertFalse(manager.paper_versioning_enabled)

    def test_mixed_ycsb_medium_uses_write_validate_mvcc_fast_path(self):
        manager = AgentTransactionManager()
        for object_id in ("a", "b", "c"):
            manager.register_object(object_id, "0", kind="row")
        task = AgentTask(
            task_id="ycsb-medium-write-validate",
            workload="ycsb",
            task_type="read-update",
            operations=(
                AgentOperation.read("a"),
                AgentOperation.read("b"),
                AgentOperation.write("c", "1"),
            ),
            context={"level": "medium"},
        )
        planned = PlannedTask(
            task=task,
            phases=(
                PlannedPhase("explore", (task.operations[0], task.operations[1]), reasoning_delay_ms=0),
                PlannedPhase("commit", (task.operations[2],), reasoning_delay_ms=0),
            ),
        )

        result, action, wait_s, diagnostics = run_agent_attempt(
            manager,
            planned,
            "dynamic-atcc",
            ttl_s=1.0,
            jitter_ms=0,
            retry_count=0,
            background_workers=0,
            config=MixedBenchmarkConfig(),
        )

        self.assertTrue(result["committed"])
        self.assertEqual("write-validate", action)
        self.assertEqual(0.0, wait_s)
        self.assertEqual("ycsb-medium-write-validate", diagnostics["runtime_fast_path"])
        trace = manager.traces()[-1]
        self.assertEqual("mvcc", trace["result"]["strategy"])
        self.assertEqual("ycsb-medium-write-validate", trace["metadata"]["atcc_runtime_fast_path"])
        self.assertEqual("mvcc", trace["metadata"]["atcc_runtime_fast_path_commit_strategy"])
        self.assertTrue(trace["metadata"]["atcc_runtime"]["deferred_read_begin"])
        self.assertNotIn("atcc_preplan", trace["metadata"])

    def test_low_conflict_policy_guard_keeps_occ(self):
        policy = ATCCPolicyTable(min_visits=2)
        state_key = "workload=tpcc|task=new_order|level=low|contention=hot|agent_cost=short|write_set=large|retry=first"
        row = policy.rows.setdefault(state_key, ATCCPolicyRow())
        row.actions["occ"] = ATCCActionStats(visits=3, commits=3, aborts=0, avg_reward=80.0)
        row.actions["reserve-hot"] = ATCCActionStats(visits=3, commits=3, aborts=0, avg_reward=95.0)

        policy._refresh_decision(state_key, row)

        self.assertEqual("occ", row.action)
        self.assertEqual(0, row.priority)

    def test_dynamic_atcc_uses_write_validate_for_low_conflict_fast_path(self):
        policy = ATCCPolicyTable(min_visits=1)
        atcc = DynamicATCC(policy=policy)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="payment",
            level="low",
            read_count=2,
            write_count=3,
            hot_write_count=1,
            retry_count=0,
            hot_targets=("tpcc:warehouse:0:ytd",),
            hot_read_targets=(),
            write_targets=("tpcc:warehouse:0:ytd", "tpcc:district:0:0:orders"),
            reasoning_delay_ms=10,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="lock-before-commit"))
        row.visits = 2
        row.actions["lock-before-commit"] = ATCCActionStats(visits=2, commits=2, avg_reward=100.0)

        decision = atcc.decide(features)

        self.assertEqual("write-validate", decision.action)
        self.assertEqual("runtime-low-conflict-write-validate", decision.metadata["selected_action_source"])

    def test_sparse_high_conflict_policy_uses_risk_prior(self):
        policy = ATCCPolicyTable(min_visits=5)
        state_key = "workload=ycsb|task=read-update|level=high|contention=hot|agent_cost=very-long|write_set=small|retry=first"

        self.assertEqual("lock-before-commit", policy.action_for(state_key))

    def test_dynamic_atcc_folds_protection_into_single_strategy(self):
        policy = ATCCPolicyTable(min_visits=1)
        atcc = DynamicATCC(policy=policy)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="new_order",
            level="high",
            read_count=2,
            write_count=6,
            hot_write_count=3,
            retry_count=0,
            hot_targets=("tpcc:district:0:0:next_order_id",),
            read_targets=("tpcc:warehouse:0:tax", "tpcc:item:1"),
            hot_read_targets=(),
            write_targets=("tpcc:district:0:0:next_order_id", "tpcc:district:0:0:orders"),
            phase_count=3,
            reasoning_delay_ms=180,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="lock-write-set"))
        row.visits = 2

        decision = atcc.decide(features)

        self.assertEqual("dynamic-atcc", atcc.name)
        self.assertEqual("lock-before-commit", decision.action)
        self.assertEqual("deferred-protect", decision.metadata["execution_path"])
        self.assertGreater(decision.metadata["risk_score"], 0)
        self.assertGreater(decision.priority, 0)
        self.assertEqual("runtime-risk-deferred-protect", decision.metadata["selected_action_source"])

    def test_dynamic_atcc_bounds_tpcc_medium_hot_writes_before_retry(self):
        policy = ATCCPolicyTable(min_visits=1)
        atcc = DynamicATCC(policy=policy)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="new_order",
            level="medium",
            read_count=0,
            write_count=6,
            hot_write_count=3,
            retry_count=0,
            hot_targets=("tpcc:district:0:0:next_order_id",),
            hot_read_targets=(),
            write_targets=("tpcc:district:0:0:next_order_id", "tpcc:district:0:0:orders"),
            phase_count=3,
            reasoning_delay_ms=110,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="reserve-hot"))
        row.visits = 3
        row.actions["reserve-hot"] = ATCCActionStats(visits=3, commits=3, avg_reward=100.0)

        decision = atcc.decide(features)

        self.assertEqual("reserve-hot-rw-k", decision.action)
        self.assertEqual(("tpcc:district:0:0:next_order_id",), decision.targets)
        self.assertEqual("runtime-tpcc-medium-write-set-protect", decision.metadata["selected_action_source"])

        retry_decision = atcc.decide(dataclasses.replace(features, retry_count=1))
        self.assertEqual("lock-before-commit", retry_decision.action)
        self.assertEqual("runtime-tpcc-medium-retry-write-set-protect", retry_decision.metadata["selected_action_source"])

    def test_dynamic_atcc_uses_hot_reservation_for_tpcc_medium_payment(self):
        policy = ATCCPolicyTable(min_visits=1)
        atcc = DynamicATCC(policy=policy)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="payment",
            level="medium",
            read_count=0,
            write_count=3,
            hot_write_count=2,
            retry_count=0,
            hot_targets=("tpcc:warehouse:0:ytd", "tpcc:district:0:0:orders"),
            hot_read_targets=(),
            write_targets=(
                "tpcc:warehouse:0:ytd",
                "tpcc:customer:0:0:1:balance",
                "tpcc:district:0:0:orders",
            ),
            phase_count=2,
            reasoning_delay_ms=25,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="occ"))
        row.visits = 3
        row.actions["occ"] = ATCCActionStats(visits=3, commits=1, aborts=2, avg_reward=-10.0)

        decision = atcc.decide(features)

        self.assertEqual("reserve-hot", decision.action)
        self.assertEqual(("tpcc:warehouse:0:ytd", "tpcc:district:0:0:orders"), decision.targets)
        self.assertEqual("runtime-tpcc-medium-write-set-protect", decision.metadata["selected_action_source"])

    def test_dynamic_atcc_protects_tpcc_medium_mixed_read_write_hot_writes(self):
        policy = ATCCPolicyTable(min_visits=1)
        atcc = DynamicATCC(policy=policy)
        features = ATCCFeatures(
            workload="tpcc",
            task_type="new_order",
            level="medium",
            read_count=6,
            write_count=4,
            hot_write_count=2,
            retry_count=0,
            hot_targets=("tpcc:district:0:0:next_order_id",),
            hot_read_targets=("tpcc:stock:0:1",),
            write_targets=("tpcc:district:0:0:next_order_id", "tpcc:stock:0:1"),
            read_targets=("tpcc:warehouse:0:tax", "tpcc:item:1"),
            phase_count=3,
            reasoning_delay_ms=110,
        )
        row = policy.rows.setdefault(features.state_key, ATCCPolicyRow(action="occ"))
        row.visits = 3
        row.actions["occ"] = ATCCActionStats(visits=3, commits=1, aborts=2, avg_reward=-10.0)

        decision = atcc.decide(features)

        self.assertEqual("reserve-hot-rw", decision.action)
        self.assertEqual(
            "runtime-tpcc-medium-write-set-protect",
            decision.metadata["selected_action_source"],
        )

    def test_dynamic_atcc_retry_upgrades_priority_and_protection(self):
        atcc = DynamicATCC(policy=ATCCPolicyTable(min_visits=5))
        first = ATCCFeatures(
            workload="ycsb",
            task_type="read-update",
            level="medium",
            read_count=3,
            write_count=2,
            hot_write_count=1,
            retry_count=0,
            hot_targets=("ycsb:record:0:field:0",),
            hot_read_targets=("ycsb:record:0:field:1",),
            write_targets=("ycsb:record:0:field:0", "ycsb:record:1:field:0"),
            phase_count=3,
            reasoning_delay_ms=45,
        )
        retry = dataclasses.replace(first, retry_count=1)

        first_decision = atcc.decide(first)
        retry_decision = atcc.decide(retry)

        self.assertIn(first_decision.action, {"occ", "write-validate", "lock-before-commit", "reserve-hot-rw"})
        self.assertEqual("reserve-hot-rw", retry_decision.action)
        self.assertEqual("hot-set-reservation", retry_decision.metadata["execution_path"])
        self.assertGreaterEqual(retry_decision.priority, first_decision.priority)

    def test_mixed_2pl_baseline_exposes_guard_metrics(self):
        stdout = io.StringIO()
        status = mixed.main(
            [
                "--workload",
                "tpcc",
                "--level",
                "high",
                "--cc",
                "2pl-nowait",
                "--duration",
                "0.2",
                "--agents",
                "1",
                "--background",
                "1",
                "--reasoning-profile",
                "light",
                "--reasoning-scale",
                "0.1",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        row = report["cc_results"][0]
        self.assertEqual("2pl-nowait", row["cc"])
        self.assertIn("2pl-nowait", row["action_counts"])
        self.assertIn("agent_guard_wait_ms", row)
        self.assertIn("background_guard_wait_ms", row)

    def test_mixed_runtime_bamboo_and_polaris_are_internal_traditional_ccs(self):
        stdout = io.StringIO()
        status = mixed.main(
            [
                "--workload",
                "tpcc",
                "--level",
                "medium",
                "--cc",
                "bamboo,polaris",
                "--duration",
                "0.2",
                "--agents",
                "1",
                "--background",
                "1",
                "--reasoning-profile",
                "light",
                "--reasoning-scale",
                "0.1",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        rows = {row["cc"]: row for row in report["cc_results"]}
        self.assertEqual({"bamboo", "polaris"}, set(rows))
        self.assertIn("bamboo", rows["bamboo"]["action_counts"])
        self.assertIn("polaris", rows["polaris"]["action_counts"])
        self.assertIn("agent_guard_wait_ms", rows["bamboo"])
        self.assertIn("agent_guard_wait_ms", rows["polaris"])

    def test_matrix_cli_emits_speedup_summary(self):
        stdout = io.StringIO()
        status = matrix.main(
            [
                "--workloads",
                "ycsb",
                "--levels",
                "low",
                "--seeds",
                "920104",
                "--cc",
                "occ,dynamic-atcc",
                "--duration",
                "0.1",
                "--clients",
                "10",
                "--agent-ratio",
                "0.8",
                "--reasoning-profile",
                "none",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        self.assertEqual("mixed-starvation-matrix", report["mode"])
        self.assertEqual(["ycsb"], report["workloads"])
        self.assertEqual(["low"], report["levels"])
        self.assertEqual([920104], report["seeds"])
        self.assertEqual(10, report["clients"])
        self.assertEqual(8, report["agent_workers"])
        self.assertEqual(2, report["background_workers"])
        self.assertEqual(
            [{"clients": 10, "agent_workers": 8, "background_workers": 2}],
            report["client_worker_mix"],
        )
        self.assertGreaterEqual(len(report["summary"]), 2)
        self.assertIn("agent_tps_speedup_vs_occ", report["summary"][0])
        self.assertIn("agent_p9999_latency_ms_mean", report["summary"][0])
        self.assertIn("agent_avg_tokens_mean", report["summary"][0])
        self.assertIn("runs", report)

    def test_paper_style_matrix_sweeps_clients_and_figures(self):
        stdout = io.StringIO()
        status = matrix.main(
            [
                "--paper-style",
                "--workloads",
                "ycsb",
                "--levels",
                "low",
                "--seeds",
                "920104",
                "--client-counts",
                "8,16",
                "--cc",
                "occ",
                "--duration",
                "0.1",
                "--reasoning-profile",
                "none",
                "--max-attempts-per-task",
                "2",
                "--agent-retry-backoff-ms",
                "1,1",
                "--background-retry-backoff-ms",
                "1,1",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        self.assertEqual("paper", report["workload_profile"])
        self.assertEqual("procedure", report["background_mode"])
        self.assertTrue(report["retry_until_commit"])
        self.assertEqual([8, 16], report["client_counts"])
        self.assertEqual(
            [
                {"clients": 8, "agent_workers": 6, "background_workers": 2},
                {"clients": 16, "agent_workers": 13, "background_workers": 3},
            ],
            report["client_worker_mix"],
        )
        self.assertEqual({8, 16}, {row["clients"] for row in report["summary"]})
        self.assertIn("paper_figures", report)
        for name in ("agent_throughput", "total_throughput", "avg_tokens", "p9999_latency_ms"):
            self.assertIn(name, report["paper_figures"])
            self.assertEqual(2, len(report["paper_figures"][name]))

    def test_matrix_clients_supports_all_agent_split_and_zipfian_override(self):
        stdout = io.StringIO()
        status = matrix.main(
            [
                "--workloads",
                "ycsb",
                "--levels",
                "medium",
                "--workload-profile",
                "paper",
                "--zipfian",
                "0.8",
                "--seeds",
                "920104",
                "--client-counts",
                "4,8",
                "--cc",
                "occ",
                "--duration",
                "0.1",
                "--agent-ratio",
                "1.0",
                "--reasoning-profile",
                "none",
                "--atcc-pure-policy",
            ],
            stdout=stdout,
        )

        self.assertEqual(0, status)
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["atcc_pure_policy"])
        self.assertEqual(0.8, report["ycsb_zipf_theta"])
        self.assertEqual(
            [
                {"clients": 4, "agent_workers": 4, "background_workers": 0},
                {"clients": 8, "agent_workers": 8, "background_workers": 0},
            ],
            report["client_worker_mix"],
        )
        self.assertEqual({0}, {row["background_workers"] for row in report["summary"]})
        self.assertEqual({True}, {row["atcc_pure_policy"] for row in report["summary"]})


if __name__ == "__main__":
    unittest.main()

import dataclasses
import io
import json
import threading
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.cc import ConcurrencyControlRegistry, LockConflict, ReservationTable
from agent.benchmarks.mixed import (
    MixedBenchmarkConfig,
    apply_atcc_experiment_overrides,
    can_defer_transaction_begin,
    can_defer_read_heavy_transaction_begin,
    mixed_transaction_metadata,
    run_agent_with_deferred_commit_lock,
    run_agent_with_deferred_read_optimistic,
    run_agent_with_deferred_read_reservation,
    run_agent_attempt,
)
from agent.benchmarks.phases import PlannedPhase, PlannedTask
from agent.cli import compare, matrix, mixed, train_atcc
from agent.cc.atcc.common import ATCCDecision
from agent.cc.atcc.dynamic import DynamicATCC
from agent.cc.atcc.features import ATCCFeatures
from agent.cc.atcc.policy import ATCCActionStats, ATCCPolicyTable, ATCCPolicyRow
from agent.runtime import AgentTransactionManager
from agent.workloads import AgentOperation, AgentTask, build_workload


class CompareCliTests(unittest.TestCase):
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
            "dynamic-atcc",
        ]

        self.assertEqual(expected, registry.expand("all"))
        for name in expected:
            self.assertEqual(name, registry.resolve(name).name)
        with self.assertRaises(ValueError):
            registry.resolve("semantic")

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
                    "0.2",
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
                    "reserve-read-write-set",
                    "lock-before-commit",
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

    def test_dynamic_atcc_keeps_full_set_until_reservation_convoy(self):
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
        self.assertEqual(3, len(decision.targets))
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
        self.assertEqual(3, decision.metadata["post_policy_override_target_limit"])
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
            background_workers=5,
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
            read_count=0,
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
            read_count=0,
            write_count=6,
            hot_write_count=3,
            retry_count=0,
            hot_targets=("tpcc:district:0:0:next_order_id",),
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

    def test_dynamic_atcc_prefers_deferred_protection_for_medium_hot_writes(self):
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

        self.assertEqual("lock-before-commit", decision.action)
        self.assertEqual("runtime-hot-write-deferred-protect", decision.metadata["selected_action_source"])

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

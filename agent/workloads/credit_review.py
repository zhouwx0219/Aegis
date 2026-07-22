"""Streaming enterprise credit-review workload for agentic transactions."""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import time
from typing import Any, Callable, Iterable

from .base import ObjectSpec


@dataclasses.dataclass(frozen=True)
class CreditReviewConfig:
    company_count: int = 256
    sector_count: int = 8
    region_count: int = 4
    zipf_theta: float = 0.99
    reasoning_scale: float = 1.0
    commit_apply_ms: int = 24
    compliance_shards: int = 4

    def normalized(self) -> "CreditReviewConfig":
        if int(self.company_count) < 8:
            raise ValueError("company_count must be at least 8")
        if int(self.sector_count) < 2 or int(self.region_count) < 2:
            raise ValueError("sector_count and region_count must be at least 2")
        if float(self.zipf_theta) < 0.0:
            raise ValueError("zipf_theta must be non-negative")
        if float(self.reasoning_scale) < 0.0:
            raise ValueError("reasoning_scale must be non-negative")
        if int(self.commit_apply_ms) < 0:
            raise ValueError("commit_apply_ms must be non-negative")
        if int(self.compliance_shards) < 2:
            raise ValueError("compliance_shards must be at least 2")
        return dataclasses.replace(
            self,
            company_count=int(self.company_count),
            sector_count=int(self.sector_count),
            region_count=int(self.region_count),
            zipf_theta=float(self.zipf_theta),
            reasoning_scale=float(self.reasoning_scale),
            commit_apply_ms=int(self.commit_apply_ms),
            compliance_shards=int(self.compliance_shards),
        )


@dataclasses.dataclass(frozen=True)
class CreditReviewTaskSpec:
    task_seed: int
    company_id: int


@dataclasses.dataclass(frozen=True)
class CreditReviewExecution:
    reasoning_ms: int
    reasoning_tokens: int
    operation_count: int
    branch: str
    revealed_targets: tuple[str, ...]
    commit_admission_wait_ms: float = 0.0


class CreditReviewWorkload:
    """A data-dependent Explore-Refine-Commit workflow.

    Only the company identifier is known at task admission. The sector, region,
    risk branch, and all branch-specific objects are derived after the profile
    read has executed. The runtime never receives a complete access footprint.
    """

    name = "credit_review"

    def __init__(self, config: CreditReviewConfig | None = None):
        self.config = (config or CreditReviewConfig()).normalized()
        weights = [
            1.0 / ((rank + 1) ** self.config.zipf_theta)
            for rank in range(self.config.company_count)
        ]
        total = sum(weights)
        running = 0.0
        self._company_cdf: list[float] = []
        for weight in weights:
            running += weight / total
            self._company_cdf.append(running)
        self._company_cdf[-1] = 1.0

    def objects(self) -> Iterable[ObjectSpec]:
        for company_id in range(self.config.company_count):
            sector = stable_int(self.config.sector_count, "sector", company_id)
            region = stable_int(self.config.region_count, "region", company_id)
            risk_roll = stable_int(100, "risk", company_id)
            risk = "low" if risk_roll < 45 else "medium" if risk_roll < 80 else "high"
            payment_score = 35 + stable_int(66, "payments", company_id)
            financial_score = 300 + stable_int(701, "financials", company_id)
            current_limit = 20_000 + stable_int(80_001, "limit", company_id)
            prefix = self.company_prefix(company_id)
            yield ObjectSpec(f"{prefix}:profile", f"{sector}|{region}|{risk}", kind="row")
            yield ObjectSpec(f"{prefix}:payments", str(payment_score), kind="row")
            yield ObjectSpec(f"{prefix}:financials", str(financial_score), kind="row")
            yield ObjectSpec(f"{prefix}:limit", str(current_limit), kind="row")
            yield ObjectSpec(f"{prefix}:risk_status", "unreviewed", kind="row")
            yield ObjectSpec(f"{prefix}:audit", "", kind="text")
            yield ObjectSpec(f"{prefix}:outbox", "", kind="text")
            yield ObjectSpec(
                f"credit:covenant:{company_id}",
                str(40 + stable_int(61, "covenant", company_id)),
                kind="row",
            )
            yield ObjectSpec(
                f"credit:collateral:{company_id}",
                str(20 + stable_int(81, "collateral", company_id)),
                kind="row",
            )
        for risk, factor in (("low", 12), ("medium", 0), ("high", -18)):
            yield ObjectSpec(f"credit:policy:{risk}", str(factor), kind="row")
            yield ObjectSpec(f"credit:risk:{risk}:review_queue", "none", kind="row")
        for committee in range(4):
            yield ObjectSpec(
                f"credit:committee:{committee}:last_decision",
                "none",
                kind="row",
            )
        for shard in range(self.config.compliance_shards):
            yield ObjectSpec(
                f"credit:compliance:{shard}:decision_log_head",
                "none",
                kind="row",
            )
            yield ObjectSpec(
                f"credit:compliance:{shard}:decision_sequence",
                "0",
                kind="row",
            )
        for sector in range(self.config.sector_count):
            yield ObjectSpec(
                f"credit:sector:{sector}:outlook",
                str(-10 + stable_int(31, "outlook", sector)),
                kind="row",
            )
            yield ObjectSpec(
                f"credit:growth:{sector}",
                str(30 + stable_int(71, "growth", sector)),
                kind="row",
            )
            yield ObjectSpec(
                f"credit:sector:{sector}:exposure",
                "none",
                kind="row",
            )
        for region in range(self.config.region_count):
            yield ObjectSpec(
                f"credit:region:{region}:regulation",
                str(-8 + stable_int(17, "regulation", region)),
                kind="row",
            )
            yield ObjectSpec(
                f"credit:watchlist:{region}",
                str(stable_int(11, "watchlist", region)),
                kind="row",
            )
            yield ObjectSpec(
                f"credit:region:{region}:exposure",
                "none",
                kind="row",
            )
        for sector in range(self.config.sector_count):
            for region in range(self.config.region_count):
                yield ObjectSpec(
                    f"credit:portfolio:{sector}:{region}:last_decision",
                    "none",
                    kind="row",
                )

    def register(self, manager: Any) -> None:
        for spec in self.objects():
            manager.register_object(spec.object_id, spec.initial_value, kind=spec.kind)

    def task_for(self, *, seed: int, worker_id: int, sequence: int) -> CreditReviewTaskSpec:
        task_seed = stable_u64("task", seed, worker_id, sequence)
        sample = task_seed / float(2**64 - 1)
        company_id = min(
            self.config.company_count - 1,
            bisect.bisect_left(self._company_cdf, sample),
        )
        return CreditReviewTaskSpec(task_seed=task_seed, company_id=company_id)

    def cursor(
        self,
        task: CreditReviewTaskSpec,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> "CreditReviewCursor":
        return CreditReviewCursor(self, task, sleep_fn=sleep_fn)

    @staticmethod
    def company_prefix(company_id: int) -> str:
        return f"credit:company:{int(company_id)}"


class CreditReviewCursor:
    def __init__(
        self,
        workload: CreditReviewWorkload,
        task: CreditReviewTaskSpec,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.workload = workload
        self.task = task
        self.sleep_fn = sleep_fn
        self.reasoning_ms = 0
        self.reasoning_tokens = 0
        self.operation_count = 0
        self.branch = ""
        self.revealed_targets: list[str] = []
        self.commit_admission_wait_ms = 0.0

    def execute(
        self,
        txn: Any,
        *,
        before_access: Callable[[Any, str, str], float | None] | None = None,
        before_commit_batch: Callable[[Any, tuple[str, ...]], float] | None = None,
    ) -> CreditReviewExecution:
        prefix = self.workload.company_prefix(self.task.company_id)

        txn.enter_phase("explore")
        profile = self._read(txn, f"{prefix}:profile", "profile", "explore", before_access)
        sector, region, risk = parse_profile(profile)
        self.branch = risk
        payments = to_int(
            self._read(txn, f"{prefix}:payments", "payments", "explore", before_access)
        )
        financials = to_int(
            self._read(txn, f"{prefix}:financials", "financials", "explore", before_access)
        )

        txn.enter_phase("refine")
        policy = to_int(
            self._read(txn, f"credit:policy:{risk}", "policy", "refine", before_access)
        )
        outlook = to_int(
            self._read(
                txn,
                f"credit:sector:{sector}:outlook",
                "sector-outlook",
                "refine",
                before_access,
            )
        )
        regulation = to_int(
            self._read(
                txn,
                f"credit:region:{region}:regulation",
                "regional-policy",
                "refine",
                before_access,
            )
        )
        if risk == "low":
            branch_evidence = to_int(
                self._read(
                    txn,
                    f"credit:growth:{sector}",
                    "growth-evidence",
                    "refine",
                    before_access,
                )
            )
        elif risk == "medium":
            branch_evidence = to_int(
                self._read(
                    txn,
                    f"credit:covenant:{self.task.company_id}",
                    "covenant-evidence",
                    "refine",
                    before_access,
                )
            )
        else:
            collateral = to_int(
                self._read(
                    txn,
                    f"credit:collateral:{self.task.company_id}",
                    "collateral-evidence",
                    "refine",
                    before_access,
                )
            )
            watchlist = to_int(
                self._read(
                    txn,
                    f"credit:watchlist:{region}",
                    "watchlist-evidence",
                    "refine",
                    before_access,
                )
            )
            branch_evidence = collateral - 5 * watchlist

        score = payments + financials // 20 + outlook + regulation + branch_evidence // 5
        committee = abs(score) % 4
        compliance_shard = committee % self.workload.config.compliance_shards
        txn.enter_phase("commit")
        commit_targets = (
            f"{prefix}:limit",
            f"{prefix}:risk_status",
            f"credit:portfolio:{sector}:{region}:last_decision",
            f"credit:sector:{sector}:exposure",
            f"credit:region:{region}:exposure",
            f"credit:risk:{risk}:review_queue",
            f"credit:committee:{committee}:last_decision",
            f"credit:compliance:{compliance_shard}:decision_log_head",
            f"credit:compliance:{compliance_shard}:decision_sequence",
            f"{prefix}:audit",
            f"{prefix}:outbox",
        )
        for label in (
            "current-limit",
            "credit-limit",
            "risk-status",
            "portfolio-decision",
            "audit",
            "outbox",
        ):
            self._reason(label, "commit")
        # The external service completes before the final state-dependent read
        # for every concurrency-control system. Aegis may then admit the
        # materialized suffix, but it does not receive a different operation
        # order from the baselines.
        self.sleep_fn(self.workload.config.commit_apply_ms / 1000.0)
        if before_commit_batch is not None:
            self.commit_admission_wait_ms += max(
                0.0,
                float(before_commit_batch(txn, commit_targets)),
            )
        current_limit = to_int(
            self._read(
                txn,
                f"{prefix}:limit",
                "current-limit",
                "commit",
                before_access,
                reason=False,
            )
        )
        decision_sequence = to_int(
            self._read(
                txn,
                f"credit:compliance:{compliance_shard}:decision_sequence",
                "compliance-sequence",
                "commit",
                before_access,
                reason=False,
            )
        )
        adjustment = policy * 250 + (score - 100) * 20
        if risk == "high":
            adjustment = min(-1_000, adjustment)
        elif risk == "low":
            adjustment = max(1_000, adjustment)
        new_limit = max(1_000, current_limit + adjustment)
        decision = "increase" if new_limit > current_limit else "decrease" if new_limit < current_limit else "hold"

        self._write(
            txn,
            f"{prefix}:limit",
            str(new_limit),
            "credit-limit",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"{prefix}:risk_status",
            risk,
            "risk-status",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"credit:portfolio:{sector}:{region}:last_decision",
            f"{self.task.company_id}:{decision}:{score}",
            "portfolio-decision",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"credit:sector:{sector}:exposure",
            f"{self.task.company_id}:{decision}:{new_limit}",
            "sector-exposure",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"credit:region:{region}:exposure",
            f"{self.task.company_id}:{decision}:{new_limit}",
            "region-exposure",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"credit:risk:{risk}:review_queue",
            f"{self.task.company_id}:{decision}:{score}",
            "risk-queue",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"credit:committee:{committee}:last_decision",
            f"{self.task.company_id}:{decision}:{score}",
            "committee-decision",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"credit:compliance:{compliance_shard}:decision_log_head",
            f"{decision_sequence + 1}:{self.task.company_id}:{decision}:{score}",
            "compliance-decision",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"credit:compliance:{compliance_shard}:decision_sequence",
            str(decision_sequence + 1),
            "compliance-sequence",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"{prefix}:audit",
            f"risk={risk};decision={decision};score={score}",
            "audit",
            "commit",
            before_access,
            reason=False,
        )
        self._write(
            txn,
            f"{prefix}:outbox",
            f"credit-review:{decision}:{new_limit}",
            "outbox",
            "commit",
            before_access,
            reason=False,
        )
        return self.snapshot()

    def snapshot(self) -> CreditReviewExecution:
        return CreditReviewExecution(
            reasoning_ms=int(self.reasoning_ms),
            reasoning_tokens=int(self.reasoning_tokens),
            operation_count=int(self.operation_count),
            branch=str(self.branch),
            revealed_targets=tuple(self.revealed_targets),
            commit_admission_wait_ms=float(self.commit_admission_wait_ms),
        )

    def _read(
        self,
        txn: Any,
        object_id: str,
        label: str,
        phase: str,
        before_access: Callable[[Any, str, str], float | None] | None,
        *,
        reason: bool = True,
    ) -> str:
        if reason:
            self._reason(label, phase)
        if before_access is not None:
            self.commit_admission_wait_ms += max(
                0.0,
                float(before_access(txn, "read", object_id) or 0.0),
            )
        self.revealed_targets.append(str(object_id))
        self.operation_count += 1
        return str(txn.read(object_id).value)

    def _write(
        self,
        txn: Any,
        object_id: str,
        value: str,
        label: str,
        phase: str,
        before_access: Callable[[Any, str, str], float | None] | None,
        *,
        reason: bool = True,
    ) -> None:
        if reason:
            self._reason(label, phase)
        if before_access is not None:
            self.commit_admission_wait_ms += max(
                0.0,
                float(before_access(txn, "write", object_id) or 0.0),
            )
        self.revealed_targets.append(str(object_id))
        self.operation_count += 1
        txn.write(object_id, value)

    def _reason(self, label: str, phase: str) -> None:
        delay_ranges = {
            "explore": (10, 24),
            "refine": (14, 30),
            "commit": (6, 16),
        }
        token_ranges = {
            "explore": (260, 620),
            "refine": (360, 820),
            "commit": (160, 420),
        }
        low_ms, high_ms = delay_ranges[phase]
        low_tokens, high_tokens = token_ranges[phase]
        delay_ms = stable_range(low_ms, high_ms, self.task.task_seed, label, "delay")
        tokens = stable_range(low_tokens, high_tokens, self.task.task_seed, label, "tokens")
        scaled_ms = int(round(delay_ms * self.workload.config.reasoning_scale))
        self.reasoning_ms += max(0, scaled_ms)
        self.reasoning_tokens += int(tokens)
        if scaled_ms > 0:
            self.sleep_fn(scaled_ms / 1000.0)


def parse_profile(value: str) -> tuple[int, int, str]:
    parts = str(value).split("|")
    if len(parts) != 3 or parts[2] not in {"low", "medium", "high"}:
        raise ValueError(f"invalid credit profile: {value}")
    return int(parts[0]), int(parts[1]), parts[2]


def to_int(value: Any) -> int:
    return int(float(str(value)))


def stable_u64(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_int(modulus: int, *parts: Any) -> int:
    return stable_u64(*parts) % max(1, int(modulus))


def stable_range(low: int, high: int, *parts: Any) -> int:
    if int(high) <= int(low):
        return int(low)
    return int(low) + stable_int(int(high) - int(low) + 1, *parts)

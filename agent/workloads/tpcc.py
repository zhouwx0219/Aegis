"""General agent-style workload derived from DBx1000 TPC-C transactions."""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, Iterable, Sequence, Tuple

from .base import (
    AgentCandidate,
    AgentOperation,
    AgentTask,
    AgentWorkload,
    ObjectSpec,
    WorkloadManifest,
)


@dataclasses.dataclass(frozen=True)
class TPCCConfig:
    warehouses: int = 1
    districts_per_warehouse: int = 10
    customers_per_district: int = 300
    items: int = 1000
    initial_stock: int = 100
    order_lines: int = 5
    candidates_per_task: int = 3
    transaction_mix: Tuple[Tuple[str, float], ...] = (
        ("new_order", 0.45),
        ("payment", 0.43),
        ("order_status", 0.04),
        ("delivery", 0.04),
        ("stock_level", 0.04),
    )

    def __post_init__(self) -> None:
        if min(
            self.warehouses,
            self.districts_per_warehouse,
            self.customers_per_district,
            self.items,
            self.initial_stock,
            self.order_lines,
            self.candidates_per_task,
        ) <= 0:
            raise ValueError("TPC-C dimensions must be positive")
        supported = {
            "new_order",
            "payment",
            "order_status",
            "delivery",
            "stock_level",
        }
        if not self.transaction_mix or any(
            name not in supported or weight < 0
            for name, weight in self.transaction_mix
        ):
            raise ValueError("invalid TPC-C transaction mix")
        if sum(weight for _, weight in self.transaction_mix) <= 0:
            raise ValueError("TPC-C transaction mix must have positive weight")


class TPCCAgentWorkload(AgentWorkload):
    name = "agent-tpcc-semantic"
    workload_layer = "semantic"

    def __init__(self, config: TPCCConfig = TPCCConfig()):
        self.config = config

    @staticmethod
    def _warehouse(warehouse: int, field: str) -> str:
        return f"tpcc:warehouse:{warehouse}:{field}"

    @staticmethod
    def _district(warehouse: int, district: int, field: str) -> str:
        return f"tpcc:district:{warehouse}:{district}:{field}"

    @staticmethod
    def _customer(warehouse: int, district: int, customer: int, field: str) -> str:
        return f"tpcc:customer:{warehouse}:{district}:{customer}:{field}"

    @staticmethod
    def _stock(warehouse: int, item: int, field: str) -> str:
        return f"tpcc:stock:{warehouse}:{item}:{field}"

    def manifest(self) -> WorkloadManifest:
        return WorkloadManifest(
            name=self.name,
            benchmark_family="TPC-C",
            source_system="DBx1000",
            source_files=(
                "third_party/dbx1000/benchmarks/tpcc.h",
                "third_party/dbx1000/benchmarks/tpcc_wl.cpp",
                "third_party/dbx1000/benchmarks/tpcc_txn.cpp",
                "third_party/dbx1000/benchmarks/tpcc_query.cpp",
                "third_party/dbx1000/benchmarks/TPCC_full_schema.txt",
            ),
            preserved_semantics=(
                "warehouse/district/customer/stock object families",
                "new-order, payment, order-status, delivery, and stock-level task mix",
                "stock lower-bound constraint for new-order",
                "order and history append streams",
                "customer delivery status CAS",
            ),
            agent_adaptations=(
                "flattened versioned KV object layout",
                "natural-language task envelope",
                "ranked K candidate plans",
                "typed delta/append/read/CAS operations",
                "deterministic generation without requiring an LLM",
            ),
            workload_layer=self.workload_layer,
            canonical_name=self.name,
            config=dataclasses.asdict(self.config),
        )

    def objects(self) -> Iterable[ObjectSpec]:
        for warehouse in range(self.config.warehouses):
            yield ObjectSpec(self._warehouse(warehouse, "ytd"), "0", "counter")
            for district in range(self.config.districts_per_warehouse):
                yield ObjectSpec(self._district(warehouse, district, "ytd"), "0", "counter")
                yield ObjectSpec(
                    self._district(warehouse, district, "next_order_id"),
                    "1",
                    "counter",
                )
                yield ObjectSpec(self._district(warehouse, district, "orders"), "", "text")
                yield ObjectSpec(self._district(warehouse, district, "history"), "", "text")
                for customer in range(self.config.customers_per_district):
                    yield ObjectSpec(
                        self._customer(warehouse, district, customer, "balance"),
                        "0",
                        "counter",
                    )
                    yield ObjectSpec(
                        self._customer(warehouse, district, customer, "payment_count"),
                        "0",
                        "counter",
                    )
                    yield ObjectSpec(
                        self._customer(warehouse, district, customer, "status"),
                        "active",
                        "row",
                    )
            for item in range(self.config.items):
                yield ObjectSpec(
                    self._stock(warehouse, item, "quantity"),
                    str(self.config.initial_stock),
                    "counter",
                )
                yield ObjectSpec(self._stock(warehouse, item, "ytd"), "0", "counter")

    def _scope(self, rng: random.Random) -> Tuple[int, int, int]:
        warehouse = rng.randrange(self.config.warehouses)
        district = rng.randrange(self.config.districts_per_warehouse)
        customer = rng.randrange(self.config.customers_per_district)
        return warehouse, district, customer

    def _new_order(
        self, task_index: int, rng: random.Random, warehouse: int, district: int
    ) -> Tuple[AgentCandidate, ...]:
        candidates = []
        for candidate_index in range(self.config.candidates_per_task):
            items = rng.sample(
                range(self.config.items),
                min(self.config.order_lines, self.config.items),
            )
            operations = [
                AgentOperation.delta(
                    self._district(warehouse, district, "next_order_id"), 1
                ),
                AgentOperation.append(
                    self._district(warehouse, district, "orders"),
                    f"|order:{task_index}:{candidate_index}",
                    commutative=True,
                ),
            ]
            for item in items:
                quantity = rng.randint(1, 5)
                operations.append(
                    AgentOperation.delta(
                        self._stock(warehouse, item, "quantity"),
                        -quantity,
                        constrained=True,
                        lower_bound=0,
                    )
                )
                operations.append(
                    AgentOperation.delta(
                        self._stock(warehouse, item, "ytd"), quantity
                    )
                )
            candidates.append(
                AgentCandidate(
                    f"tpcc-new-order-{task_index}-{candidate_index}",
                    float(self.config.candidates_per_task - candidate_index),
                    tuple(operations),
                    metadata={"items": items, "source": "DBx1000/TPCC"},
                )
            )
        return tuple(candidates)

    def _payment(
        self,
        task_index: int,
        rng: random.Random,
        warehouse: int,
        district: int,
    ) -> Tuple[AgentCandidate, ...]:
        candidates = []
        amount = rng.randint(100, 5000)
        customer_choices = rng.sample(
            range(self.config.customers_per_district),
            min(self.config.candidates_per_task, self.config.customers_per_district),
        )
        for candidate_index, customer in enumerate(customer_choices):
            operations = (
                AgentOperation.delta(self._warehouse(warehouse, "ytd"), amount),
                AgentOperation.delta(self._district(warehouse, district, "ytd"), amount),
                AgentOperation.delta(
                    self._customer(warehouse, district, customer, "balance"), -amount
                ),
                AgentOperation.delta(
                    self._customer(warehouse, district, customer, "payment_count"), 1
                ),
                AgentOperation.append(
                    self._district(warehouse, district, "history"),
                    f"|payment:{task_index}:{customer}:{amount}",
                    commutative=True,
                ),
            )
            candidates.append(
                AgentCandidate(
                    f"tpcc-payment-{task_index}-{candidate_index}",
                    float(len(customer_choices) - candidate_index),
                    operations,
                    metadata={"customer": customer, "amount": amount},
                )
            )
        return tuple(candidates)

    def _read_task(
        self,
        task_index: int,
        task_type: str,
        rng: random.Random,
        warehouse: int,
        district: int,
        customer: int,
    ) -> Tuple[AgentCandidate, ...]:
        if task_type == "order_status":
            operations = (
                AgentOperation.read(
                    self._customer(warehouse, district, customer, "balance")
                ),
                AgentOperation.read(self._district(warehouse, district, "orders")),
            )
        else:
            items = rng.sample(
                range(self.config.items), min(self.config.order_lines, self.config.items)
            )
            operations = tuple(
                AgentOperation.read(self._stock(warehouse, item, "quantity"))
                for item in items
            )
        return (
            AgentCandidate(
                f"tpcc-{task_type}-{task_index}", 1.0, tuple(operations)
            ),
        )

    def _delivery(
        self,
        task_index: int,
        warehouse: int,
        district: int,
        customer: int,
    ) -> Tuple[AgentCandidate, ...]:
        customers = [
            (customer + offset) % self.config.customers_per_district
            for offset in range(
                min(self.config.candidates_per_task, self.config.customers_per_district)
            )
        ]
        return tuple(
            AgentCandidate(
                f"tpcc-delivery-{task_index}-{index}",
                float(len(customers) - index),
                (
                    AgentOperation.cas(
                        self._customer(warehouse, district, candidate, "status"),
                        "active",
                        "delivered",
                    ),
                ),
                metadata={"customer": candidate},
            )
            for index, candidate in enumerate(customers)
        )

    def generate_tasks(self, count: int, *, seed: int = 0) -> Sequence[AgentTask]:
        if count < 0:
            raise ValueError("task count must be non-negative")
        rng = random.Random(seed)
        names = [name for name, _ in self.config.transaction_mix]
        weights = [weight for _, weight in self.config.transaction_mix]
        tasks = []
        for task_index in range(count):
            task_type = rng.choices(names, weights=weights, k=1)[0]
            warehouse, district, customer = self._scope(rng)
            if task_type == "new_order":
                candidates = self._new_order(task_index, rng, warehouse, district)
            elif task_type == "payment":
                candidates = self._payment(task_index, rng, warehouse, district)
            elif task_type == "delivery":
                candidates = self._delivery(
                    task_index, warehouse, district, customer
                )
            else:
                candidates = self._read_task(
                    task_index,
                    task_type,
                    rng,
                    warehouse,
                    district,
                    customer,
                )
            tasks.append(
                AgentTask(
                    task_id=f"tpcc-{task_index}",
                    workload=self.name,
                    task_type=task_type,
                    request=f"Complete the TPC-C {task_type} business task.",
                    candidates=candidates,
                    context={
                        "warehouse": warehouse,
                        "district": district,
                        "customer": customer,
                        "source": "DBx1000/TPCC",
                        "agent_phase_sequence": _agent_phase_sequence(task_type),
                    },
                )
            )
        return tasks


def _agent_phase_sequence(task_type: str) -> Tuple[str, ...]:
    if task_type == "new_order":
        return ("explore", "refine", "commit")
    if task_type == "payment":
        return ("refine", "commit")
    if task_type in {"order_status", "stock_level"}:
        return ("explore", "refine")
    if task_type == "delivery":
        return ("commit",)
    return ("commit",)


class TPCCFaithfulAgentWorkload(TPCCAgentWorkload):
    """Agent-side TPC-C layer aligned with DBx1000's native TPCC surface.

    DBx1000's vendored benchmark models Payment and NewOrder. This layer keeps
    that transaction family boundary and uses one candidate per task so native
    DBx1000 CC results can be compared without mixing in agent re-planning.
    """

    name = "agent-tpcc-faithful"
    workload_layer = "faithful"

    def __init__(self, config: TPCCConfig = TPCCConfig()):
        native_mix = tuple(
            (name, weight)
            for name, weight in config.transaction_mix
            if name in {"new_order", "payment"} and weight > 0
        )
        if not native_mix:
            raise ValueError(
                "faithful DBx1000 TPC-C layer needs at least one new_order/payment entry"
            )
        super().__init__(
            dataclasses.replace(
                config, candidates_per_task=1, transaction_mix=native_mix
            )
        )

    def manifest(self) -> WorkloadManifest:
        manifest = super().manifest()
        return WorkloadManifest(
            name=self.name,
            benchmark_family=manifest.benchmark_family,
            source_system=manifest.source_system,
            source_files=manifest.source_files,
            preserved_semantics=(
                "DBx1000-modeled new-order and payment transaction families",
                "warehouse/district/customer/stock object families",
                "stock lower-bound constraint for new-order",
                "order and history append streams",
                "single candidate per request for native comparability",
            ),
            agent_adaptations=(
                "flattened versioned KV object layout",
                "agent transaction envelope without K-candidate re-planning",
                "typed delta/append operations over versioned KV objects",
                "deterministic generation without requiring an LLM",
            ),
            workload_layer=self.workload_layer,
            canonical_name=self.name,
            config=dataclasses.asdict(self.config),
        )

"""TPC-C-style single-plan workload."""

from __future__ import annotations

import dataclasses
import random
from typing import Iterable, Sequence, Tuple

from .base import AgentOperation, AgentTask, AgentWorkload, ObjectSpec


@dataclasses.dataclass(frozen=True)
class TPCCConfig:
    level: str = "low"
    profile: str = "small"
    warehouses: int = 4
    districts_per_warehouse: int = 4
    customers_per_district: int = 40
    items: int = 160
    initial_stock: int = 100
    order_lines: int = 4
    transaction_mix: Tuple[Tuple[str, float], ...] = (("new_order", 1.0),)
    trace_mode: bool = False

    def __post_init__(self) -> None:
        if min(
            self.warehouses,
            self.districts_per_warehouse,
            self.customers_per_district,
            self.items,
            self.initial_stock,
            self.order_lines,
        ) <= 0:
            raise ValueError("TPC-C dimensions must be positive")


def tpcc_config(level: str, profile: str = "small") -> TPCCConfig:
    level = str(level).strip().lower()
    profile = str(profile).strip().lower()
    if profile == "small":
        configs = small_tpcc_configs()
    elif profile == "paper":
        configs = paper_tpcc_configs()
    else:
        raise ValueError(f"unsupported TPC-C profile: {profile}")
    if level not in configs:
        raise ValueError(f"unsupported TPC-C level: {level}")
    return configs[level]


def with_warehouses(config: TPCCConfig, warehouses: int | None = None) -> TPCCConfig:
    if warehouses is None:
        return config
    value = int(warehouses)
    if value <= 0:
        raise ValueError("TPC-C warehouse override must be positive")
    return dataclasses.replace(config, warehouses=value)


def small_tpcc_configs() -> dict[str, TPCCConfig]:
    return {
        "low": TPCCConfig(
            level="low",
            warehouses=4,
            districts_per_warehouse=4,
            customers_per_district=40,
            items=160,
            order_lines=4,
        ),
        "medium": TPCCConfig(
            level="medium",
            warehouses=2,
            districts_per_warehouse=3,
            customers_per_district=40,
            items=120,
            order_lines=6,
        ),
        "high": TPCCConfig(
            level="high",
            warehouses=1,
            districts_per_warehouse=2,
            customers_per_district=40,
            items=100,
            order_lines=8,
        ),
    }


def paper_tpcc_configs() -> dict[str, TPCCConfig]:
    # The paper-style profile keeps the high-contention 1 warehouse setting,
    # but makes low/medium less artificially hot and includes Payment.
    mix: Tuple[Tuple[str, float], ...] = (("new_order", 0.55), ("payment", 0.45))
    return {
        "low": TPCCConfig(
            level="low",
            profile="paper",
            warehouses=48,
            districts_per_warehouse=5,
            customers_per_district=100,
            items=500,
            order_lines=5,
            transaction_mix=mix,
        ),
        "medium": TPCCConfig(
            level="medium",
            profile="paper",
            warehouses=2,
            districts_per_warehouse=3,
            customers_per_district=60,
            items=200,
            order_lines=8,
            transaction_mix=mix,
        ),
        "high": TPCCConfig(
            level="high",
            profile="paper",
            warehouses=1,
            districts_per_warehouse=2,
            customers_per_district=40,
            items=100,
            order_lines=10,
            transaction_mix=mix,
        ),
    }


class TPCCWorkload(AgentWorkload):
    name = "tpcc"
    family = "tpcc"

    def __init__(self, config: TPCCConfig):
        self.config = config
        self.level = config.level

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

    @staticmethod
    def _item(item: int, field: str) -> str:
        return f"tpcc:item:{item}:{field}"

    @staticmethod
    def _order(warehouse: int, district: int, order: int) -> str:
        return f"tpcc:order:{warehouse}:{district}:{order}"

    @staticmethod
    def _new_order_row(warehouse: int, district: int, order: int) -> str:
        return f"tpcc:new-order:{warehouse}:{district}:{order}"

    @staticmethod
    def _order_line(warehouse: int, district: int, order: int, line: int) -> str:
        return f"tpcc:order-line:{warehouse}:{district}:{order}:{line}"

    @staticmethod
    def _history(warehouse: int, district: int, customer: int, sequence: int) -> str:
        return f"tpcc:history:{warehouse}:{district}:{customer}:{sequence}"

    def objects(self) -> Iterable[ObjectSpec]:
        for item in range(self.config.items):
            yield ObjectSpec(self._item(item, "price"), "1", "row")
        for warehouse in range(self.config.warehouses):
            yield ObjectSpec(self._warehouse(warehouse, "ytd"), "0", "counter")
            yield ObjectSpec(self._warehouse(warehouse, "tax"), "0", "row")
            for district in range(self.config.districts_per_warehouse):
                yield ObjectSpec(self._district(warehouse, district, "next_order_id"), "1", "counter")
                yield ObjectSpec(self._district(warehouse, district, "orders"), "", "text")
                yield ObjectSpec(self._district(warehouse, district, "tax"), "0", "row")
                yield ObjectSpec(self._district(warehouse, district, "ytd"), "0", "counter")
                for customer in range(self.config.customers_per_district):
                    yield ObjectSpec(self._customer(warehouse, district, customer, "balance"), "0", "counter")
                    yield ObjectSpec(self._customer(warehouse, district, customer, "status"), "active", "row")
                    yield ObjectSpec(self._customer(warehouse, district, customer, "discount"), "0", "row")
                    yield ObjectSpec(self._customer(warehouse, district, customer, "ytd_payment"), "0", "counter")
                    yield ObjectSpec(self._customer(warehouse, district, customer, "payment_count"), "0", "counter")
            for item in range(self.config.items):
                yield ObjectSpec(self._stock(warehouse, item, "quantity"), str(self.config.initial_stock), "counter")
                yield ObjectSpec(self._stock(warehouse, item, "ytd"), "0", "counter")
                yield ObjectSpec(self._stock(warehouse, item, "order_count"), "0", "counter")

    def generate_tasks(self, count: int, *, seed: int = 0) -> Sequence[AgentTask]:
        if count < 0:
            raise ValueError("task count must be non-negative")
        rng = random.Random(seed)
        names = [name for name, _weight in self.config.transaction_mix]
        weights = [weight for _name, weight in self.config.transaction_mix]
        tasks = []
        for task_index in range(count):
            task_type = rng.choices(names, weights=weights, k=1)[0]
            warehouse = rng.randrange(self.config.warehouses)
            district = rng.randrange(self.config.districts_per_warehouse)
            customer = rng.randrange(self.config.customers_per_district)
            if task_type == "payment":
                operations = self._payment(task_index, rng, warehouse, district, customer)
            else:
                task_type = "new_order"
                operations = self._new_order(task_index, rng, warehouse, district, customer)
            tasks.append(
                AgentTask(
                    task_id=f"tpcc-{task_index}",
                    workload=self.name,
                    task_type=task_type,
                    operations=operations,
                    context={
                        "level": self.level,
                        "profile": self.config.profile,
                        "transaction_mix": tuple(name for name, _weight in self.config.transaction_mix),
                        "warehouse": warehouse,
                        "district": district,
                        "customer": customer,
                        "agent_cost_class": agent_cost_class(self.level, task_index, task_type),
                        "phase_shape": phase_shape(self.level, task_index, task_type),
                        "side_effect_cost_ms": side_effect_cost_ms(self.level, task_index, task_type),
                    },
                )
            )
        return tasks

    def _new_order(
        self,
        task_index: int,
        rng: random.Random,
        warehouse: int,
        district: int,
        customer: int,
    ) -> Tuple[AgentOperation, ...]:
        if self.config.profile != "paper":
            return self._legacy_new_order(task_index, rng, warehouse, district)
        line_count = (
            rng.randint(5, 15)
            if self.config.profile == "paper" and self.config.trace_mode
            else self.config.order_lines
        )
        items = rng.sample(range(self.config.items), min(line_count, self.config.items))
        operations = [
            AgentOperation.read(self._warehouse(warehouse, "tax"), phase="explore"),
            AgentOperation.read(self._district(warehouse, district, "tax"), phase="explore"),
            AgentOperation.read(
                self._district(warehouse, district, "next_order_id"),
                phase="explore",
            ),
            AgentOperation.read(
                self._customer(warehouse, district, customer, "discount"),
                phase="explore",
            ),
        ]
        for item in items:
            operations.append(AgentOperation.read(self._item(item, "price"), phase="refine"))
            operations.append(
                AgentOperation.read(
                    self._stock(warehouse, item, "quantity"),
                    phase="refine",
                )
            )
        operations.extend(
            [
            AgentOperation.write(
                self._district(warehouse, district, "next_order_id"),
                f"order-next:{task_index}",
                phase="commit",
            ),
            AgentOperation.write(
                self._district(warehouse, district, "orders"),
                f"order-log:{task_index}",
                phase="commit",
            ),
            ]
        )
        for line, item in enumerate(items):
            quantity = rng.randint(1, 5)
            operations.append(
                AgentOperation.write(
                    self._stock(warehouse, item, "quantity"),
                    f"stock-q:{task_index}:{item}:{quantity}",
                    phase="commit",
                )
            )
            operations.append(
                AgentOperation.write(
                    self._stock(warehouse, item, "ytd"),
                    f"stock-ytd:{task_index}:{item}:{quantity}",
                    phase="commit",
                )
            )
            operations.append(
                AgentOperation.write(
                    self._stock(warehouse, item, "order_count"),
                    f"stock-orders:{task_index}:{item}",
                    phase="commit",
                )
            )
            if self.config.trace_mode:
                operations.append(
                    AgentOperation.write(
                        self._order_line(warehouse, district, task_index, line),
                        f"line:{customer}:{item}:{quantity}",
                        phase="commit",
                    )
                )
        if self.config.trace_mode:
            operations.extend(
                (
                    AgentOperation.write(
                        self._order(warehouse, district, task_index),
                        f"order:{customer}:{len(items)}",
                        phase="commit",
                    ),
                    AgentOperation.write(
                        self._new_order_row(warehouse, district, task_index),
                        "pending",
                        phase="commit",
                    ),
                )
            )
        return tuple(operations)

    def _legacy_new_order(
        self,
        task_index: int,
        rng: random.Random,
        warehouse: int,
        district: int,
    ) -> Tuple[AgentOperation, ...]:
        items = rng.sample(range(self.config.items), min(self.config.order_lines, self.config.items))
        operations = [
            AgentOperation.write(
                self._district(warehouse, district, "next_order_id"),
                f"order-next:{task_index}",
            ),
            AgentOperation.write(
                self._district(warehouse, district, "orders"),
                f"order-log:{task_index}",
            ),
        ]
        for item in items:
            quantity = rng.randint(1, 5)
            operations.append(
                AgentOperation.write(
                    self._stock(warehouse, item, "quantity"),
                    f"stock-q:{task_index}:{item}:{quantity}",
                )
            )
            operations.append(
                AgentOperation.write(
                    self._stock(warehouse, item, "ytd"),
                    f"stock-ytd:{task_index}:{item}:{quantity}",
                )
            )
        return tuple(operations)

    def _payment(
        self,
        task_index: int,
        rng: random.Random,
        warehouse: int,
        district: int,
        customer: int,
    ) -> Tuple[AgentOperation, ...]:
        if self.config.profile != "paper":
            amount = rng.randint(100, 5000)
            return (
                AgentOperation.write(
                    self._warehouse(warehouse, "ytd"),
                    f"payment-ytd:{task_index}:{amount}",
                ),
                AgentOperation.write(
                    self._customer(warehouse, district, customer, "balance"),
                    f"payment-balance:{task_index}:{amount}",
                ),
                AgentOperation.write(
                    self._district(warehouse, district, "orders"),
                    f"payment-log:{task_index}:{customer}:{amount}",
                ),
            )
        amount = rng.randint(100, 5000)
        return (
            AgentOperation.read(self._warehouse(warehouse, "ytd"), phase="explore"),
            AgentOperation.read(self._district(warehouse, district, "ytd"), phase="explore"),
            AgentOperation.read(
                self._customer(warehouse, district, customer, "balance"),
                phase="explore",
            ),
            AgentOperation.read(
                self._customer(warehouse, district, customer, "status"),
                phase="refine",
            ),
            AgentOperation.write(
                self._warehouse(warehouse, "ytd"),
                f"payment-ytd:{task_index}:{amount}",
                phase="commit",
            ),
            AgentOperation.write(
                self._district(warehouse, district, "ytd"),
                f"payment-district-ytd:{task_index}:{amount}",
                phase="commit",
            ),
            AgentOperation.write(
                self._customer(warehouse, district, customer, "balance"),
                f"payment-balance:{task_index}:{amount}",
                phase="commit",
            ),
            AgentOperation.write(
                self._customer(warehouse, district, customer, "ytd_payment"),
                f"payment-customer-ytd:{task_index}:{amount}",
                phase="commit",
            ),
            AgentOperation.write(
                self._customer(warehouse, district, customer, "payment_count"),
                f"payment-count:{task_index}",
                phase="commit",
            ),
            AgentOperation.write(
                (
                    self._history(warehouse, district, customer, task_index)
                    if self.config.trace_mode
                    else self._district(warehouse, district, "orders")
                ),
                f"payment-history:{amount}",
                phase="commit",
            ),
        )


def agent_cost_class(level: str, task_index: int, task_type: str) -> str:
    level = str(level).strip().lower()
    if task_type == "new_order" and level == "high":
        return "expensive"
    if task_type == "new_order" and level == "medium":
        return "normal"
    if task_index % 4 == 0:
        return "normal"
    return "cheap"


def phase_shape(level: str, task_index: int, task_type: str) -> str:
    level = str(level).strip().lower()
    if task_type == "new_order" and level in {"medium", "high"}:
        return "tool_heavy"
    if task_index % 2 == 0:
        return "multi_stage"
    return "short"


def side_effect_cost_ms(level: str, task_index: int, task_type: str) -> int:
    level = str(level).strip().lower()
    if task_type == "new_order" and level == "high":
        return 60
    if task_type == "new_order" and level == "medium":
        return 25
    return 0

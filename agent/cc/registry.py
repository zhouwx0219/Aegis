"""Registry for traditional CC and ATCC strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from agent.cc.atcc import (
    ATCCPolicyTable,
    DynamicATCC,
)
from agent.cc.base import ConcurrencyControl
from agent.cc.traditional import (
    BambooConcurrencyControl,
    MvccConcurrencyControl,
    OccConcurrencyControl,
    PolarisConcurrencyControl,
    SiloConcurrencyControl,
    TicTocConcurrencyControl,
    TwoPhaseLockingConcurrencyControl,
)


class ConcurrencyControlRegistry:
    """Named CC registry with explicit all-strategy expansion."""

    def __init__(
        self,
        *,
        atcc_policy: ATCCPolicyTable | None = None,
        atcc_options: Mapping[str, Any] | None = None,
    ):
        self._strategies: Dict[str, ConcurrencyControl] = {}
        policy = atcc_policy or ATCCPolicyTable()
        atcc_kwargs = dict(atcc_options or {})
        for strategy in (
            OccConcurrencyControl(),
            TwoPhaseLockingConcurrencyControl("2pl-nowait", "nowait"),
            TwoPhaseLockingConcurrencyControl("2pl-wait-die", "wait-die"),
            MvccConcurrencyControl(),
            SiloConcurrencyControl(),
            TicTocConcurrencyControl(),
            BambooConcurrencyControl(),
            PolarisConcurrencyControl(),
            DynamicATCC(policy=policy, **atcc_kwargs),
            DynamicATCC(
                name="static-atcc",
                policy=ATCCPolicyTable(),
                decision_mode="static",
                priority_enabled=False,
                **atcc_kwargs,
            ),
            DynamicATCC(
                name="static-atcc-priority",
                policy=ATCCPolicyTable(),
                decision_mode="static",
                priority_enabled=True,
                **atcc_kwargs,
            ),
            DynamicATCC(
                name="trained-atcc",
                policy=policy,
                priority_enabled=False,
                **atcc_kwargs,
            ),
            DynamicATCC(
                name="trained-atcc-priority",
                policy=policy,
                priority_enabled=True,
                **atcc_kwargs,
            ),
        ):
            self.register(strategy)

    @classmethod
    def from_policy_file(
        cls,
        path: str | Path | None,
        *,
        atcc_options: Mapping[str, Any] | None = None,
    ) -> "ConcurrencyControlRegistry":
        if path is None or str(path).strip() == "":
            return cls(atcc_options=atcc_options)
        return cls(atcc_policy=ATCCPolicyTable.load_json(Path(path)), atcc_options=atcc_options)

    def register(self, strategy: ConcurrencyControl) -> None:
        name = normalize_name(strategy.name)
        if not name:
            raise ValueError("CC strategy name must not be empty")
        self._strategies[name] = strategy

    def resolve(self, name: str) -> ConcurrencyControl:
        normalized = normalize_name(name)
        if normalized not in self._strategies:
            raise ValueError(f"unknown CC strategy: {name}")
        return self._strategies[normalized]

    def names(self) -> List[str]:
        return list(self.expand("all"))

    def expand(self, value: str | Iterable[str]) -> List[str]:
        if isinstance(value, str):
            raw = [item.strip() for item in value.split(",") if item.strip()]
        else:
            raw = [str(item).strip() for item in value if str(item).strip()]
        if not raw or raw == ["all"]:
            return [
                "occ",
                "2pl-nowait",
                "2pl-wait-die",
                "mvcc",
                "silo",
                "tictoc",
                "bamboo",
                "polaris",
                "dynamic-atcc",
            ]
        names = []
        for item in raw:
            normalized = normalize_name(item)
            self.resolve(normalized)
            names.append(normalized)
        return names

    def strategies(self) -> Dict[str, dict]:
        return {name: strategy.to_dict() for name, strategy in self._strategies.items()}


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")

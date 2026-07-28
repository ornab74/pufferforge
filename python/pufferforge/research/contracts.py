from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    """Raised when an experiment contract is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    max_trials: int = 20
    max_steps_per_trial: int = 1_000_000
    max_wall_seconds: float | None = None
    max_gpu_hours: float | None = None
    max_memory_gb: float | None = None

    def validate(self) -> None:
        if self.max_trials < 2:
            raise ContractError("max_trials must be at least 2")
        if self.max_steps_per_trial <= 0:
            raise ContractError("max_steps_per_trial must be positive")
        for name in ("max_wall_seconds", "max_gpu_hours", "max_memory_gb"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ContractError(f"{name} must be positive when set")


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    direction: str = "maximize"
    aggregation: str = "mean"
    minimum: float | None = None
    maximum: float | None = None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.name.strip():
            raise ContractError("metric name cannot be empty")
        if self.direction not in {"maximize", "minimize"}:
            raise ContractError("metric direction must be maximize or minimize")
        if self.aggregation not in {"mean", "median", "iqm", "sum", "last", "minimum", "maximum"}:
            raise ContractError(f"unsupported aggregation: {self.aggregation}")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ContractError(f"metric {self.name}: minimum exceeds maximum")
        if self.weight <= 0:
            raise ContractError("metric weight must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentArm:
    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    parent: str | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ContractError("arm name cannot be empty")
        if not self.description.strip():
            raise ContractError(f"arm {self.name} needs a description")
        _ensure_jsonable(dict(self.parameters), f"arm {self.name} parameters")


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    probability_superior: float = 0.95
    minimum_effect_size: float = 0.02
    probability_futile: float = 0.95
    maximum_regression: float = 0.02
    minimum_trials_per_arm: int = 3
    maximum_trials_per_arm: int = 20
    safety_metrics: tuple[str, ...] = ()

    def validate(self) -> None:
        for name in ("probability_superior", "probability_futile"):
            value = getattr(self, name)
            if not 0.5 < value < 1.0:
                raise ContractError(f"{name} must be between 0.5 and 1")
        if self.minimum_effect_size < 0:
            raise ContractError("minimum_effect_size must be non-negative")
        if self.maximum_regression < 0:
            raise ContractError("maximum_regression must be non-negative")
        if self.minimum_trials_per_arm < 2:
            raise ContractError("minimum_trials_per_arm must be at least 2")
        if self.maximum_trials_per_arm < self.minimum_trials_per_arm:
            raise ContractError("maximum_trials_per_arm must be >= minimum_trials_per_arm")


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    name: str
    hypothesis: str
    control: str
    arms: tuple[ExperimentArm, ...]
    primary_metric: MetricSpec
    secondary_metrics: tuple[MetricSpec, ...] = ()
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    promotion: PromotionPolicy = field(default_factory=PromotionPolicy)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    invariants: tuple[str, ...] = ()
    confounders: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self) -> None:
        if not self.name.strip():
            raise ContractError("experiment name cannot be empty")
        if not self.hypothesis.strip():
            raise ContractError("hypothesis cannot be empty")
        if len(self.arms) < 2:
            raise ContractError("an A/B experiment requires at least two arms")
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ContractError("arm names must be unique")
        if self.control not in set(names):
            raise ContractError("control must match an arm name")
        if len(self.seeds) < 2:
            raise ContractError("at least two seeds are required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ContractError("seeds must be unique")
        if self.schema_version != 1:
            raise ContractError(f"unsupported schema_version: {self.schema_version}")
        for arm in self.arms:
            arm.validate()
        self.primary_metric.validate()
        for metric in self.secondary_metrics:
            metric.validate()
        metric_names = [self.primary_metric.name, *(m.name for m in self.secondary_metrics)]
        if len(metric_names) != len(set(metric_names)):
            raise ContractError("metric names must be unique")
        unknown_safety = set(self.promotion.safety_metrics) - set(metric_names)
        if unknown_safety:
            raise ContractError(f"unknown safety metrics: {sorted(unknown_safety)}")
        self.budget.validate()
        self.promotion.validate()
        _ensure_jsonable(dict(self.metadata), "metadata")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentContract":
        arms = tuple(ExperimentArm(**{**arm, "tags": tuple(arm.get("tags", ()))}) for arm in data["arms"])
        primary = MetricSpec(**data["primary_metric"])
        secondary = tuple(MetricSpec(**metric) for metric in data.get("secondary_metrics", ()))
        budget = ResourceBudget(**data.get("budget", {}))
        promotion_data = dict(data.get("promotion", {}))
        if "safety_metrics" in promotion_data:
            promotion_data["safety_metrics"] = tuple(promotion_data["safety_metrics"])
        promotion = PromotionPolicy(**promotion_data)
        contract = cls(
            name=str(data["name"]),
            hypothesis=str(data["hypothesis"]),
            control=str(data["control"]),
            arms=arms,
            primary_metric=primary,
            secondary_metrics=secondary,
            budget=budget,
            promotion=promotion,
            seeds=tuple(int(seed) for seed in data.get("seeds", (1, 2, 3, 4, 5))),
            invariants=tuple(str(v) for v in data.get("invariants", ())),
            confounders=tuple(str(v) for v in data.get("confounders", ())),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", 1)),
        )
        contract.validate()
        return contract

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentContract":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @property
    def stable_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(payload.encode("utf-8")).hexdigest()[:20]


def _ensure_jsonable(value: Any, label: str) -> None:
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be JSON serializable") from exc


def merge_parameters(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_parameters(result[key], value)
        else:
            result[key] = value
    return result


def validate_contracts(contracts: Sequence[ExperimentContract]) -> None:
    names: set[str] = set()
    identifiers: set[str] = set()
    for contract in contracts:
        contract.validate()
        if contract.name in names:
            raise ContractError(f"duplicate contract name: {contract.name}")
        if contract.stable_id in identifiers:
            raise ContractError(f"duplicate contract contents: {contract.name}")
        names.add(contract.name)
        identifiers.add(contract.stable_id)

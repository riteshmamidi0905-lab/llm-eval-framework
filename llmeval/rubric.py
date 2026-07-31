"""Rubric and Criterion definitions.

A Rubric is an ordered collection of weighted Criteria. Each Criterion names a
metric (a registered scoring function) and the weight it contributes to the
final score. Rubrics can be declared in Python or loaded from YAML/JSON so that
non-engineers (policy, quality) can edit evaluation guidelines without touching
code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass(frozen=True)
class Criterion:
    """A single scored dimension in a rubric.

    Attributes:
        name: Human-readable label, e.g. "faithfulness".
        metric: Name of a registered metric function (see llmeval.metrics).
        weight: Relative weight in the aggregate score. Weights are normalized
            across the rubric, so they need not sum to 1.
        threshold: Optional pass/fail cut-off in [0, 1]. If set, the criterion
            is marked as failing when its score is below the threshold.
        params: Extra keyword arguments forwarded to the metric function.
    """

    name: str
    metric: str
    weight: float = 1.0
    threshold: float | None = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Criterion '{self.name}' has negative weight")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Criterion '{self.name}' threshold must be in [0, 1]")


@dataclass
class Rubric:
    """An ordered set of weighted criteria."""

    name: str
    criteria: List[Criterion]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("A rubric must define at least one criterion")
        total = sum(c.weight for c in self.criteria)
        if total <= 0:
            raise ValueError("Rubric weights must sum to a positive number")
        self._total_weight = total

    def normalized_weights(self) -> Dict[str, float]:
        """Return each criterion's weight normalized to sum to 1.0."""
        return {c.name: c.weight / self._total_weight for c in self.criteria}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rubric":
        criteria = [
            Criterion(
                name=c["name"],
                metric=c["metric"],
                weight=float(c.get("weight", 1.0)),
                threshold=c.get("threshold"),
                params=c.get("params", {}),
            )
            for c in data["criteria"]
        ]
        return cls(name=data.get("name", "unnamed"), criteria=criteria)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Rubric":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Install pyyaml to load YAML rubrics") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

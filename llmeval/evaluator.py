"""The Evaluator ties rubrics, cases, and metrics together."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .cases import EvalCase
from .metrics import get_metric
from .rubric import Rubric

# A scorer takes a case and returns a float in [0, 1].
Scorer = Callable[..., float]


@dataclass
class CriterionScore:
    name: str
    score: float
    weight: float
    passed: Optional[bool]


@dataclass
class CaseResult:
    case_id: str
    aggregate: float
    passed: bool
    criteria: List[CriterionScore]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Aggregate results across an evaluation run."""

    rubric_name: str
    case_results: List[CaseResult]

    @property
    def mean_score(self) -> float:
        if not self.case_results:
            return 0.0
        return statistics.fmean(r.aggregate for r in self.case_results)

    @property
    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for r in self.case_results if r.passed) / len(self.case_results)

    def criterion_means(self) -> Dict[str, float]:
        """Mean score per criterion across all cases — useful for spotting the
        weakest evaluation dimension (e.g. faithfulness) at a glance."""
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for res in self.case_results:
            for c in res.criteria:
                sums[c.name] = sums.get(c.name, 0.0) + c.score
                counts[c.name] = counts.get(c.name, 0) + 1
        return {k: sums[k] / counts[k] for k in sums}

    def summary(self) -> Dict[str, object]:
        return {
            "rubric": self.rubric_name,
            "n_cases": len(self.case_results),
            "mean_score": round(self.mean_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "criterion_means": {k: round(v, 4) for k, v in self.criterion_means().items()},
        }


class Evaluator:
    """Runs a rubric over a list of cases.

    Custom scorers (e.g. an ``LLMJudge`` instance) can be supplied per criterion
    name via ``custom_scorers``, overriding the built-in metric lookup. A case
    passes overall when its aggregate meets ``pass_threshold`` AND every
    criterion with its own threshold passes.
    """

    def __init__(
        self,
        rubric: Rubric,
        custom_scorers: Optional[Dict[str, Scorer]] = None,
        pass_threshold: float = 0.7,
    ) -> None:
        self.rubric = rubric
        self.custom_scorers = custom_scorers or {}
        self.pass_threshold = pass_threshold
        self._weights = rubric.normalized_weights()

    def _score_case(self, case: EvalCase) -> CaseResult:
        criteria_scores: List[CriterionScore] = []
        aggregate = 0.0
        hard_fail = False
        for crit in self.rubric.criteria:
            scorer = self.custom_scorers.get(crit.name) or get_metric(crit.metric)
            score = float(scorer(case, **crit.params))
            score = max(0.0, min(1.0, score))
            weight = self._weights[crit.name]
            aggregate += weight * score
            passed = None
            if crit.threshold is not None:
                passed = score >= crit.threshold
                if not passed:
                    hard_fail = True
            criteria_scores.append(CriterionScore(crit.name, score, weight, passed))
        overall_pass = (aggregate >= self.pass_threshold) and not hard_fail
        return CaseResult(
            case_id=case.case_id or "",
            aggregate=aggregate,
            passed=overall_pass,
            criteria=criteria_scores,
            metadata=dict(case.metadata),
        )

    def evaluate(self, cases: List[EvalCase]) -> EvalResult:
        results = [self._score_case(c) for c in cases]
        return EvalResult(rubric_name=self.rubric.name, case_results=results)

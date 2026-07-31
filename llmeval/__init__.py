"""llmeval — a lightweight, rubric-driven framework for evaluating LLM outputs.

Public API:
    from llmeval import Evaluator, Rubric, Criterion, EvalCase

The framework is designed to run fully offline using deterministic heuristic
metrics, with an optional "LLM-as-judge" backend for graded scoring when an
API key is available.
"""

from .rubric import Criterion, Rubric
from .cases import EvalCase, load_cases
from .evaluator import Evaluator, EvalResult
from .report import to_json, write_json, render_html, write_html_report
from .metrics import available_metrics, register, get_metric

__all__ = [
    "Criterion",
    "Rubric",
    "EvalCase",
    "load_cases",
    "Evaluator",
    "EvalResult",
    "to_json",
    "write_json",
    "render_html",
    "write_html_report",
    "available_metrics",
    "register",
    "get_metric",
]

__version__ = "0.1.0"

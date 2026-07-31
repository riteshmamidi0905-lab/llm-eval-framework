import json
from pathlib import Path

import pytest

from llmeval import Criterion, EvalCase, Evaluator, Rubric
from llmeval.cases import load_cases
from llmeval.cli import run

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def simple_rubric():
    return Rubric(
        name="test",
        criteria=[
            Criterion("faith", "faithfulness", weight=0.5, threshold=0.5),
            Criterion("overlap", "token_f1", weight=0.5),
        ],
    )


def test_weights_are_normalized():
    r = Rubric("r", [Criterion("a", "token_f1", weight=3), Criterion("b", "rouge_l", weight=1)])
    w = r.normalized_weights()
    assert w["a"] == pytest.approx(0.75)
    assert w["b"] == pytest.approx(0.25)


def test_rubric_requires_criteria():
    with pytest.raises(ValueError):
        Rubric("empty", [])


def test_grounded_case_passes_and_hallucination_fails():
    grounded = EvalCase(prompt="q", output="open daily seven to eight",
                        reference="open daily seven to eight",
                        context="the shop is open daily seven to eight")
    hallucinated = EvalCase(prompt="q", output="free parking lot open all night",
                            reference="metered street parking only",
                            context="metered street parking only")
    ev = Evaluator(simple_rubric(), pass_threshold=0.6)
    res = ev.evaluate([grounded, hallucinated])
    by_id = {r.case_id: r for r in res.case_results}
    # case_ids default to positional indexes here
    assert res.case_results[0].passed is True
    assert res.case_results[1].passed is False


def test_hard_threshold_forces_fail_even_with_high_aggregate():
    # Output perfectly matches reference (high token_f1) but is ungrounded.
    case = EvalCase(prompt="q", output="totally invented answer",
                    reference="totally invented answer",
                    context="unrelated grounding context about weather")
    ev = Evaluator(simple_rubric(), pass_threshold=0.1)
    res = ev.evaluate([case])
    r = res.case_results[0]
    assert r.aggregate > 0.1          # aggregate would pass on its own
    assert r.passed is False          # but faithfulness threshold hard-fails it


def test_summary_and_criterion_means():
    cases = load_cases(EXAMPLES / "sample_dataset.jsonl")
    rubric = Rubric.from_file(EXAMPLES / "rubric.json")
    res = Evaluator(rubric).evaluate(cases)
    summary = res.summary()
    assert summary["n_cases"] == len(cases)
    assert 0.0 <= summary["mean_score"] <= 1.0
    assert "faithfulness" in summary["criterion_means"]


def test_cli_runs_and_writes_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    code = run([
        "--rubric", str(EXAMPLES / "rubric.json"),
        "--cases", str(EXAMPLES / "sample_dataset.jsonl"),
        "--json", str(out),
    ])
    assert code == 0
    report = json.loads(out.read_text())
    assert "summary" in report and "cases" in report
    assert len(report["cases"]) == 6


def test_cli_fail_under_gate(tmp_path):
    # An impossibly high bar should trip the CI gate and return non-zero.
    code = run([
        "--rubric", str(EXAMPLES / "rubric.json"),
        "--cases", str(EXAMPLES / "sample_dataset.jsonl"),
        "--fail-under", "0.999",
    ])
    assert code == 1

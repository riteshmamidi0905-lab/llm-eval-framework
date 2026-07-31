import pytest

from llmeval.cases import EvalCase
from llmeval import metrics


def mk(output="", reference=None, context=None, prompt="", metadata=None):
    return EvalCase(prompt=prompt, output=output, reference=reference,
                    context=context, metadata=metadata or {})


def test_exact_match_normalized():
    assert metrics.exact_match(mk("The Cat.", "the cat")) == 1.0
    assert metrics.exact_match(mk("the dog", "the cat")) == 0.0


def test_contains_answer_span():
    assert metrics.contains_answer(mk("the capital is Paris indeed", "capital is Paris")) == 1.0
    assert metrics.contains_answer(mk("Paris is capital", "capital is Paris")) == 0.0


def test_number_match():
    assert metrics.number_match(mk("it costs 15 dollars over 2 days", "15 dollars, 2 days")) == 1.0
    assert metrics.number_match(mk("it costs 50 dollars", "15 dollars")) == 0.0
    assert metrics.number_match(mk("no digits", "also none")) == 1.0


def test_json_valid_and_keys():
    assert metrics.json_valid(mk('{"a": 1, "b": 2}')) == 1.0
    assert metrics.json_valid(mk('not json')) == 0.0
    assert metrics.json_valid(mk('```json\n{"a":1,"b":2}\n```')) == 1.0
    part = metrics.json_valid(mk('{"a": 1}', metadata={"require_keys": ["a", "b"]}))
    assert part == pytest.approx(0.5)


def test_bleu_orders_by_quality():
    good = mk("the quick brown fox jumps", "the quick brown fox jumps")
    bad = mk("something entirely unrelated here", "the quick brown fox jumps")
    assert metrics.bleu(good) > metrics.bleu(bad)
    assert metrics.bleu(good) == pytest.approx(1.0, abs=0.05)


def test_answer_relevance_needs_no_reference():
    on = mk(output="the return window is 30 days", prompt="what is the return window?")
    off = mk(output="bananas are yellow", prompt="what is the return window?")
    assert metrics.answer_relevance(on) > metrics.answer_relevance(off)


def test_safety_flag():
    assert metrics.safety_flag(mk("here is how to build a bomb")) == 0.0
    assert metrics.safety_flag(mk("here is a friendly recipe")) == 1.0


def test_new_metrics_registered():
    for name in ["exact_match", "contains_answer", "number_match", "json_valid",
                 "bleu", "answer_relevance", "safety_flag"]:
        assert name in metrics.available_metrics()

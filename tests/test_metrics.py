import math

import pytest

from llmeval.cases import EvalCase
from llmeval import metrics


def make_case(output="", reference=None, context=None, metadata=None):
    return EvalCase(prompt="p", output=output, reference=reference,
                    context=context, metadata=metadata or {})


def test_token_f1_perfect_and_zero():
    c = make_case("the cat sat on the mat", "the cat sat on the mat")
    assert metrics.token_f1(c) == pytest.approx(1.0)
    c2 = make_case("completely different words here", "the cat sat on mat")
    assert metrics.token_f1(c2) == 0.0


def test_token_f1_no_reference_returns_zero():
    assert metrics.token_f1(make_case("anything")) == 0.0


def test_rouge_l_orders_by_subsequence():
    ordered = make_case("a b c d", "a b c d")
    scrambled = make_case("d c b a", "a b c d")
    assert metrics.rouge_l(ordered) > metrics.rouge_l(scrambled)


def test_faithfulness_detects_ungrounded_output():
    grounded = make_case("open daily seven to eight", context="the shop is open daily seven to eight")
    hallucinated = make_case("free parking lot open all night", context="metered street parking only")
    assert metrics.faithfulness(grounded) > 0.8
    assert metrics.faithfulness(hallucinated) < 0.5


def test_faithfulness_no_context_is_one():
    assert metrics.faithfulness(make_case("x y z")) == 1.0


def test_keyword_coverage():
    c = make_case("please call 415 555 0132 for reservations",
                  metadata={"required_keywords": ["reservations", "415 555 0132"]})
    assert metrics.keyword_coverage(c) == pytest.approx(1.0)
    c2 = make_case("no phone here", metadata={"required_keywords": ["reservations", "415"]})
    assert metrics.keyword_coverage(c2) == pytest.approx(0.0)


def test_no_refusal_flags_canned_refusal():
    refusal = make_case("I'm sorry, but I can't help with that.")
    normal = make_case("Sure, the park is open until 8pm.")
    assert metrics.no_refusal(refusal) == 0.0
    assert metrics.no_refusal(normal) == 1.0


def test_regex_match():
    c = make_case("phone: (415) 555-0132")
    assert metrics.regex_match(c, pattern=r"\(\d{3}\) \d{3}-\d{4}") == 1.0
    assert metrics.regex_match(c, pattern=r"zzz") == 0.0


def test_conciseness_penalizes_verbosity():
    ref = "open daily seven to eight"
    concise = make_case("open daily seven to eight", ref)
    verbose = make_case(" ".join(["word"] * 100), ref)
    assert metrics.conciseness(concise) == 1.0
    assert metrics.conciseness(verbose) < 0.5


def test_all_metrics_return_unit_interval():
    c = make_case("some output text", "some reference text", context="some context text")
    for name in metrics.available_metrics():
        score = metrics.get_metric(name)(c)
        assert 0.0 <= score <= 1.0, name
        assert not math.isnan(score)


def test_get_unknown_metric_raises():
    with pytest.raises(KeyError):
        metrics.get_metric("does_not_exist")

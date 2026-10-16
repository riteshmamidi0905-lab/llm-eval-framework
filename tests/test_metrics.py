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


# --- regression tests added by the verification pass (known-value checks) -----------------------
from llmeval.cases import EvalCase as _EC
from llmeval import metrics as _M


def _c(out, ref=None, ctx=None):
    return _EC(prompt="q", output=out, reference=ref, context=ctx)


def test_token_f1_identical_stopword_only_text_scores_one():
    assert _M.token_f1(_c("the", "the")) == 1.0            # was 0.0: both sides empty after stop-word removal
    assert _M.token_f1(_c("", "the cat")) == 0.0


def test_token_f1_known_value():
    # tokens after stop-word removal: {cat, sat, mat} vs {cat, mat}: P=2/3, R=1, F1=0.8
    assert abs(_M.token_f1(_c("the cat sat on the mat", "the cat is on the mat")) - 0.8) < 1e-9


def test_rouge_l_known_value():
    # LCS("the cat sat on the mat","the cat is on the mat") = 5 of 6 tokens each -> F1 = 5/6
    assert abs(_M.rouge_l(_c("the cat sat on the mat", "the cat is on the mat")) - 5 / 6) < 1e-9


def test_number_match_handles_thousands_separators_and_trailing_zeros():
    assert _M.number_match(_c("revenue was 1,234.50", "1234.5")) == 1.0   # was 0.0
    assert _M.number_match(_c("the count is 234", "the count is 1,234")) == 0.0   # '1,234' is not 1 and 234
    assert _M.number_match(_c("about 3 items", "no numbers here")) == 1.0


def test_bleu_two_gram_known_value():
    # BLEU-2, no brevity penalty (candidate longer): sqrt((6/7)*(5/6))
    assert abs(_M.bleu(_c("the cat sat on the mat today", "the cat sat on the mat")) - (6 / 7 * 5 / 6) ** 0.5) < 1e-9

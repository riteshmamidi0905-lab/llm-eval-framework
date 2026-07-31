# Metric reference

Every metric is a function `metric(case, **params) -> float` returning a score in
`[0, 1]`. Reference-based metrics return `0.0` when no reference is supplied;
context-based ones return `1.0` when there is nothing to check against.

| Metric | Needs | Measures |
| --- | --- | --- |
| `faithfulness` | context | Fraction of output content grounded in context (hallucination proxy). Higher = better grounded. |
| `token_f1` | reference | Bag-of-words F1 vs reference. |
| `rouge_l` | reference | Longest-common-subsequence F-measure (order-sensitive). |
| `bleu` | reference | Sentence-level BLEU up to `max_n`-grams with brevity penalty. |
| `exact_match` | reference | 1.0 if normalized output == reference. |
| `contains_answer` | reference | 1.0 if the reference appears as a token span in the output. |
| `number_match` | reference | Fraction of reference numbers present in the output (catches figure drift). |
| `keyword_coverage` | `keywords` | Share of required keywords/phrases present. |
| `regex_match` | `pattern` | Output matches an expected pattern. |
| `json_valid` | — | Output parses as JSON (+ `require_keys` present) — structured-output reliability. |
| `answer_relevance` | prompt | Overlap of output with the question's content words (no reference needed). |
| `conciseness` | reference | Penalizes outputs far longer than the reference. |
| `no_refusal` | — | 0.0 on canned/over-triggered refusals, else 1.0. |
| `safety_flag` | — | 0.0 if output contains unsafe terms (placeholder — back with a moderation model). |

## Parameters

Pass metric parameters through a criterion's `params` object in the rubric:

```json
{ "name": "must_cite", "metric": "regex_match", "weight": 0.2,
  "params": { "pattern": "\\bsource:" } }
```

```json
{ "name": "schema", "metric": "json_valid", "weight": 0.3,
  "params": { "require_keys": ["intent", "confidence"] } }
```

## Thresholds vs weights

- **weight** — how much the criterion contributes to the aggregate (normalized).
- **threshold** — an optional hard gate; if the score dips below it, the case
  fails overall regardless of the aggregate. Use for non-negotiables (safety,
  no-refusal, minimum faithfulness).

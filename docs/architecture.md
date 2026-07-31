# Architecture

`llmeval` is organized as a small pipeline with clear seams so each stage can be
swapped independently.

```
            ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐
 cases ───▶ │  Rubric  │──▶│Evaluator │──▶│  Metrics  │──▶│  EvalResult│
 (JSONL)    │(weighted │   │ (scores, │   │ (registry │   │ (summary,  │
            │criteria) │   │aggregate)│   │ of fns /  │   │ per-case)  │
            └──────────┘   └────┬─────┘   │LLM judges)│   └─────┬──────┘
                                │         └───────────┘         │
                                ▼                               ▼
                          custom_scorers                 report.py ──▶ JSON / HTML
                          (LLMJudge, …)                  server.py ──▶ REST API
```

## Components

| Module | Responsibility |
| --- | --- |
| `rubric.py` | `Criterion` + `Rubric`: weighted, thresholded evaluation dimensions, loadable from JSON/YAML so non-engineers can edit guidelines. |
| `cases.py` | `EvalCase` + dataset loaders (JSONL/JSON). |
| `metrics.py` | A **registry** of deterministic scoring functions (`@register`). Each returns `[0,1]`. Add your own with a decorator. |
| `judges.py` | Optional `LLMJudge` — score a criterion with an LLM via any `chat(prompt)->str` callable (OpenAI/Anthropic adapters included). |
| `evaluator.py` | Runs a rubric over cases: weighted aggregation, hard thresholds, pass/fail, per-criterion means. |
| `report.py` | JSON + self-contained interactive HTML dashboard. |
| `server.py` | FastAPI service (`/evaluate`, `/evaluate/html`, `/metrics`). |
| `cli.py` | `python -m llmeval.cli` with `--json`, `--html`, and a `--fail-under` CI gate. |

## Design principles

1. **Deterministic core, optional intelligence.** Heuristic metrics run offline
   in CI with no keys; LLM judges are opt-in for nuanced dimensions.
2. **Everything returns `[0,1]`.** Uniform scoring makes weighting and
   aggregation trivial and comparable across dimensions.
3. **Hard thresholds are separate from weights.** A criterion can contribute
   little to the aggregate yet still hard-fail a case (e.g. a safety gate).
4. **Config over code.** Rubrics are data (JSON/YAML), so evaluation policy is
   editable and version-controllable without touching the engine.

## Extending

- **New metric:** decorate a function with `@register("name")` in `metrics.py`.
- **New backend:** implement `chat(prompt) -> str` and wrap it in `LLMJudge`.
- **New output:** consume `EvalResult` / `to_json(result)` in your own reporter.

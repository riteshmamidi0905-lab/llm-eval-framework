"""Command-line interface: evaluate a dataset against a rubric.

Usage:
    python -m llmeval.cli --rubric rubric.json --cases cases.jsonl
    python -m llmeval.cli --rubric rubric.json --cases cases.jsonl --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cases import load_cases
from .evaluator import Evaluator
from .rubric import Rubric


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate LLM outputs against a rubric.")
    p.add_argument("--rubric", required=True, help="Path to rubric JSON/YAML.")
    p.add_argument("--cases", required=True, help="Path to cases JSONL/JSON.")
    p.add_argument("--threshold", type=float, default=0.7, help="Pass threshold.")
    p.add_argument("--json", dest="json_out", help="Write full report JSON here.")
    p.add_argument("--html", dest="html_out", help="Write an interactive HTML dashboard here.")
    p.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit non-zero if mean score falls below this value (for CI gates).",
    )
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rubric = Rubric.from_file(args.rubric)
    cases = load_cases(args.cases)
    evaluator = Evaluator(rubric, pass_threshold=args.threshold)
    result = evaluator.evaluate(cases)

    summary = result.summary()
    print(json.dumps(summary, indent=2))

    if args.json_out:
        report = {
            "summary": summary,
            "cases": [
                {
                    "case_id": r.case_id,
                    "aggregate": round(r.aggregate, 4),
                    "passed": r.passed,
                    "criteria": [
                        {"name": c.name, "score": round(c.score, 4), "passed": c.passed}
                        for c in r.criteria
                    ],
                }
                for r in result.case_results
            ],
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote detailed report to {args.json_out}", file=sys.stderr)

    if args.html_out:
        from .report import write_html_report

        write_html_report(result, args.html_out, title=f"Evaluation — {rubric.name}")
        print(f"Wrote HTML dashboard to {args.html_out}", file=sys.stderr)

    if args.fail_under is not None and result.mean_score < args.fail_under:
        print(
            f"FAIL: mean score {result.mean_score:.3f} < fail-under {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())

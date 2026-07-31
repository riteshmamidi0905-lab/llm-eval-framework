"""Run the comprehensive benchmark and emit JSON + an HTML dashboard.

    python benchmarks/run_benchmark.py

Writes benchmarks/report.json and benchmarks/report.html, and prints a summary
plus a per-category breakdown (categories come from each case's metadata).
"""

from collections import defaultdict
from pathlib import Path

from llmeval import Evaluator, Rubric, load_cases, write_json, write_html_report

HERE = Path(__file__).parent


def main() -> None:
    rubric = Rubric.from_file(HERE / "rubric.json")
    cases = load_cases(HERE / "dataset.jsonl")
    result = Evaluator(rubric, pass_threshold=0.7).evaluate(cases)

    print("=== Benchmark summary ===")
    for k, v in result.summary().items():
        print(f"{k}: {v}")

    # Per-category pass rate from case metadata.
    buckets = defaultdict(list)
    for r in result.case_results:
        buckets[r.metadata.get("category", "uncategorized")].append(r.passed)
    print("\n=== Pass rate by category ===")
    for cat, passes in sorted(buckets.items()):
        print(f"{cat:12s} {sum(passes)}/{len(passes)}")

    write_json(result, HERE / "report.json")
    write_html_report(result, HERE / "report.html", title="RAG-QA Benchmark")
    print("\nWrote report.json and report.html")


if __name__ == "__main__":
    main()

"""End-to-end example: evaluate a small RAG-QA dataset offline.

Run from the repository root:
    python examples/run_example.py
"""

from pathlib import Path

from llmeval import Evaluator, Rubric, load_cases

HERE = Path(__file__).parent


def main() -> None:
    rubric = Rubric.from_file(HERE / "rubric.json")
    cases = load_cases(HERE / "sample_dataset.jsonl")

    evaluator = Evaluator(rubric, pass_threshold=0.7)
    result = evaluator.evaluate(cases)

    print("=== Run summary ===")
    for k, v in result.summary().items():
        print(f"{k}: {v}")

    print("\n=== Per-case ===")
    for r in result.case_results:
        flags = [c.name for c in r.criteria if c.passed is False]
        status = "PASS" if r.passed else "FAIL"
        note = f"  (failed: {', '.join(flags)})" if flags else ""
        print(f"{r.case_id:8s} {status}  score={r.aggregate:.3f}{note}")

    weakest = min(result.criterion_means().items(), key=lambda kv: kv[1])
    print(f"\nWeakest dimension: {weakest[0]} ({weakest[1]:.3f})")


if __name__ == "__main__":
    main()

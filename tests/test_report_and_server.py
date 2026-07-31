import json
from pathlib import Path

import pytest

from llmeval import Evaluator, Rubric, load_cases, write_html_report, write_json, to_json

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _result():
    rubric = Rubric.from_file(EXAMPLES / "rubric.json")
    cases = load_cases(EXAMPLES / "sample_dataset.jsonl")
    return Evaluator(rubric).evaluate(cases)


def test_to_json_structure():
    data = to_json(_result())
    assert "summary" in data and "cases" in data
    assert len(data["cases"]) == 6
    assert "criteria" in data["cases"][0]


def test_write_json(tmp_path):
    p = write_json(_result(), tmp_path / "r.json")
    loaded = json.loads(Path(p).read_text())
    assert loaded["summary"]["n_cases"] == 6


def test_html_report_is_self_contained(tmp_path):
    p = write_html_report(_result(), tmp_path / "r.html", title="My Report")
    html = Path(p).read_text()
    assert "<html" in html and "My Report" in html
    # Data is embedded (no external fetch needed) and no CDN dependency.
    assert "const DATA =" in html
    assert "http://" not in html and "https://" not in html


# --- server ---------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    import llmeval.server as server
    return TestClient(server.app)


def _payload():
    return {
        "rubric": {"name": "t", "criteria": [
            {"name": "faith", "metric": "faithfulness", "weight": 1.0}]},
        "cases": [
            {"prompt": "q", "output": "open daily seven to eight",
             "context": "the shop is open daily seven to eight", "reference": "seven to eight"}
        ],
    }


def test_server_health_and_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    names = client.get("/metrics").json()["metrics"]
    assert "faithfulness" in names and "bleu" in names


def test_server_evaluate_json(client):
    r = client.post("/evaluate", json=_payload())
    assert r.status_code == 200
    assert r.json()["summary"]["n_cases"] == 1


def test_server_evaluate_html(client):
    r = client.post("/evaluate/html", json=_payload())
    assert r.status_code == 200
    assert "<html" in r.text


def test_server_rejects_bad_rubric(client):
    r = client.post("/evaluate", json={"rubric": {"criteria": []}, "cases": []})
    assert r.status_code == 400

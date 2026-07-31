"""FastAPI server exposing the evaluation platform over HTTP.

Run:
    pip install "llmeval[api]"   # fastapi + uvicorn
    uvicorn llmeval.server:app --reload

Endpoints:
    GET  /health
    GET  /metrics                       -> list available metric names
    POST /evaluate                      -> run a rubric over cases, return JSON report
    POST /evaluate/html                 -> same, but return an HTML dashboard

FastAPI/pydantic are imported lazily so the core library stays dependency-free.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install API extras: pip install "llmeval[api]"') from exc

from .cases import EvalCase
from .evaluator import Evaluator
from .metrics import available_metrics
from .report import render_html, to_json
from .rubric import Rubric

app = FastAPI(title="llmeval", version="0.1.0",
              description="Rubric-driven LLM output evaluation platform.")


class CaseIn(BaseModel):
    prompt: str = ""
    output: str = ""
    reference: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = {}
    case_id: Optional[str] = None


class EvaluateRequest(BaseModel):
    rubric: Dict[str, Any]
    cases: List[CaseIn]
    pass_threshold: float = 0.7
    title: str = "LLM Evaluation Report"


def _run(req: EvaluateRequest):
    try:
        rubric = Rubric.from_dict(req.rubric)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"bad rubric: {exc}") from exc
    if not req.cases:
        raise HTTPException(status_code=400, detail="no cases provided")
    cases = [
        EvalCase(prompt=c.prompt, output=c.output, reference=c.reference,
                 context=c.context, metadata=c.metadata, case_id=c.case_id or str(i))
        for i, c in enumerate(req.cases)
    ]
    try:
        return Evaluator(rubric, pass_threshold=req.pass_threshold).evaluate(cases), rubric
    except KeyError as exc:  # unknown metric name
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    return {"metrics": available_metrics()}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    result, _ = _run(req)
    return to_json(result)


@app.post("/evaluate/html", response_class=HTMLResponse)
def evaluate_html(req: EvaluateRequest) -> "HTMLResponse":
    result, _ = _run(req)
    return HTMLResponse(render_html(result, title=req.title))

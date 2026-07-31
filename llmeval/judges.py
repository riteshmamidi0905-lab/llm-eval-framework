"""Optional LLM-as-judge backend.

This module lets a rubric criterion be scored by an LLM instead of a heuristic.
It is intentionally decoupled from any single vendor: you pass in a callable
``chat(prompt: str) -> str`` and the judge handles prompt construction and
robust score parsing. Concrete adapters for OpenAI/Anthropic are provided but
imported lazily so the core framework has no hard dependency on them.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .cases import EvalCase

ChatFn = Callable[[str], str]

_JUDGE_TEMPLATE = """You are a strict evaluation judge. Score the ASSISTANT
OUTPUT on the criterion "{criterion}" using this guideline:

{guideline}

Return ONLY a JSON object: {{"score": <float 0-1>, "reason": "<short reason>"}}.

--- PROMPT ---
{prompt}
--- CONTEXT ---
{context}
--- ASSISTANT OUTPUT ---
{output}
--- REFERENCE (optional) ---
{reference}
"""

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')


class LLMJudge:
    """Scores a case against a natural-language guideline via an LLM."""

    def __init__(self, chat: ChatFn, criterion: str, guideline: str) -> None:
        self.chat = chat
        self.criterion = criterion
        self.guideline = guideline

    def __call__(self, case: EvalCase, **_: object) -> float:
        prompt = _JUDGE_TEMPLATE.format(
            criterion=self.criterion,
            guideline=self.guideline,
            prompt=case.prompt or "(none)",
            context=case.context or "(none)",
            output=case.output or "(none)",
            reference=case.reference or "(none)",
        )
        raw = self.chat(prompt)
        return self._parse_score(raw)

    @staticmethod
    def _parse_score(raw: str) -> float:
        m = _SCORE_RE.search(raw or "")
        if not m:
            # Fall back to the first bare float in the response.
            m = re.search(r"([0-9]*\.?[0-9]+)", raw or "")
        if not m:
            raise ValueError(f"Could not parse a score from judge output: {raw!r}")
        return max(0.0, min(1.0, float(m.group(1))))


def openai_chat(model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> ChatFn:  # pragma: no cover
    """Return a chat callable backed by the OpenAI API (lazy import)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def _chat(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return _chat


def anthropic_chat(model: str = "claude-3-5-sonnet-latest", api_key: Optional[str] = None) -> ChatFn:  # pragma: no cover
    """Return a chat callable backed by the Anthropic API (lazy import)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def _chat(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    return _chat

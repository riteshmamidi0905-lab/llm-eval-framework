"""Evaluation cases and dataset loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class EvalCase:
    """A single record to be evaluated.

    Attributes:
        prompt: The user/system input that produced the output.
        output: The model's response under evaluation.
        reference: Optional gold/reference answer for reference-based metrics.
        context: Optional retrieved context (for RAG faithfulness checks).
        metadata: Free-form tags (e.g. category, locale, policy area).
    """

    prompt: str
    output: str
    reference: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    case_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalCase":
        return cls(
            prompt=data.get("prompt", ""),
            output=data.get("output", ""),
            reference=data.get("reference"),
            context=data.get("context"),
            metadata=data.get("metadata", {}),
            case_id=data.get("case_id") or data.get("id"),
        )


def load_cases(path: Union[str, Path]) -> List[EvalCase]:
    """Load evaluation cases from a JSONL or JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    cases: List[EvalCase] = []
    if path.suffix == ".jsonl":
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("case_id", str(i))
            cases.append(EvalCase.from_dict(record))
    else:
        data = json.loads(text)
        records = data if isinstance(data, list) else data.get("cases", [])
        for i, record in enumerate(records):
            record.setdefault("case_id", str(i))
            cases.append(EvalCase.from_dict(record))
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases

"""Shared utilities for resumable LLM-based report evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_sha256(*parts: str) -> str:
    content = "\n\n".join(parts).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def completed_pmids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if record.get("status") == "ok" and record.get("pmid"):
                completed.add(str(record["pmid"]))
    return completed


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model response does not contain a JSON object.")
        parsed = json.loads(candidate[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object.")
    return parsed


def request_json_completion(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.0,
    max_attempts: int = 4,
) -> tuple[dict[str, Any], str, int]:
    last_error: Exception | None = None
    retry_messages = list(messages)

    for attempt in range(1, max_attempts + 1):
        content: str | None = None
        try:
            response = client.chat.completions.create(
                model=model,
                messages=retry_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("The API returned an empty response.")
            parsed = extract_json_object(content)
            return validator(parsed), content, attempt
        except Exception as exc:  # API clients expose several provider-specific errors.
            last_error = exc
            if attempt < max_attempts:
                if content:
                    if "exact" in str(exc).casefold() and "quote" in str(exc).casefold():
                        correction = (
                            "The previous JSON used invalid report_evidence. Every "
                            "report_evidence item must be one contiguous passage copied "
                            "character-for-character from the CANDIDATE REPORT, not from "
                            "the reference key point. Do not paraphrase, shorten, combine "
                            "non-contiguous passages, or insert ellipses. You may select a "
                            "different short passage from the report. If the report contains "
                            "no passage supporting the assigned label, correct the label and "
                            "components; use status missing with an empty report_evidence list "
                            "when the key point is absent. Return the complete corrected JSON "
                            "object and no additional text."
                        )
                    else:
                        correction = (
                            "The previous JSON response was invalid: "
                            f"{exc} Correct only the formatting or schema issue "
                            "while preserving the reference-supported meaning. "
                            "Return the complete corrected JSON object and no "
                            "additional text."
                        )
                    retry_messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": correction,
                            },
                        ]
                    )
                time.sleep(min(2 ** (attempt - 1), 8))

    assert last_error is not None
    raise RuntimeError(
        f"No valid response after {max_attempts} attempts: {last_error}"
    ) from last_error

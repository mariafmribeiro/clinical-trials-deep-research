#!/usr/bin/env python3
"""Extract atomic evaluation key points from original Cochrane abstract text."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "evaluation" / "reports"))

from evaluation_common import (
    append_jsonl,
    completed_pmids,
    request_json_completion,
    text_sha256,
    utc_now,
)


DEFAULT_GROUND_TRUTH = REPOSITORY_ROOT / "benchmark" / "benchmark.json"
DEFAULT_DEV_PMIDS = REPOSITORY_ROOT / "benchmark" / "benchmark.json"
REFERENCE_OUTPUT = REPOSITORY_ROOT / "outputs" / "evaluation" / "references"
DEFAULT_SOURCES = REFERENCE_OUTPUT / "cochrane_sections.jsonl"
DEFAULT_OUTPUT = REFERENCE_OUTPUT / "reference_keypoints.jsonl"
DEFAULT_ERRORS = REFERENCE_OUTPUT / "reference_keypoints_errors.jsonl"

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "google/gemma-4-31B-it"
PROMPT_VERSION = "keypoints-v4-original"

ALLOWED_CATEGORIES = {
    "conclusion",
    "comparison",
    "benefit",
    "harm",
    "certainty",
    "limitation",
    "other",
}
ALLOWED_IMPORTANCE = {"core", "supporting"}


SYSTEM_PROMPT = """You prepare fixed evaluation criteria for clinical evidence reports.
Use only the supplied research question and original Cochrane abstract excerpt. Do
not introduce medical knowledge, assumptions, trial results, or recommendations
that are absent from the excerpt. Return only valid JSON."""


def build_user_prompt(research_query: str, source_text: str) -> str:
    return f"""Research question:
{research_query}

Original Cochrane abstract excerpt:
<reference>
{source_text}
</reference>

Extract between 3 and 5 atomic key points that an evidence synthesis should cover
to answer the research question consistently with the reference. The first key
point must state the authors' overall conclusion, including any conclusion that
the evidence is insufficient or too uncertain to determine an effect. Classify
this first point as category "conclusion" and importance "core". Do not replace
the authors' overall interpretation with a stronger conclusion inferred from an
individual statistically significant or non-significant outcome.

Use the remaining points for the main clinically relevant comparisons or
outcomes, important harms when discussed, and material statements about
uncertainty, evidence certainty, or limitations.

Each key point must:
- express one independently assessable proposition;
- preserve the direction and uncertainty of the reference;
- distinguish no clear difference, uncertain evidence, insufficient evidence,
  and evidence of no effect;
- avoid combining unrelated outcomes;
- contain no information absent from the reference;
- be understandable without reading the other key points.

Prioritize propositions needed to answer the research question. Do not create key
points about search methods, publication metadata, study counts, sample sizes, or
generic calls for future research unless they materially change the interpretation
of the evidence. Describe the direction and uncertainty of comparative effects
without including risk ratios, odds ratios, hazard ratios, confidence intervals,
or p-values. Numerical thresholds that define an outcome may be retained when
needed to identify that outcome.

Return exactly this JSON structure:
{{
  "overall_conclusion": "one concise statement preserving the reference's answer and uncertainty",
  "key_points": [
    {{
      "category": "conclusion|comparison|benefit|harm|certainty|limitation|other",
      "importance": "core|supporting",
      "statement": "one atomic reference-supported proposition"
    }}
  ]
}}
"""


def validate_keypoints(payload: dict[str, Any]) -> dict[str, Any]:
    conclusion = payload.get("overall_conclusion")
    if not isinstance(conclusion, str) or len(conclusion.strip()) < 15:
        raise ValueError("overall_conclusion must be a non-empty statement.")

    points = payload.get("key_points")
    if not isinstance(points, list) or not 3 <= len(points) <= 5:
        raise ValueError("key_points must contain between 3 and 5 items.")

    normalized: list[dict[str, str]] = []
    seen_statements: set[str] = set()
    for index, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            raise ValueError(f"Key point {index} must be an object.")

        category = str(point.get("category", "")).strip().lower()
        importance = str(point.get("importance", "")).strip().lower()
        statement = str(point.get("statement", "")).strip()
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Invalid category for key point {index}: {category}")
        if importance not in ALLOWED_IMPORTANCE:
            raise ValueError(f"Invalid importance for key point {index}: {importance}")
        if len(statement) < 15:
            raise ValueError(f"Key point {index} is too short.")

        comparison_key = " ".join(statement.lower().split())
        if comparison_key in seen_statements:
            raise ValueError(f"Key point {index} duplicates an earlier statement.")
        seen_statements.add(comparison_key)
        normalized.append(
            {
                "id": f"KP{index}",
                "category": category,
                "importance": importance,
                "statement": statement,
            }
        )

    if not any(point["importance"] == "core" for point in normalized):
        raise ValueError("At least one key point must have core importance.")
    if normalized[0]["category"] != "conclusion":
        raise ValueError("The first key point must use the conclusion category.")
    if normalized[0]["importance"] != "core":
        raise ValueError("The first key point must have core importance.")

    return {
        "overall_conclusion": conclusion.strip(),
        "key_points": normalized,
    }


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def selected_pmids(path: Path) -> list[str]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")

    pmids: list[str] = []
    for item in data:
        value = item.get("pmid") if isinstance(item, dict) else item
        if value is not None:
            pmids.append(str(value))
    return list(dict.fromkeys(pmids))


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return {
        str(record["pmid"]): record
        for record in data
        if isinstance(record, dict) and record.get("pmid") is not None
    }


def load_jsonl_by_pmid(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Source file not found: {path}. Run fetch_reference_sections.py first."
        )

    records: dict[str, dict[str, Any]] = {}
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
            if record.get("status") == "ok" and record.get("pmid") is not None:
                records[str(record["pmid"])] = record
    return records


def source_for_record(
    fetched_source: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if fetched_source:
        selected_text = str(fetched_source.get("selected_text", "")).strip()
        if selected_text:
            return {
                "source_name": "cochrane_pubmed_original",
                "source_type": fetched_source.get("selection_type", "unknown"),
                "source_text": selected_text,
                "source_url": fetched_source.get("source_url", ""),
                "source_fetched_at": fetched_source.get("fetched_at", ""),
                "source_sections": fetched_source.get("selected_sections", []),
            }

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--pmids-file", type=Path, default=DEFAULT_DEV_PMIDS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument(
        "--pmid",
        action="append",
        help="Process one PMID. May be supplied repeatedly and overrides --pmids-file.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--limit", type=int, help="Process at most this many PMIDs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pmids = list(dict.fromkeys(args.pmid)) if args.pmid else selected_pmids(args.pmids_file)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        pmids = pmids[: args.limit]

    ground_truth_by_pmid = load_ground_truth(args.ground_truth)
    source_by_pmid = load_jsonl_by_pmid(args.sources)
    already_done = completed_pmids(args.output)

    work_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_pmids: list[str] = []
    for pmid in pmids:
        if pmid in already_done:
            continue
        ground_truth = ground_truth_by_pmid.get(pmid)
        if ground_truth is None:
            raise ValueError(f"PMID {pmid} is absent from {args.ground_truth}.")
        if not str(ground_truth.get("research_query", "")).strip():
            raise ValueError(f"PMID {pmid} has no research_query.")
        source = source_for_record(
            source_by_pmid.get(pmid),
        )
        if source is None:
            missing_pmids.append(pmid)
        else:
            work_items.append((ground_truth, source))

    print(f"Selected PMIDs: {len(pmids)}")
    print(f"Already completed: {sum(pmid in already_done for pmid in pmids)}")
    print(f"Original sources available: {len(work_items)}")
    print(f"Missing sources: {len(missing_pmids)}")
    print(f"Output: {args.output}")
    if missing_pmids:
        print("Missing PMIDs: " + ", ".join(missing_pmids[:10]))

    if not work_items:
        if missing_pmids:
            print(
                "No processable sources. Run fetch_reference_sections.py.",
                file=sys.stderr,
            )
            return 2
        print("Nothing to do.")
        return 0

    if args.dry_run:
        ground_truth, source = work_items[0]
        print(f"\nDry run for PMID {ground_truth['pmid']}")
        print(f"Source type: {source['source_type']}\n")
        print(
            build_user_prompt(
                str(ground_truth["research_query"]),
                str(source["source_text"]),
            )
        )
        return 0

    api_key = os.environ.get("GEMMA_API_KEY")
    if not api_key:
        print("GEMMA_API_KEY is not set.", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "The 'openai' package is not installed. Activate .venv or install "
            "requirements.txt.",
            file=sys.stderr,
        )
        return 2

    base_url = (
        os.environ.get("GEMMA_API_BASE")
        or os.environ.get("GEMMA_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)
    client = OpenAI(base_url=base_url, api_key=api_key)

    failures = 0
    total = len(work_items)
    for position, (ground_truth, source) in enumerate(work_items, start=1):
        pmid = str(ground_truth["pmid"])
        print(f"[{position}/{total}] PMID {pmid} ({source['source_type']})")
        source_text = str(source["source_text"])
        research_query = str(ground_truth["research_query"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(research_query, source_text),
            },
        ]
        try:
            parsed, raw_response, attempts = request_json_completion(
                client=client,
                model=model,
                messages=messages,
                validator=validate_keypoints,
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
            )
            append_jsonl(
                args.output,
                {
                    "status": "ok",
                    "pmid": pmid,
                    "title": ground_truth.get("title", ""),
                    "research_query": research_query,
                    "reference_source": source["source_name"],
                    "reference_source_type": source["source_type"],
                    "reference_source_url": source["source_url"],
                    "reference_source_fetched_at": source["source_fetched_at"],
                    "reference_sections": source["source_sections"],
                    "reference_text": source_text,
                    "reference_sha256": text_sha256(research_query, source_text),
                    "extractor_model": model,
                    "extractor_base_url": base_url,
                    "prompt_version": PROMPT_VERSION,
                    "created_at": utc_now(),
                    "attempts": attempts,
                    **parsed,
                    "raw_response": raw_response,
                },
            )
        except Exception as exc:
            failures += 1
            append_jsonl(
                args.errors,
                {
                    "status": "error",
                    "pmid": pmid,
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "reference_source_type": source["source_type"],
                    "created_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"  ERROR: {exc}", file=sys.stderr)

        if args.sleep_seconds > 0 and position < total:
            time.sleep(args.sleep_seconds)

    if missing_pmids:
        print(
            f"Skipped {len(missing_pmids)} PMIDs without an original source. "
            "Fetch them before rerunning.",
            file=sys.stderr,
        )
    print(f"Finished with {total - failures} successes and {failures} API failures.")
    return 1 if failures or missing_pmids else 0


if __name__ == "__main__":
    raise SystemExit(main())

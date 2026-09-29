#!/usr/bin/env python3
"""Prepare fixed result-aware contexts containing only mapped trials.

This is a known-trial diagnostic: records are fetched from the same
OpenSearch snapshot by exact NCT identifier, without BM25 retrieval, reranking,
learning extraction, or model-dependent selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
EXPERIMENTS_DIR = REPOSITORY_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from search_api import (  # noqa: E402
    OPENSEARCH_INDEX_NAME,
    SOURCE_FIELDS,
    build_trial_card,
    client,
)
from generate_evidence_only_reports import make_result_aware  # noqa: E402


NCT_RE = re.compile(r"^NCT\d{8}$", re.IGNORECASE)


FIELD_LIMITS = {
    "title": 220,
    "conditions": 220,
    "interventions": 280,
    "design": 220,
    "primary_outcomes": 320,
    "secondary_outcomes": 220,
    "summary": 420,
    "outcome_results": 500,
    "safety_results": 220,
    "eligibility": 260,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed result-aware contexts from exact mapped-NCT lookups."
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "benchmark.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "known_trial_contexts",
    )
    parser.add_argument("--index-name", default=OPENSEARCH_INDEX_NAME)
    parser.add_argument("--max-chars-per-trial", type=int, default=1600)
    parser.add_argument("--max-context-words", type=int, default=9000)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value if item is not None)
    return re.sub(r"\s+", " ", str(value)).strip()


def truncate_text(value: Any, max_chars: int) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return (shortened or text[:max_chars].rstrip()) + " ..."


def extract_card_field(raw_content: str, label: str, max_chars: int) -> str:
    pattern = re.compile(
        rf"^{re.escape(label)}:\s*(.+)$", re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(raw_content)
    return truncate_text(match.group(1), max_chars) if match else ""


def compact_trial_block(
    source: dict[str, Any],
    index: int,
    max_chars: int,
) -> str:
    """Mirror the final trial-level formatter used by GPT-Researcher."""
    raw_content = build_trial_card(source)
    nct_id = clean_text(source.get("nct_id")).upper()
    title = truncate_text(
        source.get("brief_title")
        or source.get("official_title")
        or extract_card_field(raw_content, "Brief title", FIELD_LIMITS["title"]),
        FIELD_LIMITS["title"],
    )
    conditions = extract_card_field(
        raw_content, "Conditions", FIELD_LIMITS["conditions"]
    )
    interventions = extract_card_field(
        raw_content, "Interventions", FIELD_LIMITS["interventions"]
    )
    study_type = extract_card_field(raw_content, "Study type", 120)
    design = extract_card_field(raw_content, "Study design", FIELD_LIMITS["design"])
    primary_outcomes = extract_card_field(
        raw_content, "Primary outcomes", FIELD_LIMITS["primary_outcomes"]
    )
    secondary_outcomes = extract_card_field(
        raw_content, "Secondary outcomes", FIELD_LIMITS["secondary_outcomes"]
    )
    summary = extract_card_field(raw_content, "Brief summary", FIELD_LIMITS["summary"])
    outcome_results = extract_card_field(
        raw_content,
        "Clinical outcome results",
        FIELD_LIMITS["outcome_results"],
    )
    safety_results = extract_card_field(
        raw_content,
        "Clinical safety results",
        FIELD_LIMITS["safety_results"],
    )
    eligibility = extract_card_field(
        raw_content,
        "Eligibility / population",
        FIELD_LIMITS["eligibility"],
    )

    design_text = "; ".join(part for part in (study_type, design) if part)
    lines = [
        f"### Trial {index}: {nct_id}",
        f"Source: local://{nct_id}",
        f"Title: {title}" if title else "",
        f"Intervention/comparator: {interventions}" if interventions else "",
        f"Posted outcome results: {outcome_results}" if outcome_results else "",
        f"Posted safety results: {safety_results}" if safety_results else "",
        f"Population/condition: {conditions}" if conditions else "",
        f"Primary outcomes: {primary_outcomes}" if primary_outcomes else "",
        f"Secondary outcomes: {secondary_outcomes}" if secondary_outcomes else "",
        f"Study type/design: {design_text}" if design_text else "",
        f"Brief summary: {summary}" if summary else "",
        f"Eligibility/population details: {eligibility}" if eligibility else "",
    ]

    kept: list[str] = []
    used_chars = 0
    for line in (line for line in lines if line):
        separator_chars = 1 if kept else 0
        remaining = max_chars - used_chars - separator_chars
        if remaining <= 0:
            break
        if len(line) > remaining:
            if remaining < 40:
                break
            line = truncate_text(line, remaining)[:remaining]
        kept.append(line)
        used_chars += separator_chars + len(line)

    return make_result_aware("\n".join(kept))


def load_selection(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty list in {path}")

    reviews: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()
    for entry in payload:
        pmid = str(entry.get("pmid") or "").strip()
        question = str(entry.get("research_query") or "").strip()
        raw_ncts = (
            entry.get("nct_ids")
            or entry.get("sections", {}).get("included_nct")
            or []
        )
        nct_ids = []
        for value in raw_ncts:
            nct_id = str(value).upper().strip()
            if not NCT_RE.fullmatch(nct_id):
                raise ValueError(f"Invalid NCT ID for PMID {pmid}: {value!r}")
            if nct_id not in nct_ids:
                nct_ids.append(nct_id)

        if not pmid or pmid in seen_pmids:
            raise ValueError(f"Missing or duplicate PMID: {pmid!r}")
        if not question:
            raise ValueError(f"Missing research_query for PMID {pmid}")
        if not nct_ids:
            raise ValueError(f"No mapped NCT IDs for PMID {pmid}")

        seen_pmids.add(pmid)
        reviews.append({"pmid": pmid, "question": question, "nct_ids": nct_ids})
    return reviews


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return sources
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            nct_id = clean_text(source.get("nct_id")).upper()
            if not NCT_RE.fullmatch(nct_id):
                raise ValueError(f"Invalid cached NCT at {path}:{line_number}")
            sources[nct_id] = source
    return sources


def exact_lookup(
    nct_ids: list[str],
    index_name: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Fetch exact IDs, trying both common keyword mappings."""
    expected = set(nct_ids)
    found: dict[str, dict[str, Any]] = {}
    fields_used: list[str] = []

    for field in ("nct_id", "nct_id.keyword"):
        missing = sorted(expected - set(found))
        if not missing:
            break
        body = {
            "size": len(missing),
            "_source": SOURCE_FIELDS,
            "query": {"terms": {field: missing}},
        }
        response = client.search(index=index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        if hits:
            fields_used.append(field)
        for hit in hits:
            source = hit.get("_source") or {}
            nct_id = clean_text(source.get("nct_id")).upper()
            if nct_id in expected:
                found[nct_id] = source

    return found, fields_used


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    reviews = load_selection(args.selection)
    expected_ids = list(
        dict.fromkeys(nct_id for review in reviews for nct_id in review["nct_ids"])
    )

    cache_dir = args.output_dir / "cache"
    cache_path = cache_dir / "ground_truth_trial_sources.jsonl"
    sources = {} if args.refresh_cache else load_cache(cache_path)
    missing_from_cache = [nct_id for nct_id in expected_ids if nct_id not in sources]
    fields_used: list[str] = []
    if missing_from_cache:
        fetched, fields_used = exact_lookup(missing_from_cache, args.index_name)
        sources.update(fetched)

    missing_ids = [nct_id for nct_id in expected_ids if nct_id not in sources]
    extra_ids = sorted(set(sources) - set(expected_ids))
    if missing_ids:
        raise RuntimeError(
            f"Exact OpenSearch lookup missed {len(missing_ids)} NCT IDs: "
            + ", ".join(missing_ids)
        )

    prepared: list[dict[str, Any]] = []
    for review in reviews:
        blocks = [
            (
                nct_id,
                compact_trial_block(
                    sources[nct_id],
                    index=index,
                    max_chars=args.max_chars_per_trial,
                ),
            )
            for index, nct_id in enumerate(review["nct_ids"], start=1)
        ]
        context = "\n\n".join(block for _, block in blocks).strip() + "\n"
        context_words = len(context.split())
        if context_words > args.max_context_words:
            raise RuntimeError(
                f"PMID {review['pmid']} requires {context_words} context words, "
                f"above the {args.max_context_words}-word limit. No block was removed."
            )
        kept_ids = [nct_id for nct_id, _ in blocks]
        if kept_ids != review["nct_ids"]:
            raise AssertionError(f"Ground-truth ordering changed for PMID {review['pmid']}")

        outcome_posted = len(
            re.findall(r"^RESULT STATUS:.*OUTCOME=POSTED", context, re.MULTILINE)
        )
        safety_posted = len(
            re.findall(r"^RESULT STATUS:.*SAFETY=POSTED", context, re.MULTILINE)
        )
        prepared.append(
            {
                **review,
                "context": context,
                "context_words": context_words,
                "context_chars": len(context),
                "outcome_results_posted_trials": outcome_posted,
                "safety_results_posted_trials": safety_posted,
                "kept_nct_ids": kept_ids,
            }
        )

    print(f"Selection: {args.selection.resolve()}")
    print(f"OpenSearch index: {args.index_name}")
    print(f"Reviews: {len(prepared)}")
    print(f"Unique mapped NCT IDs: {len(expected_ids)}")
    print(f"Exact lookup fields used: {', '.join(fields_used) if fields_used else 'cache'}")
    print(f"Missing NCT IDs: {len(missing_ids)}")
    print(f"Largest review: {max(len(item['nct_ids']) for item in prepared)} trials")
    print(f"Largest context: {max(item['context_words'] for item in prepared)} words")
    print(f"Output: {args.output_dir.resolve()}")

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    if args.output_dir.exists() and not args.overwrite:
        existing = list((args.output_dir / "contexts").glob("*_evidence_only_context.txt"))
        if existing:
            raise FileExistsError(
                f"Prepared contexts already exist in {args.output_dir}. Use --overwrite."
            )

    contexts_dir = args.output_dir / "contexts"
    metadata_dir = args.output_dir / "metadata"
    for directory in (cache_dir, contexts_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as handle:
        for nct_id in expected_ids:
            handle.write(json.dumps(sources[nct_id], ensure_ascii=False) + "\n")

    manifest_rows: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for item in prepared:
        pmid = item["pmid"]
        context_path = contexts_dir / f"{pmid}_evidence_only_context.txt"
        metadata_path = metadata_dir / f"{pmid}_metadata.json"
        context_path.write_text(item["context"], encoding="utf-8")
        metadata = {
            "pmid": pmid,
            "question": item["question"],
            "condition": "ground_truth_trial_context",
            "representation": "result_aware",
            "learnings_included": False,
            "selection_method": "exact_ground_truth_nct_lookup",
            "opensearch_index": args.index_name,
            "expected_nct_ids": item["nct_ids"],
            "retrieved_nct_ids": item["nct_ids"],
            "kept_nct_ids": item["kept_nct_ids"],
            "missing_nct_ids": [],
            "available_trial_blocks": len(item["nct_ids"]),
            "kept_trial_blocks": len(item["kept_nct_ids"]),
            "ground_truth_context_recall": 1.0,
            "context_words": item["context_words"],
            "context_chars": item["context_chars"],
            "max_context_words": args.max_context_words,
            "max_chars_per_trial": args.max_chars_per_trial,
            "outcome_results_posted_trials": item["outcome_results_posted_trials"],
            "safety_results_posted_trials": item["safety_results_posted_trials"],
            "context_sha256": sha256_text(item["context"]),
            "context_path": str(context_path.resolve()),
            "generated_at": generated_at,
        }
        write_json(metadata_path, metadata)
        manifest_rows.append(metadata)

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for metadata in manifest_rows:
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    audit = {
        "selection": str(args.selection.resolve()),
        "opensearch_index": args.index_name,
        "reviews": len(prepared),
        "unique_expected_nct_ids": len(expected_ids),
        "unique_retrieved_nct_ids": len(expected_ids),
        "missing_nct_ids": missing_ids,
        "extra_cached_nct_ids_ignored": extra_ids,
        "largest_review_trial_count": max(len(item["nct_ids"]) for item in prepared),
        "largest_context_words": max(item["context_words"] for item in prepared),
        "max_context_words": args.max_context_words,
        "max_chars_per_trial": args.max_chars_per_trial,
        "generated_at": generated_at,
    }
    write_json(args.output_dir / "preparation_audit.json", audit)
    print("Prepared all contexts with mapped-trial context recall = 1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

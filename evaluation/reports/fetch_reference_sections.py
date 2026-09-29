#!/usr/bin/env python3
"""Fetch and preserve the original PubMed abstract sections for selected reviews."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from evaluation_common import append_jsonl, completed_pmids, text_sha256, utc_now


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DEV_PMIDS = REPOSITORY_ROOT / "data" / "benchmark.json"
REFERENCE_OUTPUT = REPOSITORY_ROOT / "outputs" / "evaluation" / "references"
DEFAULT_OUTPUT = REFERENCE_OUTPUT / "cochrane_sections.jsonl"
DEFAULT_ERRORS = REFERENCE_OUTPUT / "cochrane_sections_errors.jsonl"

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SOURCE_VERSION = "pubmed-sections-v1"


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


def normalize_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", label.upper()).strip()


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def first_section(sections: dict[str, str], labels: tuple[str, ...]) -> str:
    for label in labels:
        text = sections.get(label, "").strip()
        if text:
            return text
    return ""


def extract_embedded_result_sections(text: str) -> tuple[str, str]:
    """Recover Cochrane headings embedded inside a broader PubMed XML section."""
    main_match = re.search(r"\bMAIN\s+RESULTS\s*:\s*", text, flags=re.IGNORECASE)
    conclusions_match = re.search(
        r"\bAUTHORS?['\u2019]?\s+CONCLUSIONS?\s*:\s*",
        text,
        flags=re.IGNORECASE,
    )

    main_results = ""
    authors_conclusions = ""
    if main_match:
        end = conclusions_match.start() if conclusions_match else len(text)
        main_results = text[main_match.end() : end].strip()
    if conclusions_match:
        authors_conclusions = text[conclusions_match.end() :].strip()
    return main_results, authors_conclusions


def parse_pubmed_record(xml_bytes: bytes, requested_pmid: str) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    articles = root.findall(".//PubmedArticle")
    if not articles:
        raise ValueError(f"PubMed returned no article for PMID {requested_pmid}.")

    article_record = None
    for candidate in articles:
        returned_pmid = element_text(candidate.find("./MedlineCitation/PMID"))
        if returned_pmid == requested_pmid:
            article_record = candidate
            break
    if article_record is None:
        article_record = articles[0]

    returned_pmid = element_text(article_record.find("./MedlineCitation/PMID"))
    article = article_record.find("./MedlineCitation/Article")
    if article is None:
        raise ValueError(f"PMID {requested_pmid} has no Article element.")

    title = element_text(article.find("./ArticleTitle"))
    ordered_sections: list[dict[str, str]] = []
    sections: dict[str, str] = {}

    for abstract_part in article.findall("./Abstract/AbstractText"):
        text = element_text(abstract_part)
        if not text:
            continue
        raw_label = (
            abstract_part.attrib.get("Label")
            or abstract_part.attrib.get("NlmCategory")
            or ""
        ).strip()
        if raw_label.upper() == "UNASSIGNED":
            raw_label = ""
        normalized_label = normalize_label(raw_label)
        ordered_sections.append(
            {
                "label": raw_label,
                "normalized_label": normalized_label,
                "text": text,
            }
        )
        if normalized_label:
            previous = sections.get(normalized_label)
            sections[normalized_label] = f"{previous}\n\n{text}" if previous else text

    if not ordered_sections:
        raise ValueError(f"PMID {requested_pmid} has no abstract text in PubMed.")

    full_abstract = "\n\n".join(
        f"{part['label']}: {part['text']}" if part["label"] else part["text"]
        for part in ordered_sections
    )
    main_results = first_section(sections, ("MAIN RESULTS", "RESULTS"))
    authors_conclusions = first_section(
        sections,
        ("AUTHORS CONCLUSIONS", "CONCLUSIONS", "CONCLUSION"),
    )
    main_from_xml_label = bool(main_results)
    conclusions_from_xml_label = bool(authors_conclusions)

    embedded_main, embedded_conclusions = extract_embedded_result_sections(
        full_abstract
    )
    if not main_results:
        main_results = embedded_main
    if not authors_conclusions:
        authors_conclusions = embedded_conclusions

    if main_from_xml_label and conclusions_from_xml_label:
        section_detection = "xml_labels"
    elif main_results and authors_conclusions:
        section_detection = "embedded_headings"
    else:
        section_detection = "full_abstract_fallback"

    if main_results and authors_conclusions:
        selection_type = "main_results_and_authors_conclusions"
        selected_text = (
            f"MAIN RESULTS:\n{main_results}\n\n"
            f"AUTHORS' CONCLUSIONS:\n{authors_conclusions}"
        )
        selected_sections = ["MAIN RESULTS", "AUTHORS CONCLUSIONS"]
    else:
        selection_type = "full_abstract_fallback"
        selected_text = full_abstract
        selected_sections = [
            part["normalized_label"] or "UNLABELLED" for part in ordered_sections
        ]

    return {
        "status": "ok",
        "pmid": returned_pmid or requested_pmid,
        "requested_pmid": requested_pmid,
        "title": title,
        "source": "PubMed",
        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{requested_pmid}/",
        "source_version": SOURCE_VERSION,
        "fetched_at": utc_now(),
        "abstract_sections": ordered_sections,
        "section_labels": [
            part["normalized_label"] or "UNLABELLED" for part in ordered_sections
        ],
        "main_results": main_results,
        "authors_conclusions": authors_conclusions,
        "full_abstract": full_abstract,
        "selection_type": selection_type,
        "section_detection": section_detection,
        "selected_sections": selected_sections,
        "selected_text": selected_text,
        "selected_text_sha256": text_sha256(selected_text),
    }


def build_efetch_url(pmid: str, email: str, api_key: str | None) -> str:
    parameters = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "tool": "rq3_report_evaluation",
        "email": email,
    }
    if api_key:
        parameters["api_key"] = api_key
    return f"{EFETCH_URL}?{urllib.parse.urlencode(parameters)}"


def fetch_pubmed_xml(
    *,
    pmid: str,
    email: str,
    api_key: str | None,
    timeout: float,
    max_attempts: int,
) -> tuple[bytes, int]:
    url = build_efetch_url(pmid, email, api_key)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"rq3-report-evaluation/1.0 ({email})"},
    )
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), attempt
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

    assert last_error is not None
    raise RuntimeError(
        f"PubMed request failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmids-file", type=Path, default=DEFAULT_DEV_PMIDS)
    parser.add_argument(
        "--pmid",
        action="append",
        help="Fetch one PMID. May be supplied repeatedly and overrides --pmids-file.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--limit", type=int, help="Fetch at most this many PMIDs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch selected PMIDs even when they already have a successful record.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        help="Delay between PMIDs. Defaults to 0.4 s, or 0.12 s with NCBI_API_KEY.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pmids = list(dict.fromkeys(args.pmid)) if args.pmid else selected_pmids(args.pmids_file)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        pmids = pmids[: args.limit]

    already_done = completed_pmids(args.output)
    pending = pmids if args.refresh else [pmid for pmid in pmids if pmid not in already_done]

    print(f"Selected PMIDs: {len(pmids)}")
    print(f"Previously completed: {sum(pmid in already_done for pmid in pmids)}")
    print(f"Refresh enabled: {args.refresh}")
    print(f"Pending: {len(pending)}")
    print(f"Output: {args.output}")

    if not pending:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        print(f"First pending PMID: {pending[0]}")
        print(f"Request endpoint: {EFETCH_URL}")
        print("No network request was made.")
        return 0

    email = os.environ.get("NCBI_EMAIL", "").strip()
    if not email:
        print(
            "NCBI_EMAIL is not set. Export a contact email before fetching PubMed.",
            file=sys.stderr,
        )
        return 2

    api_key = os.environ.get("NCBI_API_KEY") or None
    sleep_seconds = args.sleep_seconds
    if sleep_seconds is None:
        sleep_seconds = 0.12 if api_key else 0.4

    failures = 0
    total = len(pending)
    for position, pmid in enumerate(pending, start=1):
        print(f"[{position}/{total}] PMID {pmid}")
        try:
            xml_bytes, attempts = fetch_pubmed_xml(
                pmid=pmid,
                email=email,
                api_key=api_key,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
            )
            record = parse_pubmed_record(xml_bytes, pmid)
            record["attempts"] = attempts
            append_jsonl(args.output, record)
            print(
                f"  {record['selection_type']}: "
                f"{', '.join(record['selected_sections'])}"
            )
        except Exception as exc:
            failures += 1
            append_jsonl(
                args.errors,
                {
                    "status": "error",
                    "pmid": pmid,
                    "source": "PubMed",
                    "source_version": SOURCE_VERSION,
                    "created_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"  ERROR: {exc}", file=sys.stderr)

        if sleep_seconds > 0 and position < total:
            time.sleep(sleep_seconds)

    print(f"Finished with {total - failures} successes and {failures} failures.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

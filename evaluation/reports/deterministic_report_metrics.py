"""Deterministic report-level metrics for the RQ3 evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", flags=re.IGNORECASE)
REPORT_FILE_PATTERN = re.compile(
    r"^(?:(?P<index>\d+)_)?(?P<pmid>\d+)_generated_report\.md$"
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.upper() for value in values))


def extract_nct_ids(text: str) -> list[str]:
    return ordered_unique(NCT_PATTERN.findall(text or ""))


def split_report_sections(report_text: str) -> dict[str, Any]:
    summary_match = re.search(
        r"(?im)^\s*##\s+Summary\s*$", report_text
    )
    references_match = re.search(
        r"(?im)^\s*##\s+References\s*$", report_text
    )

    if summary_match:
        summary_start = summary_match.end()
    else:
        summary_start = 0

    if references_match and references_match.start() >= summary_start:
        summary_end = references_match.start()
        references_text = report_text[references_match.end() :].strip()
    else:
        summary_end = len(report_text)
        references_text = ""

    summary_text = report_text[summary_start:summary_end].strip()
    paragraphs = [
        block.strip()
        for block in re.split(r"\n\s*\n", summary_text)
        if block.strip() and not re.match(r"^#{1,6}\s+", block.strip())
    ]
    has_bullets = bool(re.search(r"(?m)^\s*[-*+]\s+", summary_text))

    return {
        "summary_text": summary_text,
        "references_text": references_text,
        "summary_paragraph_count": len(paragraphs),
        "summary_word_count": len(re.findall(r"\b\w+\b", summary_text)),
        "has_summary_heading": summary_match is not None,
        "has_references_heading": references_match is not None,
        "summary_has_bullets": has_bullets,
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else np.nan


def normalize_ids(values: Any) -> set[str]:
    if not values:
        return set()
    return {str(value).upper() for value in values if value}


def load_ground_truth(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}.")

    by_pmid: dict[str, dict[str, Any]] = {}
    eligible_pmids: list[str] = []
    for record in records:
        if not isinstance(record, dict) or record.get("pmid") is None:
            continue
        pmid = str(record["pmid"])
        by_pmid[pmid] = record
        if normalize_ids(record.get("nct_ids")):
            eligible_pmids.append(pmid)
    return by_pmid, list(dict.fromkeys(eligible_pmids))


def load_selected_pmids(path: Path) -> set[str]:
    values = load_json(path)
    if not isinstance(values, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    result = set()
    for value in values:
        pmid = value.get("pmid") if isinstance(value, dict) else value
        if pmid is not None:
            result.add(str(pmid))
    return result


def discover_report_files(
    run_directories: dict[str, list[Path]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for model, directories in run_directories.items():
        for run_directory in directories:
            reports_directory = run_directory / "reports"
            if not reports_directory.exists():
                issues.append(
                    {
                        "model": model,
                        "pmid": "",
                        "issue": "missing_reports_directory",
                        "path": str(reports_directory),
                    }
                )
                continue

            for report_path in reports_directory.glob("*_generated_report.md"):
                match = REPORT_FILE_PATTERN.match(report_path.name)
                if not match:
                    issues.append(
                        {
                            "model": model,
                            "pmid": "",
                            "issue": "unrecognized_report_filename",
                            "path": str(report_path),
                        }
                    )
                    continue
                index = match.group("index")
                pmid = match.group("pmid")
                contexts_directory = run_directory / "contexts"
                if index is not None:
                    context_path = (
                        contexts_directory / f"{index}_{pmid}_research_context.txt"
                    )
                else:
                    context_candidates = list(
                        contexts_directory.glob(f"{pmid}_*_context.txt")
                    )
                    context_path = (
                        max(context_candidates, key=lambda path: path.stat().st_mtime)
                        if context_candidates
                        else contexts_directory / f"{pmid}_evidence_only_context.txt"
                    )
                rows.append(
                    {
                        "model": model,
                        "pmid": pmid,
                        "run_directory": str(run_directory),
                        "report_path": str(report_path),
                        "context_path": str(context_path),
                        "report_mtime": report_path.stat().st_mtime,
                    }
                )

    discovered = pd.DataFrame(rows)
    if discovered.empty:
        return discovered, pd.DataFrame(issues)

    duplicate_mask = discovered.duplicated(["model", "pmid"], keep=False)
    for _, row in discovered.loc[duplicate_mask].iterrows():
        issues.append(
            {
                "model": row["model"],
                "pmid": row["pmid"],
                "issue": "duplicate_report_candidate",
                "path": row["report_path"],
            }
        )

    discovered = (
        discovered.sort_values("report_mtime")
        .drop_duplicates(["model", "pmid"], keep="last")
        .reset_index(drop=True)
    )
    return discovered, pd.DataFrame(issues)


def classify_citations(
    citation_ids: set[str],
    included: set[str],
    ongoing: set[str],
    awaiting: set[str],
    excluded: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for nct_id in citation_ids:
        if nct_id in included:
            counts["included"] += 1
        elif nct_id in ongoing:
            counts["ongoing"] += 1
        elif nct_id in awaiting:
            counts["awaiting"] += 1
        elif nct_id in excluded:
            counts["excluded"] += 1
        else:
            counts["uncategorized"] += 1
    return counts


def evaluate_report(
    discovered_row: dict[str, Any],
    ground_truth: dict[str, Any],
    development_pmids: set[str],
) -> dict[str, Any]:
    report_path = Path(discovered_row["report_path"])
    context_path = Path(discovered_row["context_path"])
    report_text = report_path.read_text(encoding="utf-8", errors="replace")
    context_text = (
        context_path.read_text(encoding="utf-8", errors="replace")
        if context_path.exists()
        else ""
    )

    sections = split_report_sections(report_text)
    body_ids_ordered = extract_nct_ids(sections["summary_text"])
    reference_ids_ordered = extract_nct_ids(sections["references_text"])
    report_ids_ordered = extract_nct_ids(report_text)
    context_ids_ordered = extract_nct_ids(context_text)

    body_ids = set(body_ids_ordered)
    reference_ids = set(reference_ids_ordered)
    report_ids = set(report_ids_ordered)
    context_ids = set(context_ids_ordered)

    gt_sections = ground_truth.get("sections") or {}
    included = normalize_ids(
        gt_sections.get("included_nct") or ground_truth.get("nct_ids")
    )
    ongoing = normalize_ids(gt_sections.get("ongoing_nct"))
    awaiting = normalize_ids(gt_sections.get("awaiting_assessment_nct"))
    excluded = normalize_ids(gt_sections.get("excluded_nct"))

    context_included = context_ids & included
    body_included = body_ids & included
    report_included = report_ids & included
    grounded_body_included = body_ids & context_ids & included
    grounded_report_included = report_ids & context_ids & included
    body_in_context = body_ids & context_ids
    report_in_context = report_ids & context_ids
    body_missing_from_references = body_ids - reference_ids
    references_unused_in_body = reference_ids - body_ids
    body_citation_categories = classify_citations(
        body_ids, included, ongoing, awaiting, excluded
    )
    report_citation_categories = classify_citations(
        report_ids, included, ongoing, awaiting, excluded
    )

    paragraph_requirement_met = 3 <= sections["summary_paragraph_count"] <= 5
    format_compliant = (
        sections["has_summary_heading"]
        and sections["has_references_heading"]
        and paragraph_requirement_met
        and not sections["summary_has_bullets"]
    )

    return {
        **discovered_row,
        "split": "development" if discovered_row["pmid"] in development_pmids else "test",
        "title": ground_truth.get("title", ""),
        "research_query": ground_truth.get("research_query", ""),
        "context_present": context_path.exists(),
        "ground_truth_included_count": len(included),
        "context_unique_nct_count": len(context_ids),
        "body_unique_citation_count": len(body_ids),
        "reference_unique_nct_count": len(reference_ids),
        "report_unique_nct_count": len(report_ids),
        "body_citation_occurrences": len(NCT_PATTERN.findall(sections["summary_text"])),
        "context_included_count": len(context_included),
        "body_included_count": len(body_included),
        "report_included_count": len(report_included),
        "grounded_body_included_count": len(grounded_body_included),
        "grounded_report_included_count": len(grounded_report_included),
        "body_in_context_count": len(body_in_context),
        "report_in_context_count": len(report_in_context),
        "context_included_recall": safe_ratio(len(context_included), len(included)),
        "body_included_recall": safe_ratio(len(body_included), len(included)),
        "report_included_recall": safe_ratio(len(report_included), len(included)),
        "grounded_report_included_recall": safe_ratio(
            len(grounded_report_included), len(included)
        ),
        "conditional_report_included_recall": safe_ratio(
            len(grounded_report_included), len(context_included)
        ),
        "grounded_body_included_recall": safe_ratio(
            len(grounded_body_included), len(included)
        ),
        "conditional_body_included_recall": safe_ratio(
            len(grounded_body_included), len(context_included)
        ),
        "body_context_identifier_rate": safe_ratio(len(body_in_context), len(body_ids)),
        "out_of_context_body_citation_rate": safe_ratio(
            len(body_ids - context_ids), len(body_ids)
        ),
        "body_included_citation_share": safe_ratio(len(body_included), len(body_ids)),
        "body_reference_list_coverage": safe_ratio(
            len(body_ids & reference_ids), len(body_ids)
        ),
        "unused_reference_list_rate": safe_ratio(
            len(references_unused_in_body), len(reference_ids)
        ),
        "body_missing_from_reference_count": len(body_missing_from_references),
        "unused_reference_count": len(references_unused_in_body),
        "body_included_citation_count": body_citation_categories["included"],
        "body_ongoing_citation_count": body_citation_categories["ongoing"],
        "body_awaiting_citation_count": body_citation_categories["awaiting"],
        "body_excluded_citation_count": body_citation_categories["excluded"],
        "body_uncategorized_citation_count": body_citation_categories["uncategorized"],
        "report_included_citation_count": report_citation_categories["included"],
        "report_ongoing_citation_count": report_citation_categories["ongoing"],
        "report_awaiting_citation_count": report_citation_categories["awaiting"],
        "report_excluded_citation_count": report_citation_categories["excluded"],
        "report_uncategorized_citation_count": report_citation_categories[
            "uncategorized"
        ],
        "summary_word_count": sections["summary_word_count"],
        "summary_paragraph_count": sections["summary_paragraph_count"],
        "has_summary_heading": sections["has_summary_heading"],
        "has_references_heading": sections["has_references_heading"],
        "summary_has_bullets": sections["summary_has_bullets"],
        "paragraph_requirement_met": paragraph_requirement_met,
        "format_compliant": format_compliant,
        "ground_truth_included_ids": ";".join(sorted(included)),
        "context_nct_ids": ";".join(context_ids_ordered),
        "body_citation_ids": ";".join(body_ids_ordered),
        "reference_list_nct_ids": ";".join(reference_ids_ordered),
        "report_citation_ids": ";".join(report_ids_ordered),
        "body_included_ids": ";".join(sorted(body_included)),
        "report_included_ids": ";".join(sorted(report_included)),
        "grounded_body_included_ids": ";".join(sorted(grounded_body_included)),
        "grounded_report_included_ids": ";".join(
            sorted(grounded_report_included)
        ),
        "out_of_context_body_citation_ids": ";".join(sorted(body_ids - context_ids)),
    }


def build_report_metrics(
    *,
    ground_truth_path: Path,
    development_pmids_path: Path,
    run_directories: dict[str, list[Path]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ground_truth_by_pmid, eligible_pmids = load_ground_truth(ground_truth_path)
    development_pmids = load_selected_pmids(development_pmids_path)
    discovered, issues = discover_report_files(run_directories)

    metric_rows: list[dict[str, Any]] = []
    if not discovered.empty:
        for row in discovered.to_dict(orient="records"):
            ground_truth = ground_truth_by_pmid.get(row["pmid"])
            if ground_truth is None:
                issues = pd.concat(
                    [
                        issues,
                        pd.DataFrame(
                            [
                                {
                                    "model": row["model"],
                                    "pmid": row["pmid"],
                                    "issue": "pmid_missing_from_ground_truth",
                                    "path": row["report_path"],
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                continue
            metric_rows.append(evaluate_report(row, ground_truth, development_pmids))

    metrics = pd.DataFrame(metric_rows)
    completion_rows = []
    eligible_set = set(eligible_pmids)
    test_set = eligible_set - development_pmids
    for model in run_directories:
        available = (
            set(metrics.loc[metrics["model"] == model, "pmid"])
            if not metrics.empty
            else set()
        )
        for split, expected in (
            ("development", development_pmids & eligible_set),
            ("test", test_set),
            ("all", eligible_set),
        ):
            found = available & expected
            completion_rows.append(
                {
                    "model": model,
                    "split": split,
                    "expected_reports": len(expected),
                    "available_reports": len(found),
                    "missing_reports": len(expected - found),
                    "completion_rate": safe_ratio(len(found), len(expected)),
                }
            )

    completion = pd.DataFrame(completion_rows)
    return metrics, completion, issues


MACRO_METRICS = [
    "context_included_recall",
    "report_included_recall",
    "grounded_report_included_recall",
    "conditional_report_included_recall",
    "body_included_recall",
    "grounded_body_included_recall",
    "conditional_body_included_recall",
    "body_context_identifier_rate",
    "out_of_context_body_citation_rate",
    "body_included_citation_share",
    "body_reference_list_coverage",
    "unused_reference_list_rate",
    "body_unique_citation_count",
    "summary_word_count",
    "summary_paragraph_count",
    "format_compliant",
]


def with_all_split(metrics: pd.DataFrame) -> pd.DataFrame:
    all_rows = metrics.copy()
    all_rows["split"] = "all"
    return pd.concat([metrics, all_rows], ignore_index=True)


def macro_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    expanded = with_all_split(metrics)
    summary = (
        expanded.groupby(["model", "split"], observed=True)
        .agg(reports=("pmid", "count"), **{metric: (metric, "mean") for metric in MACRO_METRICS})
        .reset_index()
    )
    return summary


def micro_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    expanded = with_all_split(metrics)
    for (model, split), group in expanded.groupby(["model", "split"], observed=True):
        gt_total = int(group["ground_truth_included_count"].sum())
        context_gt_total = int(group["context_included_count"].sum())
        body_citation_total = int(group["body_unique_citation_count"].sum())
        rows.append(
            {
                "model": model,
                "split": split,
                "reports": len(group),
                "micro_context_included_recall": safe_ratio(
                    int(group["context_included_count"].sum()), gt_total
                ),
                "micro_body_included_recall": safe_ratio(
                    int(group["body_included_count"].sum()), gt_total
                ),
                "micro_report_included_recall": safe_ratio(
                    int(group["report_included_count"].sum()), gt_total
                ),
                "micro_grounded_report_included_recall": safe_ratio(
                    int(group["grounded_report_included_count"].sum()), gt_total
                ),
                "micro_conditional_report_included_recall": safe_ratio(
                    int(group["grounded_report_included_count"].sum()),
                    context_gt_total,
                ),
                "micro_grounded_body_included_recall": safe_ratio(
                    int(group["grounded_body_included_count"].sum()), gt_total
                ),
                "micro_conditional_body_included_recall": safe_ratio(
                    int(group["grounded_body_included_count"].sum()),
                    context_gt_total,
                ),
                "micro_body_context_identifier_rate": safe_ratio(
                    int(group["body_in_context_count"].sum()), body_citation_total
                ),
            }
        )
    return pd.DataFrame(rows)


def citation_status_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    expanded = with_all_split(metrics)
    count_columns = [
        "report_included_citation_count",
        "report_ongoing_citation_count",
        "report_awaiting_citation_count",
        "report_excluded_citation_count",
        "report_uncategorized_citation_count",
    ]
    summary = (
        expanded.groupby(["model", "split"], observed=True)[count_columns]
        .sum()
        .reset_index()
    )
    total = summary[count_columns].sum(axis=1)
    for column in count_columns:
        summary[column.replace("_count", "_share")] = np.where(
            total > 0, summary[column] / total, np.nan
        )
    summary["total_unique_report_citations_across_reviews"] = total
    return summary


def export_tables(
    *,
    metrics: pd.DataFrame,
    completion: pd.DataFrame,
    issues: pd.DataFrame,
    output_directory: Path,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_report": output_directory / "deterministic_metrics_per_report.csv",
        "completion": output_directory / "report_completion.csv",
        "macro": output_directory / "deterministic_metrics_macro.csv",
        "micro": output_directory / "deterministic_metrics_micro.csv",
        "citation_status": output_directory / "citation_status_summary.csv",
        "issues": output_directory / "deterministic_evaluation_issues.csv",
    }
    metrics.to_csv(paths["per_report"], index=False)
    completion.to_csv(paths["completion"], index=False)
    macro_summary(metrics).to_csv(paths["macro"], index=False)
    micro_summary(metrics).to_csv(paths["micro"], index=False)
    citation_status_summary(metrics).to_csv(paths["citation_status"], index=False)
    issues.to_csv(paths["issues"], index=False)
    return paths

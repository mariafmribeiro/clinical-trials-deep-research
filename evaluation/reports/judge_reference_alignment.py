"""Incremental reference-alignment evaluation against Cochrane key points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from evaluation_common import append_jsonl, request_json_completion, utc_now


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
RUN_ROOT = REPOSITORY_ROOT / "outputs"
DEV_PMIDS_PATH = REPOSITORY_ROOT / "data" / "benchmark.json"
DEFAULT_KEYPOINTS = REPOSITORY_ROOT / "data" / "reference_keypoints.jsonl"
EVALUATION_OUTPUT = RUN_ROOT / "evaluation" / "report_alignment"
DEFAULT_OUTPUT = EVALUATION_OUTPUT / "judge_alignment.jsonl"
DEFAULT_ERRORS = EVALUATION_OUTPUT / "judge_alignment_errors.jsonl"
DEFAULT_REVIEW_CSV = EVALUATION_OUTPUT / "judge_alignment_review.csv"
DEFAULT_KEYPOINT_CSV = EVALUATION_OUTPUT / "judge_alignment_keypoints.csv"

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "google/gemma-4-31B-it"
PROMPT_VERSION = "judge-alignment-v4"

MODEL_RUNS = {
    "Qwen3-8B": [RUN_ROOT / "final_qwen_trial_context_b4_d2"],
    "Fleming-R1-7B": [RUN_ROOT / "final_fleming_trial_context_b4_d2"],
}

REPORT_PATTERN = re.compile(
    r"^(?:(?P<index>\d+)_)?(?P<pmid>\d+)_generated_report\.md$"
)
KEYPOINT_STATUSES = {
    "covered",
    "partially_covered",
    "missing",
    "contradicted",
}
DIFFERENCE_TYPES = {
    "none",
    "omission",
    "scope",
    "certainty",
    "factual_conflict",
    "direction_conflict",
    "evidence_sufficiency_conflict",
    "mixed",
}
CONCLUSION_LABELS = {
    "aligned",
    "partially_aligned",
    "not_addressed",
    "contradicted",
}
UNCERTAINTY_LABELS = {
    "matched",
    "stronger_than_reference",
    "weaker_than_reference",
    "not_assessable",
}
QUESTION_RELEVANCE_LABELS = {
    "direct",
    "partial",
    "off_topic",
}
QUESTION_RELEVANCE_SCORES = {
    "direct": 1.0,
    "partial": 0.5,
    "off_topic": 0.0,
}
STATUS_SCORES = {
    "covered": 1.0,
    "partially_covered": 0.5,
    "missing": 0.0,
    "contradicted": 0.0,
}

SYSTEM_PROMPT = """You are an evaluator of clinical evidence synthesis reports.
Assess the candidate report only against the supplied research question and fixed
Cochrane-derived reference criteria. Do not use external medical knowledge. Judge
clinical content rather than fluency, length, formatting, or writing style. Treat
each criterion independently. A difference in confidence alone is not a
contradiction when the direction of effect remains compatible. Reserve
'contradicted' for an explicit factual conflict, an opposite effect direction, or
a clear claim of benefit or harm when the reference criterion's central finding
is that the evidence is insufficient, unavailable, or uncertain. Return only
valid JSON. When a reference criterion contains several material propositions,
evaluate every proposition before assigning the overall label."""


def build_user_prompt(reference: dict[str, Any], report_text: str) -> str:
    key_points = [
        {
            "id": point["id"],
            "category": point["category"],
            "importance": point["importance"],
            "statement": point["statement"],
        }
        for point in reference["key_points"]
    ]
    return f"""Research question:
{reference['research_query']}

Reference overall conclusion:
{reference['overall_conclusion']}

Fixed reference key points:
{json.dumps(key_points, ensure_ascii=False, indent=2)}

Candidate report:
<report>
{report_text}
</report>

First assess whether the candidate report addresses the research question:
- direct: the report addresses the central population, intervention or exposure,
  comparison, and outcomes needed to answer the question. It may still conclude
  that the available evidence is insufficient;
- partial: the report addresses the correct general topic, but omits or
  substantially diverts from at least one central component of the question;
- off_topic: the report predominantly addresses a different population,
  intervention, comparison, outcome, or clinical problem.

Judge question relevance independently from agreement with the reference. A
topically relevant report may still omit or contradict reference findings.

Evaluate every fixed key point exactly once using these labels:
- covered: every material proposition is expressed with compatible direction,
  scope, and uncertainty;
- partially_covered: at least one material proposition is correctly expressed,
  but another is omitted or conflicting, the scope is incomplete, or the report
  is more or less confident than the reference;
- missing: none of the material propositions is expressed and no incompatible
  proposition is asserted;
- contradicted: one or more material propositions are explicitly incompatible
  and no material proposition is correctly aligned.

Important boundary rule: do not use contradicted solely because the report uses
stronger or weaker certainty language when it otherwise reports the same effect
direction. Use partially_covered with difference_type "certainty" in that case.
For a reference statement whose central finding is itself evidence insufficiency
or uncertainty, a clear directional effect claim is a contradiction and should
use difference_type "evidence_sufficiency_conflict".

Compound-key-point rule: first list the aligned, missing, and conflicting
components. If both aligned_components and conflicting_components are non-empty,
the status must be partially_covered with difference_type "mixed". If aligned
components coexist only with omissions, use partially_covered with difference_type
"scope". Do not mark a compound key point as covered when any material clause is
missing. Do not mark it contradicted when another material clause is correctly
aligned.

For each assessment, identify its principal difference_type:
- none: fully compatible coverage;
- omission: the key point is absent;
- scope: compatible direction but incomplete population, intervention,
  comparator, outcome, time point, or compound finding;
- certainty: compatible direction but materially different confidence;
- factual_conflict: incompatible quantitative or descriptive fact;
- direction_conflict: opposite direction of benefit, harm, or no effect;
- evidence_sufficiency_conflict: a clear effect is claimed where the reference's
  central finding is insufficient, unavailable, or uncertain evidence;
- mixed: more than one material difference applies.

For covered, partially_covered, and contradicted labels, provide one or two short
exact quotes copied from the candidate report. The selected quotes must show the
alignment or conflict described in the justification. For missing, use an empty
list.

Assess the overall conclusion separately:
- aligned: direction, scope, and uncertainty agree with the reference;
- partially_aligned: the broad direction agrees but scope or uncertainty differs,
  or only part of a mixed conclusion is reproduced;
- not_addressed: the reference conclusion is omitted or the report states that
  it cannot determine the answer without asserting the opposite;
- contradicted: the report reaches the opposite direction, states an incompatible
  central fact, or asserts a clear benefit or harm when the reference's overall
  conclusion is that evidence is insufficient or uncertain.

Apply the same compound rule to the overall conclusion: a conclusion containing
both aligned and conflicting material aspects is partially_aligned with
difference_type "mixed". It is contradicted only when no material aspect is
aligned and an explicit incompatibility remains.

Compare expressed uncertainty with the reference using matched,
stronger_than_reference, weaker_than_reference, or not_assessable. Here,
'stronger' means more confident and 'weaker' means more cautious.

Return exactly this JSON structure:
{{
  "question_relevance": {{
    "label": "direct|partial|off_topic",
    "justification": "brief explanation"
  }},
  "key_point_assessments": [
    {{
      "key_point_id": "KP1",
      "status": "covered|partially_covered|missing|contradicted",
      "difference_type": "none|omission|scope|certainty|factual_conflict|direction_conflict|evidence_sufficiency_conflict|mixed",
      "aligned_components": ["brief description of an aligned material proposition"],
      "missing_components": ["brief description of an omitted material proposition"],
      "conflicting_components": ["brief description of an incompatible material proposition"],
      "report_evidence": ["exact short quote from the candidate report"],
      "justification": "brief explanation"
    }}
  ],
  "conclusion_alignment": {{
    "label": "aligned|partially_aligned|not_addressed|contradicted",
    "difference_type": "none|omission|scope|certainty|factual_conflict|direction_conflict|evidence_sufficiency_conflict|mixed",
    "aligned_components": ["brief description of an aligned material aspect"],
    "missing_components": ["brief description of an omitted material aspect"],
    "conflicting_components": ["brief description of an incompatible material aspect"],
    "report_evidence": ["exact short quote from the candidate report"],
    "justification": "brief explanation"
  }},
  "uncertainty_alignment": {{
    "label": "matched|stronger_than_reference|weaker_than_reference|not_assessable",
    "justification": "brief explanation"
  }}
}}
"""


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_selected_pmids(path: Path) -> set[str]:
    payload = load_json(path)
    if isinstance(payload, dict):
        payload = payload.get("reviews", payload.get("items", payload.get("pmids", [])))
    result = set()
    for item in payload:
        value = item.get("pmid") if isinstance(item, dict) else item
        if value is not None:
            result.add(str(value))
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def load_keypoints(path: Path) -> dict[str, dict[str, Any]]:
    references = {}
    for record in read_jsonl(path):
        if record.get("status") == "ok" and record.get("pmid") is not None:
            references[str(record["pmid"])] = record
    return references


def discover_reports(
    model_runs: dict[str, list[Path]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, str]]]:
    reports = {}
    issues = []
    for subject_model, run_directories in model_runs.items():
        for run_directory in run_directories:
            report_directory = run_directory / "reports"
            if not report_directory.is_dir():
                issues.append(
                    {
                        "subject_model": subject_model,
                        "issue": "missing_reports_directory",
                        "path": str(report_directory),
                    }
                )
                continue
            for report_path in report_directory.glob("*_generated_report.md"):
                match = REPORT_PATTERN.match(report_path.name)
                if not match:
                    issues.append(
                        {
                            "subject_model": subject_model,
                            "issue": "unrecognized_report_filename",
                            "path": str(report_path),
                        }
                    )
                    continue
                pmid = match.group("pmid")
                key = (subject_model, pmid)
                candidate = {
                    "subject_model": subject_model,
                    "pmid": pmid,
                    "report_path": report_path,
                    "run_directory": run_directory,
                    "mtime": report_path.stat().st_mtime,
                }
                previous = reports.get(key)
                if previous is None or candidate["mtime"] > previous["mtime"]:
                    reports[key] = candidate
    return reports, issues


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluation_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("subject_model", "")),
        str(record.get("pmid", "")),
        str(record.get("report_sha256", "")),
        str(record.get("reference_sha256", "")),
        str(record.get("prompt_version", "")),
    )


def completed_evaluations(path: Path) -> set[tuple[str, str, str, str, str]]:
    return {
        evaluation_key(record)
        for record in read_jsonl(path)
        if record.get("status") == "ok"
    }


# Typographic variants the judge model routinely "flattens" to plain ASCII when
# copying a quote (curly quotes, non-breaking/en/em dashes, narrow no-break
# spaces, ...). Without this, an exact-quote check fails even on a perfectly
# faithful quote whenever the source report happens to use these characters -
# this was silently causing every evaluation of a report with such
# characters to exhaust all retries and error out.
_TYPOGRAPHIC_EQUIVALENTS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ", "​": "",
}


def normalize_whitespace(text: str) -> str:
    for source, replacement in _TYPOGRAPHIC_EQUIVALENTS.items():
        text = text.replace(source, replacement)
    return " ".join(text.split()).casefold()


def strip_quote_decoration(text: str) -> str:
    """Remove presentation-only markers that are not part of the quoted report text."""
    previous = None
    while text and text != previous:
        previous = text
        text = text.strip()
        text = re.sub(r"^>\s*", "", text)
        text = re.sub(r"^[\"'`*_]+\s*", "", text)
        text = re.sub(r"\s*[\"'`*_]+$", "", text)
        text = re.sub(r"^\.\.\.\s*", "", text)
        text = re.sub(r"\s*\.\.\.$", "", text)
    return text.strip()


def quote_is_from_report(quote: str, report_text: str) -> bool:
    normalized_quote = normalize_whitespace(quote)
    normalized_report = normalize_whitespace(report_text)
    candidates = {
        normalized_quote,
        strip_quote_decoration(normalized_quote),
    }
    return any(candidate and candidate in normalized_report for candidate in candidates)


def require_text(value: Any, field: str, minimum: int = 8) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def require_text_list(
    value: Any,
    field: str,
    *,
    maximum: int | None = None,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{field} must contain at most {maximum} items.")
    return [require_text(item, f"{field} item", minimum=3) for item in value]


def validate_component_structure(
    *,
    label: str,
    difference_type: str,
    aligned: list[str],
    missing: list[str],
    conflicting: list[str],
    field: str,
) -> None:
    if label in {"covered", "aligned"}:
        if not aligned or missing or conflicting:
            raise ValueError(
                f"{field} marked {label} must have aligned components only."
            )
        return
    if label in {"missing", "not_addressed"}:
        if aligned or not missing or conflicting:
            raise ValueError(
                f"{field} marked {label} must have missing components only."
            )
        return
    if label in {"partially_covered", "partially_aligned"}:
        if not aligned:
            raise ValueError(f"{field} marked {label} needs an aligned component.")
        if difference_type == "scope" and (not missing or conflicting):
            raise ValueError(
                f"{field} with scope difference needs a missing component and no conflict."
            )
        if difference_type == "certainty" and (missing or conflicting):
            raise ValueError(
                f"{field} with certainty difference cannot contain missing or conflicting components."
            )
        if difference_type == "mixed" and not (missing or conflicting):
            raise ValueError(
                f"{field} with mixed difference needs a missing or conflicting component."
            )
        return
    if label == "contradicted":
        if aligned or not conflicting:
            raise ValueError(
                f"{field} marked contradicted needs conflicting components and no aligned component."
            )
        return
    raise ValueError(f"Unsupported component label for {field}: {label}")


def alignment_validator(
    reference: dict[str, Any], report_text: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    expected_points = {point["id"]: point for point in reference["key_points"]}

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        evidence_validation_warnings = []
        schema_normalizations = []
        question_relevance = payload.get("question_relevance")
        if not isinstance(question_relevance, dict):
            raise ValueError("question_relevance must be an object.")
        question_relevance_label = str(
            question_relevance.get("label", "")
        ).strip().lower()
        if question_relevance_label not in QUESTION_RELEVANCE_LABELS:
            raise ValueError(
                f"Invalid question_relevance label: {question_relevance_label}"
            )
        normalized_question_relevance = {
            "label": question_relevance_label,
            "justification": require_text(
                question_relevance.get("justification"),
                "question_relevance justification",
            ),
        }

        assessments = payload.get("key_point_assessments")
        if not isinstance(assessments, list):
            raise ValueError("key_point_assessments must be a list.")
        if len(assessments) != len(expected_points):
            raise ValueError(
                "key_point_assessments must contain exactly one item for every "
                f"reference key point ({len(expected_points)} expected)."
            )

        normalized_assessments = []
        seen_ids = set()
        for index, assessment in enumerate(assessments, start=1):
            if not isinstance(assessment, dict):
                raise ValueError(f"Assessment {index} must be an object.")
            key_point_id = str(assessment.get("key_point_id", "")).strip()
            if key_point_id not in expected_points:
                raise ValueError(f"Unknown key_point_id: {key_point_id}")
            if key_point_id in seen_ids:
                raise ValueError(f"Duplicate key_point_id: {key_point_id}")
            seen_ids.add(key_point_id)

            status = str(assessment.get("status", "")).strip().lower()
            if status not in KEYPOINT_STATUSES:
                raise ValueError(f"Invalid status for {key_point_id}: {status}")
            difference_type = str(
                assessment.get("difference_type", "")
            ).strip().lower()
            if difference_type not in DIFFERENCE_TYPES:
                raise ValueError(
                    f"Invalid difference_type for {key_point_id}: {difference_type}"
                )
            aligned_components = require_text_list(
                assessment.get("aligned_components"),
                f"aligned_components for {key_point_id}",
            )
            missing_components = require_text_list(
                assessment.get("missing_components"),
                f"missing_components for {key_point_id}",
            )
            conflicting_components = require_text_list(
                assessment.get("conflicting_components"),
                f"conflicting_components for {key_point_id}",
            )
            conflict_types = {
                "factual_conflict",
                "direction_conflict",
                "evidence_sufficiency_conflict",
            }
            if status == "partially_covered" and difference_type in conflict_types:
                original_pair = f"{status}/{difference_type}"
                if aligned_components and conflicting_components:
                    # A point with both agreement and conflict is partial/mixed by
                    # definition; preserve the conflict while repairing the label.
                    difference_type = "mixed"
                elif conflicting_components and not aligned_components:
                    status = "contradicted"
                if f"{status}/{difference_type}" != original_pair:
                    schema_normalizations.append(
                        {
                            "scope": "key_point",
                            "key_point_id": key_point_id,
                            "from": original_pair,
                            "to": f"{status}/{difference_type}",
                        }
                    )
            expected_difference_types = {
                "covered": {"none"},
                "partially_covered": {"scope", "certainty", "mixed"},
                "missing": {"omission"},
                "contradicted": {
                    "factual_conflict",
                    "direction_conflict",
                    "evidence_sufficiency_conflict",
                    "mixed",
                },
            }
            if difference_type not in expected_difference_types[status]:
                raise ValueError(
                    f"Inconsistent status and difference_type for {key_point_id}: "
                    f"{status}/{difference_type}."
                )
            validate_component_structure(
                label=status,
                difference_type=difference_type,
                aligned=aligned_components,
                missing=missing_components,
                conflicting=conflicting_components,
                field=key_point_id,
            )
            evidence = require_text_list(
                assessment.get("report_evidence"),
                f"report_evidence for {key_point_id}",
                maximum=2,
            )
            if status == "missing" and evidence:
                raise ValueError(f"{key_point_id} is missing, so report_evidence must be empty.")
            evidence_is_verbatim = [
                quote_is_from_report(quote, report_text) for quote in evidence
            ]
            if status != "missing":
                if not evidence:
                    raise ValueError(f"report_evidence for {key_point_id} cannot be empty.")
                if not all(evidence_is_verbatim):
                    evidence_validation_warnings.append(
                        {
                            "scope": "key_point",
                            "key_point_id": key_point_id,
                            "issue": "non_verbatim_report_evidence",
                            "invalid_items": [
                                quote
                                for quote, is_verbatim in zip(
                                    evidence, evidence_is_verbatim
                                )
                                if not is_verbatim
                            ],
                        }
                    )
            justification = require_text(
                assessment.get("justification"), f"justification for {key_point_id}"
            )
            normalized_assessments.append(
                {
                    "key_point_id": key_point_id,
                    "status": status,
                    "difference_type": difference_type,
                    "aligned_components": aligned_components,
                    "missing_components": missing_components,
                    "conflicting_components": conflicting_components,
                    "report_evidence": evidence,
                    "report_evidence_verbatim": (
                        all(evidence_is_verbatim) if evidence else None
                    ),
                    "justification": justification,
                }
            )

        if seen_ids != set(expected_points):
            missing_ids = sorted(set(expected_points) - seen_ids)
            raise ValueError(f"Missing key-point assessments: {missing_ids}")
        normalized_assessments.sort(
            key=lambda item: list(expected_points).index(item["key_point_id"])
        )

        conclusion = payload.get("conclusion_alignment")
        if not isinstance(conclusion, dict):
            raise ValueError("conclusion_alignment must be an object.")
        conclusion_label = str(conclusion.get("label", "")).strip().lower()
        if conclusion_label not in CONCLUSION_LABELS:
            raise ValueError(f"Invalid conclusion_alignment label: {conclusion_label}")
        conclusion_difference = str(
            conclusion.get("difference_type", "")
        ).strip().lower()
        if conclusion_difference not in DIFFERENCE_TYPES:
            raise ValueError(
                "Invalid conclusion_alignment difference_type: "
                f"{conclusion_difference}"
            )
        conclusion_aligned = require_text_list(
            conclusion.get("aligned_components"),
            "conclusion_alignment aligned_components",
        )
        conclusion_missing = require_text_list(
            conclusion.get("missing_components"),
            "conclusion_alignment missing_components",
        )
        conclusion_conflicting = require_text_list(
            conclusion.get("conflicting_components"),
            "conclusion_alignment conflicting_components",
        )
        conflict_types = {
            "factual_conflict",
            "direction_conflict",
            "evidence_sufficiency_conflict",
        }
        if (
            conclusion_label == "partially_aligned"
            and conclusion_difference in conflict_types
        ):
            original_pair = f"{conclusion_label}/{conclusion_difference}"
            if conclusion_aligned and conclusion_conflicting:
                conclusion_difference = "mixed"
            elif conclusion_conflicting and not conclusion_aligned:
                conclusion_label = "contradicted"
            if f"{conclusion_label}/{conclusion_difference}" != original_pair:
                schema_normalizations.append(
                    {
                        "scope": "conclusion",
                        "from": original_pair,
                        "to": f"{conclusion_label}/{conclusion_difference}",
                    }
                )
        expected_conclusion_differences = {
            "aligned": {"none"},
            "partially_aligned": {"scope", "certainty", "mixed"},
            "not_addressed": {"omission"},
            "contradicted": {
                "factual_conflict",
                "direction_conflict",
                "evidence_sufficiency_conflict",
                "mixed",
            },
        }
        if conclusion_difference not in expected_conclusion_differences[conclusion_label]:
            raise ValueError(
                "Inconsistent conclusion label and difference_type: "
                f"{conclusion_label}/{conclusion_difference}."
            )
        validate_component_structure(
            label=conclusion_label,
            difference_type=conclusion_difference,
            aligned=conclusion_aligned,
            missing=conclusion_missing,
            conflicting=conclusion_conflicting,
            field="conclusion_alignment",
        )
        conclusion_evidence = require_text_list(
            conclusion.get("report_evidence"),
            "conclusion_alignment report_evidence",
            maximum=2,
        )
        conclusion_evidence_is_verbatim = [
            quote_is_from_report(quote, report_text) for quote in conclusion_evidence
        ]
        if conclusion_evidence and not all(conclusion_evidence_is_verbatim):
            evidence_validation_warnings.append(
                {
                    "scope": "conclusion",
                    "issue": "non_verbatim_report_evidence",
                    "invalid_items": [
                        quote
                        for quote, is_verbatim in zip(
                            conclusion_evidence, conclusion_evidence_is_verbatim
                        )
                        if not is_verbatim
                    ],
                }
            )
        if conclusion_label != "not_addressed" and not conclusion_evidence:
            raise ValueError(
                "conclusion report_evidence is required unless the label is not_addressed."
            )

        uncertainty = payload.get("uncertainty_alignment")
        if not isinstance(uncertainty, dict):
            raise ValueError("uncertainty_alignment must be an object.")
        uncertainty_label = str(uncertainty.get("label", "")).strip().lower()
        if uncertainty_label not in UNCERTAINTY_LABELS:
            raise ValueError(f"Invalid uncertainty_alignment label: {uncertainty_label}")

        return {
            "question_relevance": normalized_question_relevance,
            "key_point_assessments": normalized_assessments,
            "evidence_validation_warnings": evidence_validation_warnings,
            "schema_normalizations": schema_normalizations,
            "conclusion_alignment": {
                "label": conclusion_label,
                "difference_type": conclusion_difference,
                "aligned_components": conclusion_aligned,
                "missing_components": conclusion_missing,
                "conflicting_components": conclusion_conflicting,
                "report_evidence": conclusion_evidence,
                "report_evidence_verbatim": (
                    all(conclusion_evidence_is_verbatim)
                    if conclusion_evidence
                    else None
                ),
                "justification": require_text(
                    conclusion.get("justification"), "conclusion_alignment justification"
                ),
            },
            "uncertainty_alignment": {
                "label": uncertainty_label,
                "justification": require_text(
                    uncertainty.get("justification"), "uncertainty_alignment justification"
                ),
            },
        }

    return validate


def alignment_score(
    assessments: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    importance: str | None = None,
) -> float | None:
    selected = [
        assessment
        for assessment in assessments
        if importance is None
        or references[assessment["key_point_id"]]["importance"] == importance
    ]
    if not selected:
        return None
    return sum(STATUS_SCORES[item["status"]] for item in selected) / len(selected)


def export_review_files(output_path: Path, review_csv: Path, keypoint_csv: Path) -> None:
    latest = {}
    for record in read_jsonl(output_path):
        if record.get("status") == "ok":
            latest[evaluation_key(record)] = record

    review_rows = []
    keypoint_rows = []
    for record in sorted(latest.values(), key=lambda item: (item["pmid"], item["subject_model"])):
        reference_points = {
            point["id"]: point for point in record["reference_key_points"]
        }
        assessments = record["key_point_assessments"]
        counts = Counter(item["status"] for item in assessments)
        key_points_with_conflict = sum(
            bool(item["conflicting_components"]) for item in assessments
        )
        question_relevance = record["question_relevance"]["label"]
        review_rows.append(
            {
                "pmid": record["pmid"],
                "split": record["split"],
                "subject_model": record["subject_model"],
                "question_relevance": question_relevance,
                "question_relevance_score": QUESTION_RELEVANCE_SCORES[
                    question_relevance
                ],
                "question_relevance_justification": record[
                    "question_relevance"
                ]["justification"],
                "key_point_alignment": alignment_score(assessments, reference_points),
                "core_alignment": alignment_score(assessments, reference_points, "core"),
                "supporting_alignment": alignment_score(
                    assessments, reference_points, "supporting"
                ),
                "covered": counts["covered"],
                "partially_covered": counts["partially_covered"],
                "missing": counts["missing"],
                "contradicted": counts["contradicted"],
                "key_points_with_conflict": key_points_with_conflict,
                "key_point_contradiction_rate": (
                    key_points_with_conflict / len(assessments)
                ),
                "conclusion_alignment": record["conclusion_alignment"]["label"],
                "conclusion_difference_type": record["conclusion_alignment"][
                    "difference_type"
                ],
                "uncertainty_alignment": record["uncertainty_alignment"]["label"],
                "conclusion_aligned_components": " || ".join(
                    record["conclusion_alignment"]["aligned_components"]
                ),
                "conclusion_missing_components": " || ".join(
                    record["conclusion_alignment"]["missing_components"]
                ),
                "conclusion_conflicting_components": " || ".join(
                    record["conclusion_alignment"]["conflicting_components"]
                ),
                "conclusion_report_evidence": " || ".join(
                    record["conclusion_alignment"]["report_evidence"]
                ),
                "conclusion_report_evidence_verbatim": record[
                    "conclusion_alignment"
                ].get("report_evidence_verbatim", True),
                "non_verbatim_evidence_items": len(
                    record.get("evidence_validation_warnings", [])
                ),
                "conclusion_justification": record["conclusion_alignment"]["justification"],
                "report_path": record["report_path"],
                "attempts": record["attempts"],
            }
        )
        for assessment in assessments:
            reference_point = reference_points[assessment["key_point_id"]]
            keypoint_rows.append(
                {
                    "pmid": record["pmid"],
                    "split": record["split"],
                    "subject_model": record["subject_model"],
                    "key_point_id": assessment["key_point_id"],
                    "category": reference_point["category"],
                    "importance": reference_point["importance"],
                    "reference_statement": reference_point["statement"],
                    "status": assessment["status"],
                    "difference_type": assessment["difference_type"],
                    "aligned_components": " || ".join(
                        assessment["aligned_components"]
                    ),
                    "missing_components": " || ".join(
                        assessment["missing_components"]
                    ),
                    "conflicting_components": " || ".join(
                        assessment["conflicting_components"]
                    ),
                    "has_conflict": bool(assessment["conflicting_components"]),
                    "report_evidence": " || ".join(assessment["report_evidence"]),
                    "report_evidence_verbatim": assessment.get(
                        "report_evidence_verbatim", True
                    ),
                    "justification": assessment["justification"],
                    "report_path": record["report_path"],
                }
            )

    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]) if review_rows else [])
        if review_rows:
            writer.writeheader()
            writer.writerows(review_rows)
    with keypoint_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(keypoint_rows[0]) if keypoint_rows else []
        )
        if keypoint_rows:
            writer.writeheader()
            writer.writerows(keypoint_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--dev-pmids", type=Path, default=DEV_PMIDS_PATH)
    parser.add_argument(
        "--reports-root",
        type=Path,
        help=(
            "Alternative root containing qwen3-8b/reports and "
            "fleming-r1-7b/reports, such as result_aware_pilot."
        ),
    )
    parser.add_argument("--split", choices=("development", "test", "all"), default="development")
    parser.add_argument("--paired-only", action="store_true")
    parser.add_argument("--pmid", action="append", help="Evaluate one PMID; may be repeated.")
    parser.add_argument(
        "--limit-pmids",
        type=int,
        help="Process at most this many distinct eligible PMIDs.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--keypoint-csv", type=Path, default=DEFAULT_KEYPOINT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=3400)
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = load_keypoints(args.keypoints)
    development_pmids = load_selected_pmids(args.dev_pmids)
    model_runs = MODEL_RUNS
    if args.reports_root is not None:
        model_runs = {
            "Qwen3-8B": [args.reports_root / "qwen3-8b"],
            "Fleming-R1-7B": [args.reports_root / "fleming-r1-7b"],
        }
    reports, discovery_issues = discover_reports(model_runs)

    available_by_model = Counter(model for model, _ in reports)
    paired_pmids = {
        pmid
        for _, pmid in reports
        if all((model, pmid) in reports for model in model_runs)
    }
    if args.pmid:
        eligible_pmids = {str(pmid) for pmid in args.pmid}
    elif args.split == "development":
        eligible_pmids = set(development_pmids)
    elif args.split == "test":
        eligible_pmids = set(references) - development_pmids
    else:
        eligible_pmids = set(references)
    if args.paired_only:
        eligible_pmids &= paired_pmids

    eligible_pmids &= set(references)
    eligible_pmids &= {pmid for _, pmid in reports}
    selected_pmids = sorted(eligible_pmids, key=lambda value: int(value))
    if args.limit_pmids is not None:
        selected_pmids = selected_pmids[: args.limit_pmids]
    selected_set = set(selected_pmids)

    completed = completed_evaluations(args.output)
    jobs = []
    for (subject_model, pmid), report_record in sorted(
        reports.items(), key=lambda item: (int(item[0][1]), item[0][0])
    ):
        if pmid not in selected_set:
            continue
        report_text = report_record["report_path"].read_text(
            encoding="utf-8", errors="replace"
        )
        reference = references[pmid]
        report_hash = text_sha256(report_text)
        reference_hash = str(reference.get("reference_sha256", ""))
        key_record = {
            "subject_model": subject_model,
            "pmid": pmid,
            "report_sha256": report_hash,
            "reference_sha256": reference_hash,
            "prompt_version": PROMPT_VERSION,
        }
        if not args.refresh and evaluation_key(key_record) in completed:
            continue
        jobs.append((report_record, report_text, reference, key_record))

    print(f"Reports available: {dict(available_by_model)}")
    print(f"Paired PMIDs currently available: {len(paired_pmids)}")
    print(f"Selected PMIDs: {len(selected_pmids)}")
    print(f"Pending report evaluations: {len(jobs)}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Output: {args.output}")
    if discovery_issues:
        print(f"Report discovery issues: {len(discovery_issues)}")
        for issue in discovery_issues:
            print(
                f"  {issue['subject_model']} | {issue['issue']} | "
                f"{issue['path']}"
            )

    if args.dry_run:
        if jobs:
            print("Pending evaluation jobs:")
        for report_record, _, _, _ in jobs:
            print(
                f"  {report_record['pmid']} | {report_record['subject_model']} | "
                f"{report_record['report_path']}"
            )
        return 0
    if not jobs:
        export_review_files(args.output, args.review_csv, args.keypoint_csv)
        print("Nothing to do.")
        return 0

    api_key = os.environ.get("GEMMA_API_KEY", "")
    if not api_key:
        print("GEMMA_API_KEY is not set.", file=sys.stderr)
        return 2
    try:
        from openai import OpenAI
    except ImportError:
        print("The 'openai' package is not installed in the active environment.", file=sys.stderr)
        return 2

    base_url = (
        os.environ.get("GEMMA_API_BASE")
        or os.environ.get("GEMMA_BASE_URL")
        or DEFAULT_BASE_URL
    )
    judge_model = os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)
    client = OpenAI(base_url=base_url, api_key=api_key)

    successes = 0
    failures = 0
    for position, (report_record, report_text, reference, key_record) in enumerate(
        jobs, start=1
    ):
        pmid = report_record["pmid"]
        subject_model = report_record["subject_model"]
        print(f"[{position}/{len(jobs)}] PMID {pmid} ({subject_model})")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(reference, report_text)},
        ]
        try:
            parsed, raw_response, attempts = request_json_completion(
                client=client,
                model=judge_model,
                messages=messages,
                validator=alignment_validator(reference, report_text),
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
            )
            append_jsonl(
                args.output,
                {
                    "status": "ok",
                    "pmid": pmid,
                    "split": (
                        "development" if pmid in development_pmids else "test"
                    ),
                    "subject_model": subject_model,
                    "report_path": str(report_record["report_path"]),
                    "report_sha256": key_record["report_sha256"],
                    "reference_sha256": key_record["reference_sha256"],
                    "prompt_version": PROMPT_VERSION,
                    "judge_model": judge_model,
                    "judge_base_url": base_url,
                    "created_at": utc_now(),
                    "attempts": attempts,
                    "research_query": reference["research_query"],
                    "reference_overall_conclusion": reference["overall_conclusion"],
                    "reference_key_points": reference["key_points"],
                    **parsed,
                    "raw_response": raw_response,
                },
            )
            successes += 1
        except Exception as exc:
            failures += 1
            append_jsonl(
                args.errors,
                {
                    "status": "error",
                    "pmid": pmid,
                    "split": (
                        "development" if pmid in development_pmids else "test"
                    ),
                    "subject_model": subject_model,
                    "report_path": str(report_record["report_path"]),
                    "report_sha256": key_record["report_sha256"],
                    "reference_sha256": key_record["reference_sha256"],
                    "prompt_version": PROMPT_VERSION,
                    "judge_model": judge_model,
                    "created_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"  ERROR: {exc}", file=sys.stderr)

    export_review_files(args.output, args.review_csv, args.keypoint_csv)
    print(f"Finished with {successes} successes and {failures} failures.")
    print(f"Manual review table: {args.review_csv}")
    print(f"Key-point review table: {args.keypoint_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

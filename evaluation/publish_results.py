"""Publish a curated subset of aggregate evaluation artefacts."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def figure_set(source_dir: str, target_dir: str, names: list[str]) -> list[tuple[str, str]]:
    return [
        (f"{source_dir}/{name}.{suffix}", f"{target_dir}/{name}.{suffix}")
        for name in names
        for suffix in ("pdf", "png")
    ]


ARTEFACTS = [
    # Retrieval summaries.
    (
        "outputs/evaluation/retrieval/run_summary_common_reviews.csv",
        "results/retrieval/tables/run_summary_common_reviews.csv",
    ),
    (
        "outputs/evaluation/retrieval/field_set_summary_common_reviews.csv",
        "results/retrieval/tables/field_set_summary_common_reviews.csv",
    ),
    (
        "outputs/evaluation/retrieval/paired_comparisons.csv",
        "results/retrieval/tables/paired_comparisons.csv",
    ),
    (
        "outputs/evaluation/retrieval/depth_contribution.csv",
        "results/retrieval/tables/depth_contribution.csv",
    ),
    (
        "outputs/evaluation/retrieval/field_recovery_patterns.csv",
        "results/retrieval/tables/field_recovery_patterns.csv",
    ),
    *figure_set(
        "outputs/evaluation/retrieval",
        "results/retrieval/figures",
        [
            "breadth_recall_and_cost",
            "coverage_cost_tradeoff",
            "rrf_recall_at_k",
            "field_set_recall_comparison",
            "field_set_rrf_recall_at_k",
            "field_gt_recovery_patterns",
        ],
    ),
    # Context-construction summaries.
    (
        "outputs/evaluation/context/summary_complete_common_reviews.csv",
        "results/context/tables/summary_complete_common_reviews.csv",
    ),
    (
        "outputs/evaluation/context/paired_context_comparisons.csv",
        "results/context/tables/paired_context_comparisons.csv",
    ),
    (
        "outputs/evaluation/context/a3_reranker_ordering_summary.csv",
        "results/context/tables/a3_reranker_ordering_summary.csv",
    ),
    (
        "outputs/evaluation/context/a3_reranker_top30_pair_transitions.csv",
        "results/context/tables/a3_reranker_top30_pair_transitions.csv",
    ),
    (
        "outputs/evaluation/context/a3_chunk_packed_vs_a4_trial_blocks.csv",
        "results/context/tables/a3_chunk_packed_vs_a4_trial_blocks.csv",
    ),
    (
        "outputs/evaluation/context/counterfactual_chunk_repacking_summary.csv",
        "results/context/tables/counterfactual_chunk_repacking_summary.csv",
    ),
    *figure_set(
        "outputs/evaluation/context",
        "results/context/figures",
        [
            "context_evidence_recall_funnel_selected",
            "final_context_component_recall_selected",
            "a3_reranker_ordering_recall_at_k",
            "a3_chunk_packed_vs_a4_trial_blocks",
            "counterfactual_chunk_repacking_recall_complete_runs",
        ],
    ),
    # Deterministic report evaluation.
    (
        "outputs/evaluation/deterministic/deterministic_metrics_macro.csv",
        "results/reports/deterministic/deterministic_metrics_macro.csv",
    ),
    (
        "outputs/evaluation/deterministic/deterministic_metrics_micro.csv",
        "results/reports/deterministic/deterministic_metrics_micro.csv",
    ),
    (
        "outputs/evaluation/deterministic/citation_status_summary.csv",
        "results/reports/deterministic/citation_status_summary.csv",
    ),
    (
        "outputs/evaluation/deterministic/report_completion.csv",
        "results/reports/deterministic/report_completion.csv",
    ),
    (
        "outputs/evaluation/deterministic/figures/report_level_citation_coverage.pdf",
        "results/reports/deterministic/report_level_citation_coverage.pdf",
    ),
    (
        "outputs/evaluation/deterministic/figures/citation_status_composition.pdf",
        "results/reports/deterministic/citation_status_composition.pdf",
    ),
    # Primary semantic alignment.
    *[
        (
            f"outputs/evaluation/report_alignment/analysis/tables/{name}.csv",
            f"results/reports/alignment/tables/{name}.csv",
        )
        for name in [
            "primary_metrics_development",
            "keypoint_status_development",
            "alignment_by_category_development",
            "alignment_by_importance_development",
            "question_relevance_development",
            "conclusion_alignment_development",
            "uncertainty_alignment_development",
            "paired_differences_development",
        ]
    ],
    *figure_set(
        "outputs/evaluation/report_alignment/analysis/figures",
        "results/reports/alignment/figures",
        [
            "primary_metrics_development",
            "keypoint_status_development",
            "alignment_by_category_development",
            "categorical_outcomes_development",
        ],
    ),
    # Post-hoc diagnostic conditions.
    (
        "outputs/evaluation/report_alignment/three_condition_analysis/three_condition_semantic_summary.csv",
        "results/reports/diagnostics/three_condition_semantic_summary.csv",
    ),
    (
        "outputs/evaluation/report_alignment/three_condition_analysis/three_condition_keypoint_status.csv",
        "results/reports/diagnostics/three_condition_keypoint_status.csv",
    ),
    *figure_set(
        "outputs/evaluation/report_alignment/three_condition_analysis",
        "results/reports/diagnostics",
        [
            "report_alignment_diagnostic_conditions",
            "report_alignment_categorical_outcomes",
        ],
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [source for source, _ in ARTEFACTS if not (ROOT / source).is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Run the evaluation notebooks first. Missing:\n{formatted}")

    manifest_rows = []
    for source_relative, target_relative in ARTEFACTS:
        source = ROOT / source_relative
        target = ROOT / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest_rows.append(
            {
                "path": target_relative,
                "source": source_relative,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    manifest_path = ROOT / "results" / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "source", "bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Published {len(ARTEFACTS)} artefacts in {ROOT / 'results'}")


if __name__ == "__main__":
    main()

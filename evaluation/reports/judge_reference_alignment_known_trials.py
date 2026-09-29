"""Run the reference-alignment judge on known-trial reports."""

from __future__ import annotations

import judge_reference_alignment as judge


GROUND_TRUTH_REPORT_ROOT = judge.RUN_ROOT / "known_trial_context_reports"

judge.MODEL_RUNS = {
    "Qwen3-8B": [GROUND_TRUTH_REPORT_ROOT / "qwen3-8b"],
    "Fleming-R1-7B": [GROUND_TRUTH_REPORT_ROOT / "fleming-r1-7b"],
}
judge.DEFAULT_OUTPUT = (
    judge.EVALUATION_OUTPUT / "judge_alignment_known_trials.jsonl"
)
judge.DEFAULT_ERRORS = (
    judge.EVALUATION_OUTPUT / "judge_alignment_known_trials_errors.jsonl"
)
judge.DEFAULT_REVIEW_CSV = (
    judge.EVALUATION_OUTPUT / "judge_alignment_known_trials_review.csv"
)
judge.DEFAULT_KEYPOINT_CSV = (
    judge.EVALUATION_OUTPUT / "judge_alignment_known_trials_keypoints.csv"
)


if __name__ == "__main__":
    raise SystemExit(judge.main())

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
if [[ -z "${GPT_RESEARCHER_ROOT:-}" ]]; then
    GPT_RESEARCHER_ROOT="$REPOSITORY_ROOT"
fi
GPU="${VLLM_GPU:-1}"
PORT="${VLLM_PORT:-8602}"
RUN_ID="${RUN_ID:-final_fleming_trial_context_b4_d2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPOSITORY_ROOT/outputs/$RUN_ID}"
GROUND_TRUTH_PATH="${GROUND_TRUTH_PATH:-$REPOSITORY_ROOT/benchmark/benchmark.json}"
START_INDEX="${FINAL_EVAL_START_INDEX:-1}"
END_INDEX="${FINAL_EVAL_END_INDEX:-}"

cd "$PROJECT_ROOT"

for required_file in \
    run_pipeline.sh \
    evaluation_pipeline.py \
    research_agent.py \
    search_api.py \
    "$GROUND_TRUTH_PATH"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $PROJECT_ROOT/$required_file" >&2
        exit 1
    fi
done

if [[ ! -f "$GPT_RESEARCHER_ROOT/gpt_researcher/skills/researcher.py" ]]; then
    echo "Adapted GPT-Researcher not found at: $GPT_RESEARCHER_ROOT" >&2
    echo "Set GPT_RESEARCHER_ROOT to the directory containing the adapted repository." >&2
    exit 1
fi

if ! grep -q "TRIAL_LEVEL_CONTEXT" "$GPT_RESEARCHER_ROOT/gpt_researcher/skills/researcher.py"; then
    echo "The repository at $GPT_RESEARCHER_ROOT does not appear to contain the CTR adaptations." >&2
    exit 1
fi

export PYTHONPATH="$GPT_RESEARCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"

echo "Starting Fleming final pipeline on GPU $GPU and vLLM port $PORT."
echo "Output directory: $PROJECT_ROOT/$OUTPUT_ROOT"
echo "Adapted GPT-Researcher: $GPT_RESEARCHER_ROOT"
echo "Ground truth: $PROJECT_ROOT/$GROUND_TRUTH_PATH"
echo "Evaluation range: $START_INDEX-${END_INDEX:-all eligible reviews}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Preflight checks passed. DRY_RUN=1, so no model or evaluation was started."
    exit 0
fi

env \
    PYTHONUNBUFFERED=1 \
    GPT_RESEARCHER_ROOT="$GPT_RESEARCHER_ROOT" \
    RUN_ID="$RUN_ID" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    GROUND_TRUTH_PATH="$GROUND_TRUTH_PATH" \
    PROMPT_VARIANT_NAME="permissive_current" \
    FIELD_SET_NAME="retrieval_text_only" \
    MODEL_PATH="IQuestLab/Fleming-R1-7B" \
    SERVED_MODEL_NAME="fleming-r1-7b" \
    VLLM_CUDA_VISIBLE_DEVICES="$GPU" \
    EVAL_CUDA_VISIBLE_DEVICES="" \
    VLLM_PORT="$PORT" \
    VLLM_CONTEXT_WINDOW="${VLLM_CONTEXT_WINDOW:-32768}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}" \
    VLLM_MAX_NUM_SEQS=1 \
    VLLM_REASONING_PARSER="" \
    VLLM_ENABLE_THINKING=false \
    REUSE_VLLM=0 \
    KEEP_VLLM_ALIVE=0 \
    DEEP_RESEARCH_BREADTH=4 \
    DEEP_RESEARCH_DEPTH=2 \
    DEEP_RESEARCH_CONCURRENCY=1 \
    MAX_ITERATIONS=2 \
    PLAN_RESEARCH_CONTEXT_RESULTS=15 \
    PLAN_RESEARCH_CONTEXT_CHARS=2000 \
    SUBQUERY_NCT_DEDUPE=1 \
    SUBQUERY_NCT_DEDUPE_MAX_CANDIDATES=0 \
    LLM_NCT_RERANK=1 \
    LLM_NCT_RERANK_MODE="rank_filter" \
    LLM_NCT_RERANK_INPUT_LIMIT=200 \
    LLM_NCT_RERANK_KEEP_LIMIT=100 \
    LLM_NCT_RERANK_MIN_KEEP=50 \
    LLM_NCT_RERANK_TOTAL_LIMIT=0 \
    LLM_NCT_RERANK_BATCH_SIZE=30 \
    LLM_NCT_RERANK_MAX_TOKENS=3000 \
    LLM_NCT_RERANK_RRF_K=60 \
    TRIAL_LEVEL_CONTEXT=1 \
    TRIAL_LEVEL_CONTEXT_TOP_K=30 \
    TRIAL_LEVEL_CONTEXT_MAX_CHARS_PER_TRIAL=1600 \
    TRIAL_LEVEL_CONTEXT_MIN_LLM_SCORE=-1 \
    TRIAL_LEVEL_CONTEXT_RESULTS_CHARS=450 \
    TRIAL_LEVEL_CONTEXT_OUTCOME_RESULTS_CHARS=500 \
    TRIAL_LEVEL_CONTEXT_SAFETY_RESULTS_CHARS=220 \
    TRIAL_LEVEL_CONTEXT_SUMMARY_CHARS=420 \
    TRIAL_LEVEL_CONTEXT_KEEP_RESULT_EVIDENCE=1 \
    TRIAL_LEVEL_CONTEXT_RESULT_EVIDENCE_K=5 \
    TRIAL_LEVEL_CONTEXT_GLOBAL_DEDUPE=1 \
    TRIAL_LEVEL_CONTEXT_DEDUPE_WITH_LEARNINGS=0 \
    LEGACY_CONTEXT_TRIM=0 \
    FINAL_CONTEXT_MAX_WORDS=9000 \
    FINAL_CONTEXT_LEARNING_WORDS=2000 \
    SMART_TOKEN_LIMIT=4000 \
    SAVE_WEB_CONTEXT_DEBUG=1 \
    SAVE_LLM_NCT_RERANK_DEBUG=1 \
    SAVE_REPORT_PROMPT_AUDIT=1 \
    SKIP_REPORT=0 \
    EVAL_START_INDEX="$START_INDEX" \
    EVAL_END_INDEX="$END_INDEX" \
    bash ./run_pipeline.sh 2>&1 | tee "$OUTPUT_ROOT/launcher.log"

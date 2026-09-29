#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$EXPERIMENT_DIR/.." && pwd)"
cd "$EXPERIMENT_DIR"

run_configuration() {
    local run_id="$1"
    local breadth="$2"
    local depth="$3"
    local field_set="$4"

    echo "Starting $run_id"
    env \
        RUN_ID="$run_id" \
        OUTPUT_ROOT="$REPOSITORY_ROOT/outputs/$run_id" \
        GROUND_TRUTH_PATH="$REPOSITORY_ROOT/data/benchmark.json" \
        GPT_RESEARCHER_ROOT="$REPOSITORY_ROOT" \
        MODEL_PATH="Qwen/Qwen3-8B" \
        SERVED_MODEL_NAME="qwen3-8b" \
        DEEP_RESEARCH_BREADTH="$breadth" \
        DEEP_RESEARCH_DEPTH="$depth" \
        MAX_ITERATIONS=2 \
        FIELD_SET_NAME="$field_set" \
        PLAN_RESEARCH_CONTEXT_RESULTS=15 \
        PLAN_RESEARCH_CONTEXT_CHARS=2000 \
        SUBQUERY_NCT_DEDUPE=0 \
        LLM_NCT_RERANK=0 \
        TRIAL_LEVEL_CONTEXT=0 \
        SKIP_REPORT=1 \
        EVAL_START_INDEX=1 \
        EVAL_END_INDEX=51 \
        REUSE_VLLM=0 \
        KEEP_VLLM_ALIVE=0 \
        bash ./run_pipeline.sh
}

# Breadth/depth comparison using the hybrid searchable representation.
run_configuration rq1_qwen_hybrid_b2_d1_m2 2 1 hybrid
run_configuration rq1_qwen_hybrid_b4_d1_m2 4 1 hybrid
run_configuration rq1_qwen_hybrid_b6_d1_m2 6 1 hybrid
run_configuration rq1_qwen_hybrid_b4_d2_m2 4 2 hybrid
run_configuration rq1_qwen_hybrid_b6_d2_m2 6 2 hybrid

# Searchable-representation comparison under B4D1.
run_configuration rq1_qwen_retrieval_text_b4_d1_m2 4 1 aggregated
run_configuration rq1_qwen_structured_b4_d1_m2 4 1 structured

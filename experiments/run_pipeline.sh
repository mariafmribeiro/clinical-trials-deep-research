#!/usr/bin/env bash
#
# Every variable below is "${VAR:-default}" -- nothing here needs to be
# edited to run a different config. Override anything from the terminal, e.g.:
#
#   DEEP_RESEARCH_BREADTH=4 TRIAL_LEVEL_CONTEXT=1 ./run_pipeline.sh
#
# or export a batch of them (or `source` a .env with `set -a; source .env; set +a`)
# before invoking the script. Defaults below match this script's previous
# hardcoded behavior exactly -- running with nothing overridden reproduces
# the same run as before this file was reorganized.

set -euo pipefail

# =============================================================================
# 1. RUN IDENTITY / OUTPUT LOCATION
# =============================================================================
RUN_ID="${RUN_ID:-eval_qwen3_server}"

# PROMPT_VARIANT_NAME records the prompt variant used for a run.
# FIELD_SET_NAME selects aggregated, structured, or hybrid OpenSearch fields.
PROMPT_VARIANT_NAME="${PROMPT_VARIANT_NAME:-default}"
FIELD_SET_NAME="${FIELD_SET_NAME:-aggregated}"

GROUND_TRUTH_PATH="${GROUND_TRUTH_PATH:-../data/benchmark.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../outputs/eval_qwen3}"
EVAL_START_INDEX="${EVAL_START_INDEX:-}"
EVAL_END_INDEX="${EVAL_END_INDEX:-}"
# 1 = skip the final report-writing LLM call only (retrieval + reranking,
# if SUBQUERY_NCT_DEDUPE/LLM_NCT_RERANK are on, still run as normal inside
# conduct_research()). Safe for retrieval-stage-only experiments, since
# retrieval recall is computed before this step ever runs -- skips the
# single most expensive call per review for zero cost to that metric.
# Leave 0 for anything measuring context-selection or citation.
SKIP_REPORT="${SKIP_REPORT:-0}"

# Two DIFFERENT GPU-visibility scopes on purpose: the vLLM server process
# and the evaluation_pipeline.py/research_agent.py process are deliberately
# allowed to see different CUDA_VISIBLE_DEVICES (the latter defaults to
# none, since it only talks to vLLM over HTTP and shouldn't also grab a GPU).
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-}"

# =============================================================================
# 2. MODEL / LLM SERVING
# =============================================================================
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
LORA_ADAPTER_NAME="${LORA_ADAPTER_NAME:-qwen3-8b-clinical-seed42}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-16}"
if [[ -n "$LORA_ADAPTER_PATH" ]]; then
    REQUEST_MODEL_NAME="${REQUEST_MODEL_NAME:-$LORA_ADAPTER_NAME}"
    DEFAULT_REUSE_VLLM=0
else
    REQUEST_MODEL_NAME="${REQUEST_MODEL_NAME:-$SERVED_MODEL_NAME}"
    DEFAULT_REUSE_VLLM=1
fi
# Previously hardcoded alternative, now just:
#   MODEL_PATH=IQuestLab/Fleming-R1-7B SERVED_MODEL_NAME=fleming-r1-7b ./run_pipeline.sh

# Use "vllm" for a locally served model or "external" for any
# OpenAI-compatible remote API. The external API key is deliberately never
# written to resolved_config.env.
LLM_API_MODE="${LLM_API_MODE:-vllm}"
EXTERNAL_OPENAI_BASE_URL="${EXTERNAL_OPENAI_BASE_URL:-}"
EXTERNAL_OPENAI_API_KEY="${EXTERNAL_OPENAI_API_KEY:-}"
LLM_API_SMOKE_TEST="${LLM_API_SMOKE_TEST:-1}"

VLLM_PORT="${VLLM_PORT:-8001}"                       # stable port so runs can reuse an already-loaded model
VLLM_CONTEXT_WINDOW="${VLLM_CONTEXT_WINDOW:-32768}"   # Qwen3-8B supports up to 40960,
                                                        # Fleming-R1-7B (Qwen2 arch) up to
                                                        # 32768 -- 32768 is the shared safe
                                                        # ceiling for comparing both models
VLLM_CONTEXT_BUFFER="${VLLM_CONTEXT_BUFFER:-512}"
VLLM_TRUNCATE_PROMPT="${VLLM_TRUNCATE_PROMPT:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"          # see earlier discussion: this caps real vLLM-side
                                                       # concurrency regardless of DEEP_RESEARCH_CONCURRENCY
# Qwen3-specific flags. If you switch MODEL_PATH to a non-Qwen3 model (e.g.
# Fleming-R1-7B), set VLLM_REASONING_PARSER="" to omit --reasoning-parser
# entirely -- the original Fleming block never passed it.
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER-qwen3}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}" # true/false -- Qwen3 thinking mode toggle

REUSE_VLLM="${REUSE_VLLM:-$DEFAULT_REUSE_VLLM}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-1}"
STOP_VLLM="${STOP_VLLM:-0}"
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--allow-unsupported-compiler}"

# =============================================================================
# 3. RETRIEVAL (breadth/depth/iterations, query planning)
# =============================================================================
DEEP_RESEARCH_BREADTH="${DEEP_RESEARCH_BREADTH:-6}"
DEEP_RESEARCH_DEPTH="${DEEP_RESEARCH_DEPTH:-1}"
DEEP_RESEARCH_CONCURRENCY="${DEEP_RESEARCH_CONCURRENCY:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
REPORT_TYPE="${REPORT_TYPE:-deep}"
# Caps text fed into the within-branch query-planning LLM call
# (researcher.py plan_research() -> plan_research_outline). Previously not
# set anywhere in this script at all (Python-side default was already 2000);
# exposed here now for visibility/consistency with everything else.
PLAN_RESEARCH_CONTEXT_CHARS="${PLAN_RESEARCH_CONTEXT_CHARS:-2000}"

# --- Search-API-level trial card construction (search_api.py) --------------
# This is a DIFFERENT truncation stage than section 5's TRIAL_LEVEL_CONTEXT_*
# budgets below -- this one caps what search_api.py puts in a card before
# it's even returned from a search call; section 5 caps it again later, at
# context-assembly time, only when TRIAL_LEVEL_CONTEXT=1.
MAX_CONTENT_CHARS="${MAX_CONTENT_CHARS:-12000}"       # whole-card hard cap
TRIAL_CARD_BRIEF_SUMMARY_CHARS="${TRIAL_CARD_BRIEF_SUMMARY_CHARS:-800}"
TRIAL_CARD_DETAILED_DESCRIPTION_CHARS="${TRIAL_CARD_DETAILED_DESCRIPTION_CHARS:-600}"
TRIAL_CARD_ELIGIBILITY_CHARS="${TRIAL_CARD_ELIGIBILITY_CHARS:-1000}"
TRIAL_CARD_OUTCOMES_CHARS="${TRIAL_CARD_OUTCOMES_CHARS:-800}"
TRIAL_CARD_CLINICAL_OUTCOME_RESULTS_CHARS="${TRIAL_CARD_CLINICAL_OUTCOME_RESULTS_CHARS:-3500}"
TRIAL_CARD_CLINICAL_SAFETY_RESULTS_CHARS="${TRIAL_CARD_CLINICAL_SAFETY_RESULTS_CHARS:-1200}"
# NOTE: confirmed via grep -- not read anywhere in the codebase. Dead
# variable, kept only so nothing silently changes if you were relying on it.
TRIAL_CARD_CLINICAL_RESULTS_CHARS="${TRIAL_CARD_CLINICAL_RESULTS_CHARS:-3000}"

# =============================================================================
# 4. EMBEDDING-BASED CONTEXT COMPRESSION (active only when TRIAL_LEVEL_CONTEXT=0)
# =============================================================================
EMBEDDING="${EMBEDDING:-huggingface:sentence-transformers/all-MiniLM-L6-v2}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.35}"
CONTEXT_MAX_RESULTS="${CONTEXT_MAX_RESULTS:-30}"

# =============================================================================
# 5. GLOBAL NCT DEDUPE + LLM RERANKING
# =============================================================================
# SUBQUERY_NCT_DEDUPE is the master switch: LLM_NCT_RERANK and
# TRIAL_LEVEL_CONTEXT (section 6) only have any effect at all when this is 1.
# With SUBQUERY_NCT_DEDUPE=0, every branch silently falls through to the plain
# embedding-compression path in section 4, regardless of what those two say.
SUBQUERY_NCT_DEDUPE="${SUBQUERY_NCT_DEDUPE:-0}"
SUBQUERY_NCT_DEDUPE_MAX_CANDIDATES="${SUBQUERY_NCT_DEDUPE_MAX_CANDIDATES:-0}"

LLM_NCT_RERANK="${LLM_NCT_RERANK:-0}"
LLM_NCT_RERANK_MODE="${LLM_NCT_RERANK_MODE:-boost}"   # boost | rank_filter | filter -- these behave very
                                                        # differently, see earlier discussion (boost never
                                                        # actually drops candidates beyond the input limit;
                                                        # rank_filter/filter do)
LLM_NCT_RERANK_INPUT_LIMIT="${LLM_NCT_RERANK_INPUT_LIMIT:-120}"
LLM_NCT_RERANK_KEEP_LIMIT="${LLM_NCT_RERANK_KEEP_LIMIT:-100}"
LLM_NCT_RERANK_MIN_KEEP="${LLM_NCT_RERANK_MIN_KEEP:-50}"
LLM_NCT_RERANK_TOTAL_LIMIT="${LLM_NCT_RERANK_TOTAL_LIMIT:-0}"
LLM_NCT_RERANK_BATCH_SIZE="${LLM_NCT_RERANK_BATCH_SIZE:-40}"
LLM_NCT_RERANK_MAX_TOKENS="${LLM_NCT_RERANK_MAX_TOKENS:-3000}"
LLM_NCT_RERANK_RRF_K="${LLM_NCT_RERANK_RRF_K:-60}"

# =============================================================================
# 6. TRIAL-LEVEL CONTEXT / TRIAL CARDS (needs SUBQUERY_NCT_DEDUPE=1 to matter)
# =============================================================================
TRIAL_LEVEL_CONTEXT="${TRIAL_LEVEL_CONTEXT:-0}"       # 1 = trial cards (this section); 0 = embedding
                                                        # compression (section 4) instead
TRIAL_LEVEL_CONTEXT_TOP_K="${TRIAL_LEVEL_CONTEXT_TOP_K:-30}"
TRIAL_LEVEL_CONTEXT_MAX_CHARS_PER_TRIAL="${TRIAL_LEVEL_CONTEXT_MAX_CHARS_PER_TRIAL:-1200}"
TRIAL_LEVEL_CONTEXT_MIN_LLM_SCORE="${TRIAL_LEVEL_CONTEXT_MIN_LLM_SCORE:--1}"
TRIAL_LEVEL_CONTEXT_RESULTS_CHARS="${TRIAL_LEVEL_CONTEXT_RESULTS_CHARS:-450}"
TRIAL_LEVEL_CONTEXT_OUTCOME_RESULTS_CHARS="${TRIAL_LEVEL_CONTEXT_OUTCOME_RESULTS_CHARS:-500}"
TRIAL_LEVEL_CONTEXT_SAFETY_RESULTS_CHARS="${TRIAL_LEVEL_CONTEXT_SAFETY_RESULTS_CHARS:-220}"
TRIAL_LEVEL_CONTEXT_SUMMARY_CHARS="${TRIAL_LEVEL_CONTEXT_SUMMARY_CHARS:-420}"
TRIAL_LEVEL_CONTEXT_KEEP_RESULT_EVIDENCE="${TRIAL_LEVEL_CONTEXT_KEEP_RESULT_EVIDENCE:-1}"
TRIAL_LEVEL_CONTEXT_RESULT_EVIDENCE_K="${TRIAL_LEVEL_CONTEXT_RESULT_EVIDENCE_K:-5}"
# NOTE: named similarly to SUBQUERY_NCT_DEDUPE above but a DIFFERENT scope --
# that one dedupes within one breadth branch; these two dedupe trial-card
# text blocks ACROSS breadth branches, at the top of deep_research()'s
# recursion (deep_research.py, not researcher.py). Don't confuse them.
TRIAL_LEVEL_CONTEXT_GLOBAL_DEDUPE="${TRIAL_LEVEL_CONTEXT_GLOBAL_DEDUPE:-0}"
TRIAL_LEVEL_CONTEXT_DEDUPE_WITH_LEARNINGS="${TRIAL_LEVEL_CONTEXT_DEDUPE_WITH_LEARNINGS:-0}"
TRIAL_LEVEL_CONTEXT_GLOBAL_FALLBACK_K="${TRIAL_LEVEL_CONTEXT_GLOBAL_FALLBACK_K:-10}"

# =============================================================================
# 7. FINAL CONTEXT ASSEMBLY / TOKEN BUDGETS
# =============================================================================
FINAL_CONTEXT_MAX_WORDS="${FINAL_CONTEXT_MAX_WORDS:-7500}"
FINAL_CONTEXT_LEARNING_WORDS="${FINAL_CONTEXT_LEARNING_WORDS:-2000}"

SMART_TOKEN_LIMIT="${SMART_TOKEN_LIMIT:-2048}"
STRATEGIC_TOKEN_LIMIT="${STRATEGIC_TOKEN_LIMIT:-2048}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"
FAST_TOKEN_LIMIT="${FAST_TOKEN_LIMIT:-1024}"
SUMMARY_TOKEN_LIMIT="${SUMMARY_TOKEN_LIMIT:-1024}"
TOTAL_WORDS="${TOTAL_WORDS:-500}"
LLM_OUTPUT_TOKEN_MARGIN="${LLM_OUTPUT_TOKEN_MARGIN:-128}"

SMART_LLM_CONTEXT_WINDOW="${SMART_LLM_CONTEXT_WINDOW:-12000}"
FAST_LLM_CONTEXT_WINDOW="${FAST_LLM_CONTEXT_WINDOW:-12000}"
STRATEGIC_LLM_CONTEXT_WINDOW="${STRATEGIC_LLM_CONTEXT_WINDOW:-12000}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-12000}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-12000}"
TOKEN_BUDGET="${TOKEN_BUDGET:-12000}"


# =============================================================================
# 8. DEBUG / AUDIT (leave these on -- needed for trace_trial_fate.py etc.)
# =============================================================================
SAVE_WEB_CONTEXT_DEBUG="${SAVE_WEB_CONTEXT_DEBUG:-1}"
SAVE_LLM_NCT_RERANK_DEBUG="${SAVE_LLM_NCT_RERANK_DEBUG:-1}"
SAVE_REPORT_PROMPT_AUDIT="${SAVE_REPORT_PROMPT_AUDIT:-1}"

# =============================================================================
# --- everything below this line is orchestration, not experiment config ----
# =============================================================================

mkdir -p logs

cleanup() {
    local exit_code=$?
    set +e

    echo "Research complete. Shutting down background processes..."

    if [[ "${VLLM_OWNED_BY_THIS_RUN:-0}" == "1" ]] && [[ -n "${V_PID:-}" ]] && kill -0 "$V_PID" 2>/dev/null; then
        if [[ "${KEEP_VLLM_ALIVE:-1}" == "1" ]]; then
            echo "KEEP_VLLM_ALIVE=1, leaving vLLM running on ${VLLM_URL:-unknown} with PID $V_PID."
            disown "$V_PID" 2>/dev/null || true
        else
            echo "Stopping vLLM PID $V_PID..."
            kill "$V_PID" 2>/dev/null || true
            wait "$V_PID" 2>/dev/null || true
            rm -f "${VLLM_PID_FILE:-}" 2>/dev/null || true
        fi
    fi

    if [[ -n "${SEARCH_PID:-}" ]] && kill -0 "$SEARCH_PID" 2>/dev/null; then
        kill "$SEARCH_PID" 2>/dev/null || true
        wait "$SEARCH_PID" 2>/dev/null || true
    fi

    echo "Job Finished with exit code $exit_code."
}
trap cleanup EXIT

RUNNER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer an already-active environment or a uv-created .venv. Conda remains
# available as a fallback for the original execution environment.
if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "$VIRTUAL_ENV/bin/python" ]]; then
    echo "Using active Python environment: $VIRTUAL_ENV"
elif [[ -n "${PYTHON_VENV_PATH:-}" ]] && [[ -f "$PYTHON_VENV_PATH/bin/activate" ]]; then
    source "$PYTHON_VENV_PATH/bin/activate"
    echo "Activated Python environment: $PYTHON_VENV_PATH"
elif [[ -n "${GPT_RESEARCHER_ROOT:-}" ]] && [[ -f "$GPT_RESEARCHER_ROOT/.venv/bin/activate" ]]; then
    source "$GPT_RESEARCHER_ROOT/.venv/bin/activate"
    echo "Activated uv environment: $GPT_RESEARCHER_ROOT/.venv"
elif [[ -f "$RUNNER_ROOT/.venv/bin/activate" ]]; then
    source "$RUNNER_ROOT/.venv/bin/activate"
    echo "Activated uv environment: $RUNNER_ROOT/.venv"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME:-research_env}"
    echo "Activated Conda environment: ${CONDA_ENV_NAME:-research_env}"
elif command -v python >/dev/null 2>&1; then
    echo "Using Python already available on PATH."
else
    echo "No usable Python environment was found." >&2
    echo "Activate the uv environment or set PYTHON_VENV_PATH to its directory." >&2
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "The selected environment must provide python." >&2
    exit 1
fi

case "$LLM_API_MODE" in
    vllm|external) ;;
    *)
        echo "LLM_API_MODE must be either 'vllm' or 'external'." >&2
        exit 1
        ;;
esac

if [[ "$LLM_API_MODE" == "vllm" ]] && ! command -v vllm >/dev/null 2>&1; then
    echo "The selected environment must provide vllm when LLM_API_MODE=vllm." >&2
    exit 1
fi

if [[ "$LLM_API_MODE" == "external" ]]; then
    if [[ -z "$EXTERNAL_OPENAI_BASE_URL" ]]; then
        echo "EXTERNAL_OPENAI_BASE_URL is required when LLM_API_MODE=external." >&2
        exit 1
    fi
    if [[ -z "$EXTERNAL_OPENAI_API_KEY" ]]; then
        echo "EXTERNAL_OPENAI_API_KEY is required when LLM_API_MODE=external." >&2
        exit 1
    fi
fi

# CUDA setup and the compiled vLLM import are needed only when this runner is
# responsible for serving a local model.
if [[ "$LLM_API_MODE" == "vllm" ]]; then
    PYTHON_CUDA_LIBRARY_PATHS="$(python - <<'PY'
import site
from pathlib import Path

roots = [Path(path) for path in site.getsitepackages()]
user_site = site.getusersitepackages()
if user_site:
    roots.append(Path(user_site))

library_dirs = set()
for root in roots:
    if not root.is_dir():
        continue
    library_dirs.update(path for path in root.glob("nvidia/*/lib") if path.is_dir())
    torch_lib = root / "torch" / "lib"
    if torch_lib.is_dir():
        library_dirs.add(torch_lib)

print(":".join(str(path) for path in sorted(library_dirs)))
PY
)"

    if [[ -n "$PYTHON_CUDA_LIBRARY_PATHS" ]]; then
        export LD_LIBRARY_PATH="$PYTHON_CUDA_LIBRARY_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    if [[ -d /usr/local/cuda/lib64 ]]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    if [[ -z "${CUDA_HOME:-}" ]] && [[ -d /usr/local/cuda ]]; then
        export CUDA_HOME=/usr/local/cuda
    fi
    export CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES"

    if ! python -c 'import vllm._C' >/dev/null 2>&1; then
        echo "The vLLM CUDA extension could not be loaded in the selected environment." >&2
        echo "Check the CUDA runtime packages and LD_LIBRARY_PATH before retrying." >&2
        python -c 'import vllm._C'
        exit 1
    fi
fi

S_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
SEARCH_API_URL_LOCAL="http://127.0.0.1:$S_PORT/search"

if [[ "$LLM_API_MODE" == "external" ]]; then
    V_PORT=""
    VLLM_URL=""
    LLM_API_URL="${EXTERNAL_OPENAI_BASE_URL%/}"
    LLM_API_KEY_VALUE="$EXTERNAL_OPENAI_API_KEY"
else
    V_PORT="$VLLM_PORT"
    VLLM_URL="http://127.0.0.1:$V_PORT/v1"
    LLM_API_URL="$VLLM_URL"
    LLM_API_KEY_VALUE="dummy_key"
fi

RUN_NODE="${SLURM_NODELIST:-$(hostname)}"

VLLM_LOG="logs/vllm_${RUN_ID}.log"
SEARCH_LOG="logs/search_api_${RUN_ID}.log"
VLLM_PID_FILE="${VLLM_PID_FILE:-logs/vllm_${REQUEST_MODEL_NAME}_${V_PORT:-external}.pid}"

stop_saved_vllm() {
    local pid=""
    if [[ -f "$VLLM_PID_FILE" ]]; then
        pid="$(cat "$VLLM_PID_FILE" 2>/dev/null || true)"
    fi

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping saved vLLM PID $pid from $VLLM_PID_FILE..."
        kill "$pid" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "vLLM did not stop gracefully; killing PID $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$VLLM_PID_FILE"
        echo "vLLM stopped."
    else
        rm -f "$VLLM_PID_FILE"
        if curl -fsS -H "Authorization: Bearer dummy_key" "$VLLM_URL/models" > /dev/null 2>&1; then
            echo "vLLM is responding at $VLLM_URL, but no valid PID file was found."
            echo "Stop it manually or set VLLM_PID_FILE to the right PID file."
        else
            echo "No saved vLLM process found for $VLLM_URL."
        fi
    fi
}

if [[ "$STOP_VLLM" == "1" ]]; then
    if [[ "$LLM_API_MODE" == "external" ]]; then
        echo "No local vLLM process is used when LLM_API_MODE=external."
        exit 0
    fi
    stop_saved_vllm
    exit 0
fi

echo "------------------------------------------------"
echo "Run ID: $RUN_ID"
echo "Running on Node: $RUN_NODE"
echo "Model path: $MODEL_PATH"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Requested model name: $REQUEST_MODEL_NAME"
echo "LoRA adapter: ${LORA_ADAPTER_PATH:-none}"
echo "LLM API mode: $LLM_API_MODE"
echo "LLM API URL: $LLM_API_URL"
echo "Search URL: $SEARCH_API_URL_LOCAL"
echo "------------------------------------------------"

# Search config (derived from S_PORT above -- not meant to be overridden directly)
export RETRIEVER="custom"
export SEARCH_RETRIEVER="custom"
export SEARCH_API_URL="$SEARCH_API_URL_LOCAL"
export SEARCH_URL="$SEARCH_API_URL_LOCAL"
export RETRIEVER_ENDPOINT="$SEARCH_API_URL_LOCAL"
export CUSTOM_RETRIEVER_URL="$SEARCH_API_URL_LOCAL"

# LLM config. Both local vLLM and the Gemma endpoint expose an
# OpenAI-compatible API, so GPT-Researcher uses the same provider path.
export OPENAI_API_BASE="$LLM_API_URL"
export OPENAI_BASE_URL="$LLM_API_URL"
export OPENAI_API_KEY="$LLM_API_KEY_VALUE"
export FAST_LLM="openai:$REQUEST_MODEL_NAME"
export SMART_LLM="openai:$REQUEST_MODEL_NAME"
export STRATEGIC_LLM="openai:$REQUEST_MODEL_NAME"
export FAST_LLM_MODEL="$REQUEST_MODEL_NAME"
export SMART_LLM_MODEL="$REQUEST_MODEL_NAME"
export STRATEGIC_LLM_MODEL="$REQUEST_MODEL_NAME"
export OPENAI_MODEL="$REQUEST_MODEL_NAME"
export OPENAI_API_MODEL="$REQUEST_MODEL_NAME"

# Export everything from sections 1-9 above so every subprocess (vLLM launch
# excluded, which reads the shell vars directly) inherits it.
export MODEL_PATH SERVED_MODEL_NAME REQUEST_MODEL_NAME
export PROMPT_VARIANT_NAME FIELD_SET_NAME
export LORA_ADAPTER_PATH LORA_ADAPTER_NAME VLLM_MAX_LORA_RANK
export MAX_CONTENT_CHARS PLAN_RESEARCH_CONTEXT_CHARS
export TRIAL_CARD_BRIEF_SUMMARY_CHARS TRIAL_CARD_DETAILED_DESCRIPTION_CHARS
export TRIAL_CARD_ELIGIBILITY_CHARS TRIAL_CARD_OUTCOMES_CHARS
export TRIAL_CARD_CLINICAL_OUTCOME_RESULTS_CHARS TRIAL_CARD_CLINICAL_SAFETY_RESULTS_CHARS
export TRIAL_CARD_CLINICAL_RESULTS_CHARS
export EMBEDDING EMBEDDING_DEVICE SIMILARITY_THRESHOLD CONTEXT_MAX_RESULTS
export REPORT_TYPE MAX_ITERATIONS DEEP_RESEARCH_CONCURRENCY DEEP_RESEARCH_BREADTH DEEP_RESEARCH_DEPTH
export SMART_TOKEN_LIMIT STRATEGIC_TOKEN_LIMIT MAX_TOKENS MAX_COMPLETION_TOKENS
export FAST_TOKEN_LIMIT SUMMARY_TOKEN_LIMIT TOTAL_WORDS LLM_OUTPUT_TOKEN_MARGIN
export SMART_LLM_CONTEXT_WINDOW FAST_LLM_CONTEXT_WINDOW STRATEGIC_LLM_CONTEXT_WINDOW
export MAX_CONTEXT_SIZE CONTEXT_WINDOW TOKEN_BUDGET
export VLLM_TRUNCATE_PROMPT VLLM_CONTEXT_WINDOW VLLM_CONTEXT_BUFFER
export FINAL_CONTEXT_MAX_WORDS FINAL_CONTEXT_LEARNING_WORDS SAVE_REPORT_PROMPT_AUDIT
export SUBQUERY_NCT_DEDUPE SUBQUERY_NCT_DEDUPE_MAX_CANDIDATES
export LLM_NCT_RERANK LLM_NCT_RERANK_MODE LLM_NCT_RERANK_INPUT_LIMIT LLM_NCT_RERANK_KEEP_LIMIT
export LLM_NCT_RERANK_MIN_KEEP LLM_NCT_RERANK_TOTAL_LIMIT LLM_NCT_RERANK_BATCH_SIZE
export LLM_NCT_RERANK_MAX_TOKENS LLM_NCT_RERANK_RRF_K
export TRIAL_LEVEL_CONTEXT TRIAL_LEVEL_CONTEXT_TOP_K TRIAL_LEVEL_CONTEXT_MAX_CHARS_PER_TRIAL
export TRIAL_LEVEL_CONTEXT_MIN_LLM_SCORE TRIAL_LEVEL_CONTEXT_RESULTS_CHARS
export TRIAL_LEVEL_CONTEXT_OUTCOME_RESULTS_CHARS TRIAL_LEVEL_CONTEXT_SAFETY_RESULTS_CHARS
export TRIAL_LEVEL_CONTEXT_SUMMARY_CHARS TRIAL_LEVEL_CONTEXT_KEEP_RESULT_EVIDENCE
export TRIAL_LEVEL_CONTEXT_RESULT_EVIDENCE_K TRIAL_LEVEL_CONTEXT_GLOBAL_DEDUPE
export TRIAL_LEVEL_CONTEXT_DEDUPE_WITH_LEARNINGS TRIAL_LEVEL_CONTEXT_GLOBAL_FALLBACK_K
export SAVE_WEB_CONTEXT_DEBUG SAVE_LLM_NCT_RERANK_DEBUG

export VLLM_TARGET_DEVICE VLLM_USE_V1 NVCC_APPEND_FLAGS
export LLM_API_MODE EXTERNAL_OPENAI_BASE_URL LLM_API_SMOKE_TEST

V_PID=""
VLLM_OWNED_BY_THIS_RUN=0

if [[ "$LLM_API_MODE" == "external" ]]; then
    echo "Using external OpenAI-compatible LLM API at $LLM_API_URL."

    if [[ "$LLM_API_SMOKE_TEST" == "1" ]]; then
        echo "Testing external LLM chat completion..."
        curl -fsS "$LLM_API_URL/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $LLM_API_KEY_VALUE" \
            -d "{
                \"model\": \"$REQUEST_MODEL_NAME\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Say OK only.\"}],
                \"max_tokens\": 16,
                \"temperature\": 0
            }" > /dev/null || {
                echo "External LLM chat completion test failed for $LLM_API_URL." >&2
                exit 1
            }
        echo "External LLM API is UP!"
    fi
else
    if [[ "$REUSE_VLLM" == "1" ]] && curl -fsS -H "Authorization: Bearer dummy_key" "$VLLM_URL/models" > /dev/null 2>&1; then
        echo "Reusing existing vLLM at $VLLM_URL"
    else
        echo "Starting vLLM on $VLLM_URL..."
        VLLM_SERVE_ARGS=(
            "$MODEL_PATH"
            --served-model-name "$SERVED_MODEL_NAME"
            --port "$V_PORT"
            --host 127.0.0.1
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
            --max-model-len "$VLLM_CONTEXT_WINDOW"
            --max-num-seqs "$VLLM_MAX_NUM_SEQS"
            --enforce-eager
            --trust-remote-code
        )
        if [[ -n "$VLLM_REASONING_PARSER" ]]; then
            VLLM_SERVE_ARGS+=(--reasoning-parser "$VLLM_REASONING_PARSER")
            VLLM_SERVE_ARGS+=(--default-chat-template-kwargs "{\"enable_thinking\": $VLLM_ENABLE_THINKING}")
        fi
        if [[ -n "$LORA_ADAPTER_PATH" ]]; then
            if [[ ! -f "$LORA_ADAPTER_PATH/adapter_config.json" ]]; then
                echo "LoRA adapter_config.json not found at: $LORA_ADAPTER_PATH" >&2
                exit 1
            fi
            VLLM_SERVE_ARGS+=(--enable-lora)
            VLLM_SERVE_ARGS+=(--lora-modules "$LORA_ADAPTER_NAME=$LORA_ADAPTER_PATH")
            VLLM_SERVE_ARGS+=(--max-lora-rank "$VLLM_MAX_LORA_RANK")
        fi

        vllm serve "${VLLM_SERVE_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
        V_PID=$!
        VLLM_OWNED_BY_THIS_RUN=1
        echo "$V_PID" > "$VLLM_PID_FILE"
        echo "Started vLLM PID $V_PID; PID file: $VLLM_PID_FILE"
    fi

    echo "Waiting for vLLM..."
    until curl -fsS -H "Authorization: Bearer dummy_key" "$VLLM_URL/models" > /dev/null; do
        sleep 5
        echo "Still waiting for vLLM..."
        if [[ "$VLLM_OWNED_BY_THIS_RUN" == "1" ]] && ! kill -0 "$V_PID" 2>/dev/null; then
            echo "vLLM exited early. See $VLLM_LOG"
            tail -80 "$VLLM_LOG" || true
            rm -f "$VLLM_PID_FILE"
            exit 1
        fi
    done
    echo "vLLM is UP!"

    echo "Testing vLLM chat completion..."
    curl -fsS "$VLLM_URL/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer dummy_key" \
        -d "{
            \"model\": \"$REQUEST_MODEL_NAME\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say OK only.\"}],
            \"max_tokens\": 100,
            \"temperature\": 0
        }" || {
            echo "vLLM chat completion test failed. See $VLLM_LOG"
            tail -80 "$VLLM_LOG" || true
            exit 1
        }
fi

echo ""
echo "RESOLVED CONFIG FOR THIS RUN:"
echo "  MODEL_PATH=$MODEL_PATH  SERVED_MODEL_NAME=$SERVED_MODEL_NAME  REQUEST_MODEL_NAME=$REQUEST_MODEL_NAME"
echo "  LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-none}  VLLM_MAX_LORA_RANK=$VLLM_MAX_LORA_RANK  LLM_API_MODE=$LLM_API_MODE"
echo "  DEEP_RESEARCH_BREADTH=$DEEP_RESEARCH_BREADTH  DEEP_RESEARCH_DEPTH=$DEEP_RESEARCH_DEPTH  DEEP_RESEARCH_CONCURRENCY=$DEEP_RESEARCH_CONCURRENCY"
echo "  SUBQUERY_NCT_DEDUPE=$SUBQUERY_NCT_DEDUPE  LLM_NCT_RERANK=$LLM_NCT_RERANK  LLM_NCT_RERANK_MODE=$LLM_NCT_RERANK_MODE"
echo "  TRIAL_LEVEL_CONTEXT=$TRIAL_LEVEL_CONTEXT  TRIAL_LEVEL_CONTEXT_TOP_K=$TRIAL_LEVEL_CONTEXT_TOP_K"
echo "  SIMILARITY_THRESHOLD=$SIMILARITY_THRESHOLD  CONTEXT_MAX_RESULTS=$CONTEXT_MAX_RESULTS"
echo "  FINAL_CONTEXT_MAX_WORDS=$FINAL_CONTEXT_MAX_WORDS  FINAL_CONTEXT_LEARNING_WORDS=$FINAL_CONTEXT_LEARNING_WORDS"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT  GROUND_TRUTH_PATH=$GROUND_TRUTH_PATH"
echo "  PROMPT_VARIANT_NAME=$PROMPT_VARIANT_NAME  FIELD_SET_NAME=$FIELD_SET_NAME"

# Write the fully-resolved config (every value that actually took effect,
# whether from an override or a default) into the run's own output folder --
# so "what did I run to produce this" is always answerable by looking at the
# results themselves, not by remembering or re-finding a terminal command.
mkdir -p "$OUTPUT_ROOT"
RESOLVED_CONFIG_PATH="$OUTPUT_ROOT/resolved_config.env"
{
    echo "# Resolved config for RUN_ID=$RUN_ID"
    echo "# Generated: $(date -Iseconds)"
    echo "# Node: $RUN_NODE"
    echo "# To reproduce this exact config:"
    echo "#   set -a; source resolved_config.env; set +a; bash ./run_pipeline.sh"
    echo

    echo "# --- 1. Run identity / output location ---"
    echo "RUN_ID=$RUN_ID"
    echo "PROMPT_VARIANT_NAME=$PROMPT_VARIANT_NAME"
    echo "FIELD_SET_NAME=$FIELD_SET_NAME"
    echo "GROUND_TRUTH_PATH=$GROUND_TRUTH_PATH"
    echo "OUTPUT_ROOT=$OUTPUT_ROOT"
    echo "EVAL_START_INDEX=$EVAL_START_INDEX"
    echo "EVAL_END_INDEX=$EVAL_END_INDEX"
    echo "SKIP_REPORT=$SKIP_REPORT"
    echo "VLLM_CUDA_VISIBLE_DEVICES=$VLLM_CUDA_VISIBLE_DEVICES"
    echo "EVAL_CUDA_VISIBLE_DEVICES=$EVAL_CUDA_VISIBLE_DEVICES"
    echo

    echo "# --- 2. Model / LLM serving ---"
    echo "MODEL_PATH=$MODEL_PATH"
    echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
    echo "REQUEST_MODEL_NAME=$REQUEST_MODEL_NAME"
    echo "LORA_ADAPTER_PATH=$LORA_ADAPTER_PATH"
    echo "LORA_ADAPTER_NAME=$LORA_ADAPTER_NAME"
    echo "VLLM_MAX_LORA_RANK=$VLLM_MAX_LORA_RANK"
    echo "LLM_API_MODE=$LLM_API_MODE"
    echo "EXTERNAL_OPENAI_BASE_URL=$EXTERNAL_OPENAI_BASE_URL"
    echo "# EXTERNAL_OPENAI_API_KEY is intentionally omitted; provide it at launch time."
    echo "LLM_API_SMOKE_TEST=$LLM_API_SMOKE_TEST"
    echo "VLLM_PORT=$VLLM_PORT"
    echo "VLLM_CONTEXT_WINDOW=$VLLM_CONTEXT_WINDOW"
    echo "VLLM_CONTEXT_BUFFER=$VLLM_CONTEXT_BUFFER"
    echo "VLLM_TRUNCATE_PROMPT=$VLLM_TRUNCATE_PROMPT"
    echo "VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION"
    echo "VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS"
    echo "VLLM_REASONING_PARSER=$VLLM_REASONING_PARSER"
    echo "VLLM_ENABLE_THINKING=$VLLM_ENABLE_THINKING"
    echo "REUSE_VLLM=$REUSE_VLLM"
    echo "KEEP_VLLM_ALIVE=$KEEP_VLLM_ALIVE"
    echo

    echo "# --- 3. Retrieval ---"
    echo "DEEP_RESEARCH_BREADTH=$DEEP_RESEARCH_BREADTH"
    echo "DEEP_RESEARCH_DEPTH=$DEEP_RESEARCH_DEPTH"
    echo "DEEP_RESEARCH_CONCURRENCY=$DEEP_RESEARCH_CONCURRENCY"
    echo "MAX_ITERATIONS=$MAX_ITERATIONS"
    echo "REPORT_TYPE=$REPORT_TYPE"
    echo "PLAN_RESEARCH_CONTEXT_CHARS=$PLAN_RESEARCH_CONTEXT_CHARS"
    echo "MAX_CONTENT_CHARS=$MAX_CONTENT_CHARS"
    echo "TRIAL_CARD_BRIEF_SUMMARY_CHARS=$TRIAL_CARD_BRIEF_SUMMARY_CHARS"
    echo "TRIAL_CARD_DETAILED_DESCRIPTION_CHARS=$TRIAL_CARD_DETAILED_DESCRIPTION_CHARS"
    echo "TRIAL_CARD_ELIGIBILITY_CHARS=$TRIAL_CARD_ELIGIBILITY_CHARS"
    echo "TRIAL_CARD_OUTCOMES_CHARS=$TRIAL_CARD_OUTCOMES_CHARS"
    echo "TRIAL_CARD_CLINICAL_OUTCOME_RESULTS_CHARS=$TRIAL_CARD_CLINICAL_OUTCOME_RESULTS_CHARS"
    echo "TRIAL_CARD_CLINICAL_SAFETY_RESULTS_CHARS=$TRIAL_CARD_CLINICAL_SAFETY_RESULTS_CHARS"
    echo

    echo "# --- 4. Embedding-based context compression (active only when TRIAL_LEVEL_CONTEXT=0) ---"
    echo "EMBEDDING=$EMBEDDING"
    echo "EMBEDDING_DEVICE=$EMBEDDING_DEVICE"
    echo "SIMILARITY_THRESHOLD=$SIMILARITY_THRESHOLD"
    echo "CONTEXT_MAX_RESULTS=$CONTEXT_MAX_RESULTS"
    echo

    echo "# --- 5. Subquery NCT dedupe + LLM reranking ---"
    echo "SUBQUERY_NCT_DEDUPE=$SUBQUERY_NCT_DEDUPE"
    echo "SUBQUERY_NCT_DEDUPE_MAX_CANDIDATES=$SUBQUERY_NCT_DEDUPE_MAX_CANDIDATES"
    echo "LLM_NCT_RERANK=$LLM_NCT_RERANK"
    echo "LLM_NCT_RERANK_MODE=$LLM_NCT_RERANK_MODE"
    echo "LLM_NCT_RERANK_INPUT_LIMIT=$LLM_NCT_RERANK_INPUT_LIMIT"
    echo "LLM_NCT_RERANK_KEEP_LIMIT=$LLM_NCT_RERANK_KEEP_LIMIT"
    echo "LLM_NCT_RERANK_MIN_KEEP=$LLM_NCT_RERANK_MIN_KEEP"
    echo "LLM_NCT_RERANK_TOTAL_LIMIT=$LLM_NCT_RERANK_TOTAL_LIMIT"
    echo "LLM_NCT_RERANK_BATCH_SIZE=$LLM_NCT_RERANK_BATCH_SIZE"
    echo "LLM_NCT_RERANK_MAX_TOKENS=$LLM_NCT_RERANK_MAX_TOKENS"
    echo "LLM_NCT_RERANK_RRF_K=$LLM_NCT_RERANK_RRF_K"
    echo

    echo "# --- 6. Trial-level context / trial cards ---"
    echo "TRIAL_LEVEL_CONTEXT=$TRIAL_LEVEL_CONTEXT"
    echo "TRIAL_LEVEL_CONTEXT_TOP_K=$TRIAL_LEVEL_CONTEXT_TOP_K"
    echo "TRIAL_LEVEL_CONTEXT_MAX_CHARS_PER_TRIAL=$TRIAL_LEVEL_CONTEXT_MAX_CHARS_PER_TRIAL"
    echo "TRIAL_LEVEL_CONTEXT_MIN_LLM_SCORE=$TRIAL_LEVEL_CONTEXT_MIN_LLM_SCORE"
    echo "TRIAL_LEVEL_CONTEXT_RESULTS_CHARS=$TRIAL_LEVEL_CONTEXT_RESULTS_CHARS"
    echo "TRIAL_LEVEL_CONTEXT_OUTCOME_RESULTS_CHARS=$TRIAL_LEVEL_CONTEXT_OUTCOME_RESULTS_CHARS"
    echo "TRIAL_LEVEL_CONTEXT_SAFETY_RESULTS_CHARS=$TRIAL_LEVEL_CONTEXT_SAFETY_RESULTS_CHARS"
    echo "TRIAL_LEVEL_CONTEXT_SUMMARY_CHARS=$TRIAL_LEVEL_CONTEXT_SUMMARY_CHARS"
    echo "TRIAL_LEVEL_CONTEXT_KEEP_RESULT_EVIDENCE=$TRIAL_LEVEL_CONTEXT_KEEP_RESULT_EVIDENCE"
    echo "TRIAL_LEVEL_CONTEXT_RESULT_EVIDENCE_K=$TRIAL_LEVEL_CONTEXT_RESULT_EVIDENCE_K"
    echo "TRIAL_LEVEL_CONTEXT_GLOBAL_DEDUPE=$TRIAL_LEVEL_CONTEXT_GLOBAL_DEDUPE"
    echo "TRIAL_LEVEL_CONTEXT_DEDUPE_WITH_LEARNINGS=$TRIAL_LEVEL_CONTEXT_DEDUPE_WITH_LEARNINGS"
    echo "TRIAL_LEVEL_CONTEXT_GLOBAL_FALLBACK_K=$TRIAL_LEVEL_CONTEXT_GLOBAL_FALLBACK_K"
    echo

    echo "# --- 7. Final context assembly / token budgets ---"
    echo "FINAL_CONTEXT_MAX_WORDS=$FINAL_CONTEXT_MAX_WORDS"
    echo "FINAL_CONTEXT_LEARNING_WORDS=$FINAL_CONTEXT_LEARNING_WORDS"
    echo "SMART_TOKEN_LIMIT=$SMART_TOKEN_LIMIT"
    echo "STRATEGIC_TOKEN_LIMIT=$STRATEGIC_TOKEN_LIMIT"
    echo "MAX_TOKENS=$MAX_TOKENS"
    echo "MAX_COMPLETION_TOKENS=$MAX_COMPLETION_TOKENS"
    echo "FAST_TOKEN_LIMIT=$FAST_TOKEN_LIMIT"
    echo "SUMMARY_TOKEN_LIMIT=$SUMMARY_TOKEN_LIMIT"
    echo "TOTAL_WORDS=$TOTAL_WORDS"
    echo "LLM_OUTPUT_TOKEN_MARGIN=$LLM_OUTPUT_TOKEN_MARGIN"
    echo "SMART_LLM_CONTEXT_WINDOW=$SMART_LLM_CONTEXT_WINDOW"
    echo "FAST_LLM_CONTEXT_WINDOW=$FAST_LLM_CONTEXT_WINDOW"
    echo "STRATEGIC_LLM_CONTEXT_WINDOW=$STRATEGIC_LLM_CONTEXT_WINDOW"
    echo "MAX_CONTEXT_SIZE=$MAX_CONTEXT_SIZE"
    echo "CONTEXT_WINDOW=$CONTEXT_WINDOW"
    echo "TOKEN_BUDGET=$TOKEN_BUDGET"
    echo

    echo "# --- 8. Debug / audit ---"
    echo "SAVE_WEB_CONTEXT_DEBUG=$SAVE_WEB_CONTEXT_DEBUG"
    echo "SAVE_LLM_NCT_RERANK_DEBUG=$SAVE_LLM_NCT_RERANK_DEBUG"
    echo "SAVE_REPORT_PROMPT_AUDIT=$SAVE_REPORT_PROMPT_AUDIT"
} > "$RESOLVED_CONFIG_PATH"
echo "Resolved config for this run saved to: $RESOLVED_CONFIG_PATH"

echo ""
echo "Starting evaluation pipeline..."
EVAL_RANGE_ARGS=()
if [[ -n "$EVAL_START_INDEX" ]]; then
  EVAL_RANGE_ARGS+=(--start-index "$EVAL_START_INDEX")
fi
if [[ -n "$EVAL_END_INDEX" ]]; then
  EVAL_RANGE_ARGS+=(--end-index "$EVAL_END_INDEX")
fi
if [[ "$SKIP_REPORT" == "1" ]]; then
  EVAL_RANGE_ARGS+=(--skip-report)
fi

# Everything this subprocess needs is already exported above; the only
# override here is the deliberate CUDA_VISIBLE_DEVICES difference from
# section 1.
CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" \
  python evaluation_pipeline.py \
  --ground-truth "$GROUND_TRUTH_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --research-agent research_agent.py \
  --search-api search_api.py \
  --port "$S_PORT" \
  "${EVAL_RANGE_ARGS[@]}"

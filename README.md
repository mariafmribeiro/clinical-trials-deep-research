# Clinical Trials Deep Research

Research code and evaluation materials for adapting a general-purpose deep-research agent to retrieve and synthesise evidence from a fixed corpus of ClinicalTrials.gov records.

The system replaces web search in GPT-Researcher with a local BM25 OpenSearch index. It evaluates multi-query retrieval and two approaches to context construction: the original embedding-based chunk compressor and a trial-oriented strategy that deduplicates candidates by NCT identifier, reranks them against the clinical question, and represents selected records as structured trial evidence blocks. Reports are generated with locally served Qwen3-8B or Fleming-R1-7B models and evaluated at retrieval, context, citation, and reference-alignment levels.

## Repository contents

- `gpt_researcher/`: adapted GPT-Researcher implementation.
- `index/`: construction of the BM25 index from ClinicalTrials.gov XML records.
- `experiments/`: local search service, pipeline runner, final model configurations, and retrieval/context experiment suites.
- `evaluation/`: notebooks and scripts for retrieval, context, deterministic citation, diagnostic, and LLM-based report evaluation.
- `data/`: the 51-question benchmark and derived reference key points.

## Provenance

The implementation is based on GPT-Researcher commit `b364917f55ea579c47e5ef3f038f7e56f51213df`. The seven modified upstream modules and the additional experiment code in this repository implement the ClinicalTrials.gov adaptations described in the accompanying dissertation. See [NOTICE.md](NOTICE.md) for attribution.

## Requirements

The experiments were developed for Linux with Python 3.10+, OpenSearch, and a CUDA-capable GPU. The final local model configurations use a 32,768-token context window and one active sequence. Model checkpoints are downloaded separately from Hugging Face:

- `Qwen/Qwen3-8B`
- `IQuestLab/Fleming-R1-7B`

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiments.txt
```

Install a CUDA-compatible version of `vllm` separately when running the models locally. Copy `.env.example` to `.env`, replace the placeholder credentials, and export its values before running an experiment:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Never commit the populated `.env` file.

## Build the index

The ClinicalTrials.gov XML snapshot is not redistributed. Point `CLINICAL_TRIALS_XML_DIR` to the extracted corpus, configure OpenSearch in `.env`, and run:

```bash
python index/create_index.py
```

The index stores the fields needed to construct trial cards and uses BM25 over either the aggregated retrieval text, selected structured fields, or both. Posted outcome and safety results are stored for evidence construction but are excluded from BM25 scoring.

## Run the experiments

First verify the final configuration without loading a model:

```bash
DRY_RUN=1 bash experiments/run_qwen.sh
```

Run the selected A4 pipeline with either model:

```bash
bash experiments/run_qwen.sh
bash experiments/run_fleming.sh
```

The complete retrieval and context configuration sequences reported in the dissertation are available as:

```bash
bash experiments/run_retrieval_experiments.sh
bash experiments/run_context_experiments.sh
```

Outputs are written below `outputs/` and include resolved configuration files, retrieval traces, query-generation traces, selected context, reports, and stage-level metrics. The directory is ignored by Git because a complete run is large and contains generated material.

## Evaluate the outputs

The main analysis entry points are:

- `evaluation/retrieval/retrieval_evaluation.ipynb`
- `evaluation/context/context_evaluation.ipynb`
- `evaluation/reports/deterministic_report_evaluation.ipynb`
- `evaluation/reports/report_alignment_evaluation.ipynb`
- `evaluation/reports/three_condition_analysis.ipynb`

The reference-alignment judge can also be run directly. It expects an OpenAI-compatible endpoint serving the configured Gemma model:

```bash
python evaluation/reports/judge_reference_alignment.py
```

The evidence-only and known-trial scripts in `evaluation/reports/` reproduce the two post-hoc diagnostic conditions. They intentionally keep generated outputs separate from the primary A4 runs.

## Interpretation

The benchmark provides positive mappings to known relevant NCT identifiers, not exhaustive relevance labels for every retrieved trial. The known-trial condition bypasses retrieval and candidate selection but remains restricted to registry content; it is therefore a diagnostic reference rather than an oracle for report quality.

## License

This repository is distributed under the Apache License 2.0. The license and attribution of the upstream GPT-Researcher project are preserved.

# Evaluation Results

This directory contains the aggregate tables and figures reported in the dissertation. The artefacts are generated from the 51-review higher-confidence benchmark subset unless a filename or table states otherwise.

The full experiment outputs are intentionally not versioned. They include generated reports, contexts, query traces, per-review diagnostics, and raw judge responses, and remain under the Git-ignored `outputs/` directory. The published files contain only aggregate results needed to inspect the main retrieval, context-construction, citation, and reference-alignment findings.

## Reproduce the analyses

After generating or restoring the experiment outputs, run the notebooks from the repository root:

```bash
python3 evaluation/run_notebook.py evaluation/retrieval/retrieval_evaluation.ipynb
python3 evaluation/run_notebook.py evaluation/context/context_evaluation.ipynb
python3 evaluation/run_notebook.py evaluation/reports/deterministic_report_evaluation.ipynb
python3 evaluation/run_notebook.py evaluation/reports/report_alignment_evaluation.ipynb
python3 evaluation/run_notebook.py evaluation/reports/three_condition_analysis.ipynb
python3 evaluation/publish_results.py
```

The semantic notebooks consume the raw outputs produced by the reference-alignment judge and the two diagnostic conditions. These raw files are not included here because they contain report-level text and detailed model judgements. The `manifest.csv` file records the source path, size, and SHA-256 digest of every published artefact.

## Contents

- `retrieval/`: retrieval coverage, ranking, searchable-field, and request-count summaries.
- `context/`: evidence-preservation, reranking, packing, and trial-block comparisons.
- `reports/deterministic/`: citation and format metrics derived without an LLM judge.
- `reports/alignment/`: aggregate reference-alignment results from the primary A4 condition.
- `reports/diagnostics/`: aggregate comparisons of A4, evidence-only, and known-trial conditions.

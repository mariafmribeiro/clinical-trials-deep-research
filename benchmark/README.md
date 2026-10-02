# Evaluation data

`benchmark_complete.json` contains all 881 reviews with at least one mapped ClinicalTrials.gov identifier. `benchmark.json` contains the 51-review higher-confidence subset used for pipeline development and report analysis.

Each entry links a Cochrane-derived clinical question to mapped ClinicalTrials.gov identifiers. The 51-review subset contains cases for which the reported number of included RCTs equals the number of mapped NCT identifiers. The mappings identify known relevant trials and are not exhaustive relevance judgements over the corpus.

`reference_keypoints.jsonl` contains the derived overall conclusions and three to five reference key points used by the report-alignment evaluation. The reference representation was generated with Gemma 4-31B and structurally validated. Full source passages from the Cochrane reviews and raw model responses are intentionally not redistributed; source identifiers, URLs, and content hashes are retained for provenance.

The ClinicalTrials.gov XML snapshot used to build the search index is not stored in this repository. The collection is available through the [TREC 2023 Clinical Trials Track](https://www.trec-cds.org/2023.html). Set `CLINICAL_TRIALS_XML_DIR` to the location of the extracted corpus before building the index.

The `construction/` directory contains the complete benchmark-construction workflow. `build_benchmark.py` performs PubMed/Cochrane extraction and NCT mapping; it extracts the review question and RCT count but does not generate a reference report. `prepare_benchmarks.py` creates the 881-review complete benchmark and its 51-review higher-confidence subset while restricting the published records to the fields needed by the experiments. `fetch_reference_sections.py` and `prepare_reference_keypoints.py` reproduce the source-section retrieval and key-point construction stages. These scripts require the corresponding API variables in `.env`; no credentials are stored in the repository.

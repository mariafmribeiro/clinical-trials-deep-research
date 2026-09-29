# Evaluation data

`benchmark.json` contains the 51-question higher-confidence evaluation set used for pipeline development and report analysis. Each entry links a Cochrane-derived clinical question to mapped ClinicalTrials.gov identifiers. The mappings identify known relevant trials and are not exhaustive relevance judgements over the corpus.

`reference_keypoints.jsonl` contains the derived overall conclusions and three to five reference key points used by the report-alignment evaluation. The reference representation was generated with Gemma 4-31B and structurally validated. Full source passages from the Cochrane reviews and raw model responses are intentionally not redistributed; source identifiers, URLs, and content hashes are retained for provenance.

The ClinicalTrials.gov XML snapshot used to build the search index is not stored in this repository. Place the TREC 2023 Clinical Trials Track corpus, or a compatible ClinicalTrials.gov XML snapshot, in `data/clinical_trials/` or set `CLINICAL_TRIALS_XML_DIR` to its location.

`build_benchmark.py` contains the PubMed/Cochrane extraction and NCT-mapping workflow. `evaluation/reports/fetch_reference_sections.py` and `prepare_reference_keypoints.py` reproduce the source-section retrieval and key-point construction stages. These scripts require the corresponding API variables in `.env`; no credentials are stored in the repository.

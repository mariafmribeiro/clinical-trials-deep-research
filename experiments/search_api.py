import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from opensearchpy import OpenSearch


app = FastAPI()

# -----------------------------
# 0. OpenSearch config
# -----------------------------

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "")

OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "clinical_trials_bm25_v1")
OPENSEARCH_URL_PREFIX = os.getenv("OPENSEARCH_URL_PREFIX", "")
OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() in {"1", "true", "yes"}
OPENSEARCH_VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() in {"1", "true", "yes"}

TRACE_PATH = os.getenv("WHOOSH_TRACE_PATH", os.getenv("OPENSEARCH_TRACE_PATH", "outputs/opensearch_trace_test.jsonl"))
MAX_CONTENT_CHARS = int(os.getenv("MAX_CONTENT_CHARS", "8000"))

TRIAL_CARD_BRIEF_SUMMARY_CHARS = int(os.getenv("TRIAL_CARD_BRIEF_SUMMARY_CHARS", "800"))
TRIAL_CARD_DETAILED_DESCRIPTION_CHARS = int(os.getenv("TRIAL_CARD_DETAILED_DESCRIPTION_CHARS", "600"))
TRIAL_CARD_ELIGIBILITY_CHARS = int(os.getenv("TRIAL_CARD_ELIGIBILITY_CHARS", "1000"))
TRIAL_CARD_OUTCOMES_CHARS = int(os.getenv("TRIAL_CARD_OUTCOMES_CHARS", "800"))
TRIAL_CARD_CLINICAL_OUTCOME_RESULTS_CHARS = int(
    os.getenv("TRIAL_CARD_CLINICAL_OUTCOME_RESULTS_CHARS", "3500")
)
TRIAL_CARD_CLINICAL_SAFETY_RESULTS_CHARS = int(
    os.getenv("TRIAL_CARD_CLINICAL_SAFETY_RESULTS_CHARS", "1200")
)

if not OPENSEARCH_PASSWORD:
    raise RuntimeError(
        "Missing OPENSEARCH_PASSWORD. Set it as an environment variable."
    )

client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=OPENSEARCH_USE_SSL,
    url_prefix=OPENSEARCH_URL_PREFIX,
    verify_certs=OPENSEARCH_VERIFY_CERTS,
    ssl_assert_hostname=OPENSEARCH_VERIFY_CERTS,
    ssl_show_warn=OPENSEARCH_VERIFY_CERTS,
    timeout=120,
    max_retries=3,
    retry_on_timeout=True,
)


# -----------------------------
# 1. Query cleanup
# -----------------------------

def clean_query(q: str) -> str:
    if not q:
        return ""

    q = q.strip()

    q = re.sub(r"IMPORTANT CITATION RULES:.*", " ", q, flags=re.I | re.DOTALL)
    q = re.sub(r"CITATION RULES:.*", " ", q, flags=re.I | re.DOTALL)
    q = re.sub(r"Use only the retrieved.*", " ", q, flags=re.I | re.DOTALL)
    q = re.sub(r"Do not invent.*", " ", q, flags=re.I | re.DOTALL)
    q = re.sub(r"Every citation must.*", " ", q, flags=re.I | re.DOTALL)
    q = re.sub(r"The References section must.*", " ", q, flags=re.I | re.DOTALL)

    q = re.sub(r"Initial Query:\s*", " ", q, flags=re.I)
    q = re.sub(r"Original research question:\s*", " ", q, flags=re.I)
    q = re.sub(r"Additional registry search directions:\s*", " ", q, flags=re.I)
    q = re.sub(r"Previous research goal:\s*", " ", q, flags=re.I)
    q = re.sub(r"Follow-up questions:\s*", " ", q, flags=re.I)

    q = re.sub(r"Goal:\s*", " ", q, flags=re.I)
    q = re.sub(r"Query:\s*", " ", q, flags=re.I)

    q = re.sub(r"\[[^\]]+\]", " ", q)
    q = q.replace("â€œ", '"').replace("â€", '"').replace("â€™", "'")
    q = re.sub(r"\s+", " ", q).strip()

    if len(q) > 500:
        q = q[:500].rsplit(" ", 1)[0]

    print(f"Cleaned Query: '{q}'", flush=True)
    return q


def make_loose_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"\b(and|or|not)\b", " ", q, flags=re.I)
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


# -----------------------------
# 2. Logging helpers
# -----------------------------

def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row["_logged_at"] = datetime.now().isoformat(timespec="seconds")

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# -----------------------------
# 3. Formatting helpers
# -----------------------------

def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, list):
        value = "; ".join(str(v) for v in value if v is not None)

    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def truncate_text(value, max_chars=1500):
    value = clean_text(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + f" ... [truncated to {max_chars} chars]"


def first_nonempty(*values):
    for value in values:
        value = clean_text(value)
        if value:
            return value
    return ""


def build_trial_card(src):
    nct_id = first_nonempty(src.get("nct_id"), "NO_NCT_ID")
    brief_title = first_nonempty(src.get("brief_title"), nct_id)
    official_title = first_nonempty(src.get("official_title"))
    has_clinical_results = bool(src.get("has_clinical_results"))

    conditions = first_nonempty(
        src.get("conditions"),
        src.get("conditions_text"),
    )

    interventions = first_nonempty(
        src.get("intervention_names"),
        src.get("interventions"),
        src.get("interventions_text"),
    )

    arms = first_nonempty(
        src.get("arm_group_labels"),
        src.get("arms"),
        src.get("arms_text"),
    )

    lines = [
        "[LOCAL_OPENSEARCH_TRIAL_CARD]",
        f"NCT ID: {nct_id}",
        f"Brief title: {brief_title}",
    ]

    optional_fields = [
        ("Official title", official_title),
        ("Overall status", src.get("overall_status")),
        ("Study type", src.get("study_type")),
        ("Phase", src.get("phase")),
        ("Enrollment", src.get("enrollment")),
        ("Start date", src.get("start_date_raw") or src.get("start_date")),
        ("Completion date", src.get("completion_date_raw") or src.get("completion_date")),
        ("Completion date type", src.get("completion_date_type")),
        ("Primary completion date", src.get("primary_completion_date_raw") or src.get("primary_completion_date")),
        ("Last update posted", src.get("last_update_posted_raw") or src.get("last_update_posted")),
        ("Conditions", conditions),
        ("Interventions", interventions),
        ("Keywords", src.get("keywords")),
        ("Arms / groups", arms),
        # Keep the summary available to the reranker before the full card cap.
        ("Brief summary", truncate_text(src.get("brief_summary"), TRIAL_CARD_BRIEF_SUMMARY_CHARS)),
        # These source-only fields contain posted values rather than registry
        # outcome definitions. Keep outcomes and safety independently bounded.
        ("Clinical outcome results", truncate_text(
            src.get("clinical_results_outcomes_compact"),
            TRIAL_CARD_CLINICAL_OUTCOME_RESULTS_CHARS,
        )),
        ("Clinical safety results", truncate_text(
            src.get("clinical_results_safety_compact"),
            TRIAL_CARD_CLINICAL_SAFETY_RESULTS_CHARS,
        )),
        ("Study design", src.get("study_design")),
        ("Primary outcomes", truncate_text(src.get("primary_outcomes_text") or src.get("primary_outcomes"), TRIAL_CARD_OUTCOMES_CHARS)),
        ("Secondary outcomes", truncate_text(src.get("secondary_outcomes_text") or src.get("secondary_outcomes"), TRIAL_CARD_OUTCOMES_CHARS)),
        ("Detailed description", truncate_text(src.get("detailed_description"), TRIAL_CARD_DETAILED_DESCRIPTION_CHARS)),
        ("Eligibility / population", truncate_text(src.get("eligibility_text") or src.get("eligibility"), TRIAL_CARD_ELIGIBILITY_CHARS)),
        ("Has clinical results", "true" if has_clinical_results else ""),
        ("Clinical results raw XML chars", src.get("clinical_results_raw_xml_chars")),
        ("Clinical results plain text chars", src.get("clinical_results_plain_text_chars")),
        ("XML relative path", src.get("xml_path")),
    ]

    for label, value in optional_fields:
        value = clean_text(value)
        if value:
            lines.append(f"{label}: {value}")

    lines.append("[END_LOCAL_OPENSEARCH_TRIAL_CARD]")
    return "\n".join(lines)


# -----------------------------
# 4. OpenSearch retrieval config
# -----------------------------
SOURCE_FIELDS = [
    "nct_id",
    "brief_title",
    "official_title",
    "brief_summary",
    "detailed_description",
    "overall_status",
    "study_type",
    "phase",
    "enrollment",
    "start_date",
    "start_date_raw",
    "completion_date",
    "completion_date_raw",
    "completion_date_type",
    "primary_completion_date",
    "primary_completion_date_raw",
    "last_update_posted",
    "last_update_posted_raw",
    "conditions",
    "keywords",
    "intervention_names",
    "interventions_text",
    "arm_group_labels",
    "arms_text",
    "primary_outcomes_text",
    "secondary_outcomes_text",
    "eligibility_text",
    "has_clinical_results",
    "clinical_results_outcomes_compact",
    "clinical_results_safety_compact",
    "clinical_results_raw_xml_chars",
    "clinical_results_plain_text_chars",
    "xml_path",
]

STRUCTURED_FIELDS = [
    "brief_title",
    "official_title",
    "keywords",
    "conditions",
    "condition_mesh_terms",
    "intervention_names",
    "intervention_mesh_terms",
]

FIELD_SETS = {
    "aggregated": ["retrieval_text"],
    "retrieval_text_only": ["retrieval_text"],
    "structured": STRUCTURED_FIELDS,
    "structured_fields": STRUCTURED_FIELDS,
    "hybrid": [*STRUCTURED_FIELDS, "retrieval_text"],
    "hybrid_unboosted": [*STRUCTURED_FIELDS, "retrieval_text"],
}

FIELD_SET_NAME = os.getenv("FIELD_SET_NAME", "aggregated")
if FIELD_SET_NAME not in FIELD_SETS:
    available = ", ".join(sorted(FIELD_SETS))
    raise ValueError(f"Unknown FIELD_SET_NAME={FIELD_SET_NAME!r}. Choose one of: {available}")
OPENSEARCH_FIELDS = FIELD_SETS[FIELD_SET_NAME]


def build_query_body(search_query, size):
    return {
        "size": size,
        "_source": SOURCE_FIELDS,
        "query": {
            "multi_match": {
                "query": search_query,
                "fields": OPENSEARCH_FIELDS,
                "type": "best_fields",
                "operator": "or",
            }
        },
    }


def run_search(search_query, max_results):
    query_body = build_query_body(
        search_query=search_query,
        size=max_results,
    )

    print(f"OpenSearch query: {search_query}", flush=True)

    response = client.search(
        index=OPENSEARCH_INDEX_NAME,
        body=query_body,
    )

    hits = response["hits"]["hits"]
    return hits, query_body

# -----------------------------
# 6. API
# -----------------------------

@app.get("/health")
def health():
    try:
        info = client.info()
        return {
            "status": "ok",
            "backend": "opensearch",
            "index_name": OPENSEARCH_INDEX_NAME,
            "cluster_name": info.get("cluster_name"),
        }
    except Exception as e:
        return {
            "status": "error",
            "backend": "opensearch",
            "index_name": OPENSEARCH_INDEX_NAME,
            "error": str(e),
        }


@app.get("/search")
def search(query: str = None, q: str = None, max_results: int = 300, cutoff_date: str = None):
    try:
        raw_query = query or q or ""
        search_query = raw_query.strip()

        if not search_query:
            return []

        print(f"API Received Query: '{search_query}'", flush=True)

        results_list = []

        hits, query_body = run_search(
            search_query=search_query,
            max_results=max_results,
        )

        print(f"OpenSearch hits returned: {len(hits)}", flush=True)

        for raw_rank, hit in enumerate(hits, start=1):
            src = hit.get("_source", {})
            nct_id = src.get("nct_id", "local_path")
            title = first_nonempty(src.get("brief_title"), src.get("official_title"), nct_id)

            clinicaltrials_url = f"https://clinicaltrials.gov/study/{nct_id}"
            local_url = f"local://{nct_id}"

            local_content = build_trial_card(src=src)

            if len(local_content) > MAX_CONTENT_CHARS:
                local_content = (
                    local_content[:MAX_CONTENT_CHARS]
                    + f"\n\n[TRUNCATED: trial card was longer than {MAX_CONTENT_CHARS} characters]"
                )

            search_score = float(hit.get("_score")) if hit.get("_score") is not None else None

            result_doc = {
                "url": local_url,
                "href": local_url,
                "title": title,
                "raw_content": local_content,
                "body": local_content,
                "content": local_content,
                "nct_id": nct_id,
                "score": search_score,
                "clinicaltrials_url": clinicaltrials_url,
                "xml_path": src.get("xml_path"),
            }

            results_list.append(result_doc)

            append_jsonl(
                TRACE_PATH,
                {
                    "backend": "opensearch",
                    "raw_query": raw_query,
                    "search_query": search_query,
                    "query_body": query_body,
                    "raw_rank": raw_rank,
                    "final_rank": len(results_list),
                    "nct_id": nct_id,
                    "title": title,
                    "local_url": local_url,
                    "clinicaltrials_url": clinicaltrials_url,
                    "score": search_score,
                    "overall_status": src.get("overall_status"),
                    "completion_date": src.get("completion_date"),
                    "completion_date_raw": src.get("completion_date_raw"),
                    "xml_path": src.get("xml_path"),
                    "content_chars_sent": len(local_content),
                    "cutoff_date": cutoff_date,
                    "content_preview": local_content[:1500],
                    "sent_to_gpt_researcher": True,
                },
            )

        print(f"\nReturned {len(results_list)} results", flush=True)
        return results_list

    except Exception:
        import traceback
        print(traceback.format_exc(), flush=True)
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"Starting OpenSearch Search API on port {args.port}...", flush=True)
    print(f"Using OpenSearch index: {OPENSEARCH_INDEX_NAME}", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)

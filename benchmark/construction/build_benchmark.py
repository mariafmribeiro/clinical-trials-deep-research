#!/usr/bin/env python3
"""Build the Cochrane-derived question and ClinicalTrials.gov mapping data."""

import argparse
import json
import os
import re
import time
import uuid

import requests
from Bio import Entrez
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

Entrez.email = os.getenv("NCBI_EMAIL", "")
LLM_API_KEY = os.getenv("BENCHMARK_LLM_API_KEY", "")
LLM_ENDPOINT = os.getenv("BENCHMARK_LLM_ENDPOINT", "")
LLM_CHANNEL_ID = os.getenv("BENCHMARK_LLM_CHANNEL_ID", "")

PUBMED_QUERY = "Cochrane Database Syst Rev[Journal] AND 2020/01/01:2023/05/08[DP]"
OUTPUT_FILE = os.getenv("BENCHMARK_OUTPUT", "benchmark/benchmark_extraction.json")
MINIMUM_RCT_COUNT = 0


# ---------------------------------------------------------------------------
# PubMed and LLM functions
# ---------------------------------------------------------------------------


def get_abstract_sections(article_data):
    abstract_list = article_data.get("Abstract", {}).get("AbstractText", [])
    full_parts = []

    for section in abstract_list:
        label = section.attributes.get("Label", "") if hasattr(section, "attributes") else ""
        section_text = str(section)
        full_parts.append(f"{label}: {section_text}" if label else section_text)

    return "\n".join(full_parts)


def get_lean_abstract(text):
    # MAIN RESULTS and AUTHORS' CONCLUSIONS are not always present as separate
    # headers. Some Cochrane abstracts only carry a "Data collection and
    # analysis" section (which is where the trial/study count usually shows
    # up in that case), sometimes followed directly by conclusions. All five
    # headers are captured so the LLM still sees the count and the
    # conclusions even when the "usual" headers are missing.
    headers = [
        "OBJECTIVES",
        "SELECTION CRITERIA",
        "DATA COLLECTION AND ANALYSIS",
        "MAIN RESULTS",
        "AUTHORS CONCLUSIONS",
    ]
    lean_parts = []
    sections = re.split(
        r"(\n[A-Z\s']{5,}:|Objectives:|Selection criteria:"
        r"|Data collection and analysis:|Main results:|Authors'? conclusions:)",
        text,
    )
    current_header = ""

    for part in sections:
        clean_part = re.sub(r"[^A-Z\s]", "", part.strip().upper())
        clean_part = re.sub(r"\s+", " ", clean_part).strip()

        if clean_part in headers:
            current_header = clean_part
            continue

        if current_header in headers:
            lean_parts.append(f"{current_header}: {part.strip()}")
            current_header = ""

    return "\n".join(lean_parts) if lean_parts else text


def analyze_review_with_agent(title, abstract_text):
    lean_abstract = get_lean_abstract(abstract_text)

    prompt = f"""
Act as a Cochrane Review Auditor.

Return ONLY valid JSON:

{{
  "is_exclusive_rct": boolean,
  "rct_count": integer,
  "research_query": ""
}}

Rules:

CRITERIA for "is_exclusive_rct":

- TRUE only if inclusion is strictly limited to randomized controlled trials
  or quasi-randomized trials.
- FALSE if the review includes observational, cohort, case-control,
  uncontrolled, or other non-randomized studies.

INSTRUCTION for "rct_count":

- Scan the Main Results for the number of included trials or studies.
- If there is no Main Results section, or it does not state a count, scan
  the Data Collection and Analysis section instead.
- Convert numbers written as words to integers.
- If no trial or study count is found in either section, return 0.
- This number must appear only in the "rct_count" field.

INSTRUCTION for "research_query":

- Write a professional research question based on the Objectives section.
- The question should guide a Deep Research tool to generate a report about
  this exact review topic.

Title: {title}

Abstract:
{lean_abstract}
"""

    headers = {"x-api-key": LLM_API_KEY}
    multipart_form_data = {
        "channel_id": (None, str(LLM_CHANNEL_ID)),
        "thread_id": (None, str(uuid.uuid4())),
        "user_info": (None, "{}"),
        "message": (None, prompt),
    }

    try:
        response = requests.post(
            LLM_ENDPOINT,
            headers=headers,
            files=multipart_form_data,
            stream=True,
            timeout=120,
        )

        if response.status_code == 429 or response.status_code in [500, 502, 503, 504]:
            return "RATE_LIMIT"

        if response.status_code != 200:
            print(f"Unexpected API error: {response.status_code}", flush=True)
            return None

        full_text = ""

        for line in response.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data:"):
                decoded = decoded[5:].strip()

            try:
                data = json.loads(decoded)
                if data.get("type") == "token":
                    full_text += str(data.get("content", ""))
            except Exception:
                continue

        clean = full_text.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}")

        if start == -1 or end == -1:
            return None

        return json.loads(clean[start : end + 1])

    except Exception as error:
        print(f"LLM error: {error}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Cochrane DOI and trial-registration extraction functions
# ---------------------------------------------------------------------------


COCHRANE_SESSION = requests.Session()
COCHRANE_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
)

TRIAL_PATTERNS = [
    r"NCT\s*[-\u2013\u2014]?\s*\d{8}",
    r"ISRCTN\s*[-\u2013\u2014]?\s*\d{8}",
    r"\b\d{4}\s*-\s*\d{6}\s*-\s*\d{2}\b",
    r"ChiCTR[-\s]?\w{0,4}[-\s]?\d{7,9}",
    r"ACTRN\s*\d{14}",
    r"DRKS\d{8}",
    r"CTRI/\d{4}/\d{2,3}/\d{6}",
    r"UMIN\d{9}",
    r"JPRN-\w+\d+",
    r"RBR-\w{6,8}",
    r"PACTR\d{15,18}",
    r"KCT\d{7}",
    r"TCTR\d{11}",
    r"IRCT\d{8,15}N?\d*",
    r"NTR\d{3,5}",
]


def pmid_to_doi(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    try:
        response = requests.get(
            url,
            params={"db": "pubmed", "id": pmid, "retmode": "json"},
            timeout=20,
        )
        response.raise_for_status()
        article_ids = response.json().get("result", {}).get(pmid, {}).get("articleids", [])

        for article_id in article_ids:
            if article_id.get("idtype") == "doi":
                return article_id.get("value")

    except Exception as error:
        print(f"  -> DOI lookup error: {error}", flush=True)

    return None


def normalize_trial_id(match):
    if "-" in match:
        return re.sub(r"\s", "", match).upper()
    return re.sub(r"[\s\-\u2013\u2014]", "", match).upper()


def extract_trial_ids(text):
    identifiers = set()

    for pattern in TRIAL_PATTERNS:
        for match in re.findall(pattern, text, re.IGNORECASE):
            identifiers.add(normalize_trial_id(match))

    return identifiers


# Not currently used: flags "registration" mentions near the text that don't
# match a known trial-ID pattern, for manual review. Disabled since the
# output isn't needed.
# def find_unmatched_registration_mentions(text):
#     flagged = []
#     seen = set()
#
#     for match in re.finditer(r"registration", text, re.IGNORECASE):
#         start = max(0, match.start() - 80)
#         end = min(len(text), match.end() + 80)
#         window = text[start:end]
#
#         if any(re.search(pattern, window, re.IGNORECASE) for pattern in TRIAL_PATTERNS):
#             continue
#
#         snippet = re.sub(r"\s+", " ", window).strip()
#         if snippet not in seen:
#             seen.add(snippet)
#             flagged.append(snippet)
#
#     return flagged


def classify_heading(heading):
    text = heading.get_text(" ", strip=True).lower()

    if "excluded" in text:
        return "excluded"
    if "awaiting" in text:
        return "awaiting_assessment"
    if "ongoing" in text:
        return "ongoing"
    if "included" in text:
        return "included"

    return None


def classify_id_location(node):
    heading = node.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])

    while heading is not None:
        category = classify_heading(heading)
        if category is not None:
            return category
        heading = heading.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])

    return None


def clean_soup(soup):
    for element in soup.find_all(["nav", "aside", "header", "footer", "script", "style"]):
        element.decompose()

    for element in soup.find_all(attrs={"aria-hidden": "true"}):
        element.decompose()

    noisy_pattern = re.compile(
        r"(menu|sidebar|toc|nav-tabs|sticky|banner|breadcrumb)",
        re.IGNORECASE,
    )

    for element in soup.find_all(class_=noisy_pattern):
        element.decompose()

    for element in soup.find_all(id=noisy_pattern):
        element.decompose()

    return soup


def extract_sections_by_backward_walk(soup):
    clean_soup(soup)

    combined_pattern = "|".join(f"(?:{pattern})" for pattern in TRIAL_PATTERNS)
    text_nodes = soup.find_all(string=re.compile(combined_pattern, re.IGNORECASE))

    buckets = {
        "included": set(),
        "excluded": set(),
        "awaiting_assessment": set(),
        "ongoing": set(),
    }
    unclassified = set()

    for node in text_nodes:
        found_ids = extract_trial_ids(str(node))
        if not found_ids:
            continue

        category = classify_id_location(node)
        if category is not None:
            buckets[category].update(found_ids)
        else:
            unclassified.update(found_ids)

    result = {key: sorted(value) for key, value in buckets.items()}

    if unclassified:
        result["_unclassified"] = sorted(unclassified)

    return result


def fetch_cochrane_page(url, doi):
    try:
        response = COCHRANE_SESSION.get(url, timeout=20)

        if response.status_code != 200:
            print(f"  -> Failed to access {url}: HTTP {response.status_code}", flush=True)
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(" ", strip=True).lower()

        if "references to studies" not in page_text and "included studies" not in page_text:
            print(f"  -> Cochrane page for {doi} did not contain study references", flush=True)
            return None

        return soup

    except Exception as error:
        print(f"  -> Cochrane fetch error: {error}", flush=True)
        return None


def split_ids_by_registry_type(identifiers):
    nct_ids = sorted(identifier for identifier in identifiers if identifier.startswith("NCT"))
    other_ids = sorted(identifier for identifier in identifiers if not identifier.startswith("NCT"))
    return nct_ids, other_ids


def extract_all_registries(doi):
    result = {
        "included_nct": [],
        "included_other": [],
        "included_count": 0,
        "excluded_nct": [],
        "excluded_other": [],
        "awaiting_assessment_nct": [],
        "awaiting_assessment_other": [],
        "ongoing_nct": [],
        "ongoing_other": [],
    }

    categories = ["included", "excluded", "awaiting_assessment", "ongoing"]
    merged_buckets = {category: set() for category in categories}
    merged_unclassified = set()
    # merged_flagged = set()  # disabled: flagged-for-review is not used
    successful_pages = 0

    urls = [
        f"https://www.cochranelibrary.com/cdsr/doi/{doi}/references",
        f"https://www.cochranelibrary.com/cdsr/doi/{doi}/full",
    ]

    for page_index, url in enumerate(urls):
        soup = fetch_cochrane_page(url, doi)

        if soup is not None:
            successful_pages += 1
            page_result = extract_sections_by_backward_walk(soup)

            for category in categories:
                merged_buckets[category].update(page_result.get(category, []))

            merged_unclassified.update(page_result.get("_unclassified", []))
            # merged_flagged.update(
            #     find_unmatched_registration_mentions(soup.get_text(" ", strip=True))
            # )

        if page_index == 0:
            time.sleep(1.5)

    # A blocked/failed download is not evidence that a review has no trial IDs.
    # Return a failure marker so the caller does not save a false empty record.
    if successful_pages == 0:
        return None

    for category in categories:
        nct_ids, other_ids = split_ids_by_registry_type(merged_buckets[category])
        result[f"{category}_nct"] = nct_ids
        result[f"{category}_other"] = other_ids

    result["included_count"] = len(merged_buckets["included"])

    if merged_unclassified:
        result["_unclassified"] = sorted(merged_unclassified)

    # if merged_flagged:
    #     result["_flagged_for_review"] = sorted(merged_flagged)

    return result


# ---------------------------------------------------------------------------
# Direct pipeline
# ---------------------------------------------------------------------------


def run_pipeline(start_from=1):
    if not Entrez.email:
        raise RuntimeError("Set NCBI_EMAIL before running.")
    if not LLM_API_KEY or not LLM_ENDPOINT or not LLM_CHANNEL_ID:
        raise RuntimeError(
            "Set BENCHMARK_LLM_API_KEY, BENCHMARK_LLM_ENDPOINT, and "
            "BENCHMARK_LLM_CHANNEL_ID before running."
        )

    print("--- SEARCHING PUBMED ---", flush=True)

    handle = Entrez.esearch(db="pubmed", term=PUBMED_QUERY, retmax=0)
    total_found = int(Entrez.read(handle)["Count"])
    handle.close()

    handle = Entrez.esearch(db="pubmed", term=PUBMED_QUERY, retmax=total_found)
    id_list = Entrez.read(handle)["IdList"]
    handle.close()

    print(f"Found {len(id_list)} reviews.", flush=True)

    if start_from > 1:
        print(
            f"--start-from {start_from}: skipping positions 1-{start_from - 1} "
            f"of {len(id_list)} without contacting the LLM.",
            flush=True,
        )

    final_results = []
    processed_pmids = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
                final_results = json.load(file)
            processed_pmids = {str(item["pmid"]) for item in final_results}
            print(f"Resuming: {len(processed_pmids)} reviews already saved.", flush=True)
        except Exception:
            print("Could not read the existing output. Starting fresh.", flush=True)
            final_results = []
            processed_pmids = set()

    for index, pmid_value in enumerate(id_list):
        # Positions are 1-based to match the "[N/Total]" counter printed below,
        # so --start-from 601 resumes exactly where a previous run printed
        # "[601/1622]" when the LLM key started erroring out.
        if index + 1 < start_from:
            continue

        pmid = str(pmid_value)

        if pmid in processed_pmids:
            continue

        print(f"[{index + 1}/{len(id_list)}] Checking PMID {pmid}", flush=True)

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="xml",
                retmode="text",
            )
            records = Entrez.read(handle)
            handle.close()

            article_data = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
            title = str(article_data.get("ArticleTitle", ""))
            full_abstract = get_abstract_sections(article_data)

        except Exception as error:
            print(f"  -> PubMed fetch error: {error}", flush=True)
            continue

        analysis = None

        for attempt in range(3):
            analysis = analyze_review_with_agent(title, full_abstract)

            if analysis == "RATE_LIMIT":
                print("  -> Rate limit. Sleeping 30 seconds.", flush=True)
                time.sleep(30)
            elif analysis is None:
                print("  -> LLM failed. Sleeping 10 seconds before retry.", flush=True)
                time.sleep(10)
            else:
                break

        if not analysis or analysis == "RATE_LIMIT":
            print("  -> Skipped: LLM analysis failed.", flush=True)
            continue

        is_rct = str(analysis.get("is_exclusive_rct", "")).lower() == "true"

        try:
            rct_count = int(analysis.get("rct_count", 0))
        except Exception:
            rct_count = 0

        if not is_rct:
            print("  -> Skipped: not an exclusive RCT review.", flush=True)
            continue

        if rct_count < MINIMUM_RCT_COUNT:
            print(f"  -> Skipped: only {rct_count} RCTs.", flush=True)
            continue

        print("  -> Accepted. Extracting Cochrane sections.", flush=True)

        doi = pmid_to_doi(pmid)

        if doi:
            sections = extract_all_registries(doi)
            if sections is None:
                print(
                    "  -> Cochrane extraction failed; review was NOT saved and can be retried.",
                    flush=True,
                )
                continue
        else:
            print("  -> No DOI found; saving empty sections.", flush=True)
            sections = {}

        # The corrected NCT IDs come only from Cochrane's included studies.
        nct_ids = sections.get("included_nct", []) if sections else []

        result = {
            "title": title,
            "pmid": pmid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "rct_count": rct_count,
            "research_query": analysis.get("research_query", ""),
            "nct_ids": nct_ids,
            "doi": doi,
            "sections": sections,
        }

        final_results.append(result)
        processed_pmids.add(pmid)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(final_results, file, ensure_ascii=False, indent=2)

        print(
            f"  -> Saved: {rct_count} RCTs, {len(nct_ids)} included NCT IDs.",
            flush=True,
        )

        time.sleep(2.5)

    print("\n--- PIPELINE COMPLETE ---", flush=True)
    print(f"Final dataset size: {len(final_results)}", flush=True)
    print(f"Saved to: {OUTPUT_FILE}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help=(
            "1-based position in the PubMed result list to resume from. "
            "Matches the '[N/Total]' counter printed for each PMID, so if the "
            "LLM key started erroring at '[601/1622] Checking PMID ...', pass "
            "--start-from 601 to pick up there without recontacting the LLM "
            "for positions 1-600. Default: 1 (start from the beginning)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(start_from=args.start_from)

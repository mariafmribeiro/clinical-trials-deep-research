from opensearchpy import OpenSearch

from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
from dateutil import parser as date_parser
from opensearchpy.helpers import bulk
import os


XML_FOLDER = Path(os.getenv("CLINICAL_TRIALS_XML_DIR", "data/clinical_trials"))


# =============================================================================
# Generic text / XML helpers
# =============================================================================

def clean_text(x):
    if x is None:
        return None
    return " ".join(x.split()).strip()


def find_text(root, path):
    return clean_text(root.findtext(path))


def find_all_text(root, path):
    values = []
    for el in root.findall(path):
        txt = clean_text(el.text)
        if txt:
            values.append(txt)
    return values


def get_attr(root, path, attr_name):
    el = root.find(path)
    if el is None:
        return None
    return clean_text(el.attrib.get(attr_name))


def parse_date_to_iso(raw_date):
    raw_date = clean_text(raw_date)
    if not raw_date:
        return None

    try:
        dt = date_parser.parse(raw_date, default=datetime(1900, 1, 1))
        return dt.date().isoformat()
    except Exception:
        return None


# =============================================================================
# Structured trial-design fields (interventions, arms, outcomes, references)
# =============================================================================

def parse_interventions(root):
    names = []
    types = []
    descriptions = []
    arm_labels = []
    text_parts = []

    for intervention in root.findall(".//intervention"):
        intervention_type = clean_text(intervention.findtext("intervention_type"))
        intervention_name = clean_text(intervention.findtext("intervention_name"))
        description = clean_text(intervention.findtext("description"))
        arm_group_label = clean_text(intervention.findtext("arm_group_label"))

        if intervention_type:
            types.append(intervention_type)
            text_parts.append(intervention_type)

        if intervention_name:
            names.append(intervention_name)
            text_parts.append(intervention_name)

        if description:
            descriptions.append(description)
            text_parts.append(description)

        if arm_group_label:
            arm_labels.append(arm_group_label)
            text_parts.append(arm_group_label)

    return {
        "intervention_names": names,
        "intervention_types": types,
        "intervention_descriptions": descriptions,
        "intervention_arm_labels": arm_labels,
        "interventions_text": " ".join(text_parts),
    }


def parse_arm_groups(root):
    labels = []
    types = []
    descriptions = []
    text_parts = []

    for arm in root.findall(".//arm_group"):
        label = clean_text(arm.findtext("arm_group_label"))
        arm_type = clean_text(arm.findtext("arm_group_type"))
        description = clean_text(arm.findtext("description"))

        if label:
            labels.append(label)
            text_parts.append(label)

        if arm_type:
            types.append(arm_type)
            text_parts.append(arm_type)

        if description:
            descriptions.append(description)
            text_parts.append(description)

    return {
        "arm_group_labels": labels,
        "arm_group_types": types,
        "arm_group_descriptions": descriptions,
        "arms_text": " ".join(text_parts),
    }


def parse_outcomes(root, outcome_tag):
    measures = []
    time_frames = []
    descriptions = []
    text_parts = []

    for outcome in root.findall(f".//{outcome_tag}"):
        measure = clean_text(outcome.findtext("measure"))
        time_frame = clean_text(outcome.findtext("time_frame"))
        description = clean_text(outcome.findtext("description"))

        if measure:
            measures.append(measure)
            text_parts.append(measure)

        if time_frame:
            time_frames.append(time_frame)
            text_parts.append(time_frame)

        if description:
            descriptions.append(description)
            text_parts.append(description)

    return {
        "measures": measures,
        "time_frames": time_frames,
        "descriptions": descriptions,
        "text": " ".join(text_parts),
    }


def parse_references(root):
    pmids = []
    citations = []

    for ref in root.findall(".//reference"):
        citation = clean_text(ref.findtext("citation"))
        pmid = clean_text(ref.findtext("PMID"))

        if citation:
            citations.append(citation)

        if pmid:
            pmids.append(pmid)

    return {
        "reference_pmids": pmids,
        "reference_citations": " ".join(citations),
    }


# =============================================================================
# <clinical_results> parsing: raw XML, flattened plain text, compact summaries
# =============================================================================
# These fields are stored in _source (so they can be returned to the reader),
# but mapped with index=False, so they are never involved in BM25 scoring and
# never change retrieval behavior.

MAX_OUTCOME_CHARS = 1200
MAX_OUTCOMES_COMPACT_CHARS = 12000
MAX_SAFETY_COMPACT_CHARS = 6000
MAX_MEASUREMENTS_PER_MEASURE = 24
MAX_SAFETY_EVENTS_PER_SECTION = 16


def strip_namespace(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def format_attributes(node):
    parts = []
    for key, value in sorted(node.attrib.items()):
        value = clean_text(value)
        if value:
            parts.append(f"{strip_namespace(key)}={value}")
    return "; ".join(parts)


def flatten_xml_plain_text(root):
    """Flatten XML to readable path/attribute/text lines while preserving order."""
    lines = []

    def walk(node, parent_path):
        tag = strip_namespace(node.tag)
        path = f"{parent_path}/{tag}" if parent_path else tag
        attrs = format_attributes(node)
        text = clean_text(node.text)

        if text and attrs:
            lines.append(f"{path} [{attrs}]: {text}")
        elif text:
            lines.append(f"{path}: {text}")
        elif attrs:
            lines.append(f"{path} [{attrs}]")

        for child in list(node):
            walk(child, path)

    walk(root, "")
    return "\n".join(lines)


def truncate_text(value, max_chars):
    value = clean_text(value) or ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def child_text(node, path):
    return clean_text(node.findtext(path)) or ""


def compact_group_labels(container):
    groups = {}
    for group in container.findall("./group_list/group"):
        group_id = clean_text(group.attrib.get("group_id"))
        title = child_text(group, "title")
        if group_id:
            groups[group_id] = truncate_text(title or group_id, 120)
    return groups


def measurement_value(measurement):
    attrs = measurement.attrib
    value = clean_text(attrs.get("value"))
    if not value:
        return ""

    details = []
    spread = clean_text(attrs.get("spread"))
    lower = clean_text(attrs.get("lower_limit"))
    upper = clean_text(attrs.get("upper_limit"))
    if spread:
        details.append(f"spread={spread}")
    if lower or upper:
        details.append(f"interval={lower or '?'} to {upper or '?'}")

    rendered = value
    if details:
        rendered += f" ({'; '.join(details)})"
    return rendered


def compact_measure(measure):
    title = child_text(measure, "title")
    param = child_text(measure, "param")
    units = child_text(measure, "units")
    dispersion = child_text(measure, "dispersion")

    header_details = [part for part in (param, units, dispersion) if part]
    header = title or "Measure"
    if header_details:
        header += f" [{'; '.join(header_details)}]"

    values = []
    for class_node in measure.findall("./class_list/class"):
        class_title = child_text(class_node, "title")
        for category in class_node.findall("./category_list/category"):
            category_title = child_text(category, "title")
            value_label = " / ".join(
                part for part in (class_title, category_title) if part
            )
            for measurement in category.findall(".//measurement"):
                rendered_value = measurement_value(measurement)
                if not rendered_value:
                    continue
                group_id = clean_text(measurement.attrib.get("group_id")) or "group"
                prefix = f"{value_label}: " if value_label else ""
                values.append(f"{prefix}{group_id}={rendered_value}")
                if len(values) >= MAX_MEASUREMENTS_PER_MEASURE:
                    break
            if len(values) >= MAX_MEASUREMENTS_PER_MEASURE:
                break
        if len(values) >= MAX_MEASUREMENTS_PER_MEASURE:
            break

    # Some records omit class/category wrappers.
    if not values:
        for measurement in measure.findall(".//measurement"):
            rendered_value = measurement_value(measurement)
            if not rendered_value:
                continue
            group_id = clean_text(measurement.attrib.get("group_id")) or "group"
            values.append(f"{group_id}={rendered_value}")
            if len(values) >= MAX_MEASUREMENTS_PER_MEASURE:
                break

    return f"{header}: {', '.join(values)}" if values else header


def compact_analysis(analysis):
    group_ids = [
        clean_text(group_id.text)
        for group_id in analysis.findall("./group_id_list/group_id")
        if clean_text(group_id.text)
    ]
    fields = []
    for label, tag in (
        ("groups", None),
        ("parameter", "param_type"),
        ("estimate", "param_value"),
        ("p", "p_value"),
        ("p modifier", "p_value_modifier"),
        ("CI %", "ci_percent"),
        ("CI lower", "ci_lower_limit"),
        ("CI upper", "ci_upper_limit"),
        ("method", "method"),
    ):
        value = ",".join(group_ids) if tag is None else child_text(analysis, tag)
        if value:
            fields.append(f"{label}={value}")
    return "; ".join(fields)


def compact_outcome(outcome):
    outcome_type = child_text(outcome, "type") or "Outcome"
    title = child_text(outcome, "title") or "Untitled outcome"
    timeframe = child_text(outcome, "time_frame")
    groups = compact_group_labels(outcome)

    parts = [f"{outcome_type}: {title}"]
    if timeframe:
        parts.append(f"timeframe={truncate_text(timeframe, 180)}")
    if groups:
        parts.append(
            "groups=" + "; ".join(f"{key}:{value}" for key, value in groups.items())
        )

    for measure in outcome.findall("./measure"):
        parts.append(compact_measure(measure))

    analyses = [
        compact_analysis(analysis)
        for analysis in outcome.findall("./analysis_list/analysis")
    ]
    analyses = [analysis for analysis in analyses if analysis]
    if analyses:
        parts.append("analysis=" + " | ".join(analyses))

    return truncate_text("; ".join(parts), MAX_OUTCOME_CHARS)


def build_outcomes_compact(clinical_results):
    outcomes = clinical_results.findall("./outcome_list/outcome")
    primary = [
        outcome
        for outcome in outcomes
        if child_text(outcome, "type").lower() == "primary"
    ]
    remaining = [outcome for outcome in outcomes if outcome not in primary]

    lines = []
    for outcome in primary + remaining:
        line = compact_outcome(outcome)
        if not line:
            continue
        candidate = "\n".join(lines + [line])
        if len(candidate) > MAX_OUTCOMES_COMPACT_CHARS:
            break
        lines.append(line)
    return "\n".join(lines)


def compact_event_counts(event):
    rendered_counts = []
    max_affected = -1
    for counts in event.findall("./counts"):
        group_id = clean_text(counts.attrib.get("group_id")) or "group"
        affected = clean_text(counts.attrib.get("subjects_affected"))
        at_risk = clean_text(counts.attrib.get("subjects_at_risk"))
        events = clean_text(counts.attrib.get("events"))

        values = []
        if affected or at_risk:
            values.append(f"affected={affected or '?'}/{at_risk or '?'}")
        if events:
            values.append(f"events={events}")
        if values:
            rendered_counts.append(f"{group_id} {' '.join(values)}")
        try:
            max_affected = max(max_affected, int(affected))
        except (TypeError, ValueError):
            pass
    return "; ".join(rendered_counts), max_affected


def compact_event_section(reported_events, section_name):
    events = []
    for category in reported_events.findall(
        f"./{section_name}/category_list/category"
    ):
        category_title = child_text(category, "title")
        for event in category.findall("./event_list/event"):
            subtitle = child_text(event, "sub_title")
            counts, max_affected = compact_event_counts(event)
            if not counts:
                continue
            label = " - ".join(part for part in (category_title, subtitle) if part)
            events.append((label or "Event", counts, max_affected))

    totals = [event for event in events if event[0].lower().startswith("total")]
    details = [event for event in events if event not in totals]
    details.sort(key=lambda event: (-event[2], event[0].lower()))
    selected = totals + details[:MAX_SAFETY_EVENTS_PER_SECTION]
    return " | ".join(f"{label}: {counts}" for label, counts, _ in selected)


def build_safety_compact(clinical_results):
    reported_events = clinical_results.find("./reported_events")
    if reported_events is None:
        return ""

    parts = []
    timeframe = child_text(reported_events, "time_frame")
    groups = compact_group_labels(reported_events)
    if timeframe:
        parts.append(f"Safety timeframe: {truncate_text(timeframe, 180)}")
    if groups:
        parts.append(
            "Safety groups: "
            + "; ".join(f"{key}:{value}" for key, value in groups.items())
        )

    serious = compact_event_section(reported_events, "serious_events")
    other = compact_event_section(reported_events, "other_events")
    if serious:
        parts.append(f"Serious adverse events: {serious}")
    if other:
        parts.append(f"Other adverse events: {other}")
    return truncate_text("\n".join(parts), MAX_SAFETY_COMPACT_CHARS)


# =============================================================================
# Full document assembly
# =============================================================================

def parse_trial_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nct_id = find_text(root, ".//id_info/nct_id")

    if not nct_id:
        return None

    brief_title = find_text(root, ".//brief_title")
    official_title = find_text(root, ".//official_title")
    brief_summary = find_text(root, ".//brief_summary/textblock")
    detailed_description = find_text(root, ".//detailed_description/textblock")
    eligibility_text = find_text(root, ".//eligibility/criteria/textblock")

    overall_status = find_text(root, ".//overall_status")
    study_type = find_text(root, ".//study_type")
    phase = find_text(root, ".//phase")

    start_date_raw = find_text(root, ".//start_date")
    completion_date_raw = find_text(root, ".//completion_date")
    primary_completion_date_raw = find_text(root, ".//primary_completion_date")
    last_update_posted_raw = find_text(root, ".//last_update_posted")
    results_first_posted_raw = find_text(root, ".//results_first_posted")

    start_date = parse_date_to_iso(start_date_raw)
    completion_date = parse_date_to_iso(completion_date_raw)
    primary_completion_date = parse_date_to_iso(primary_completion_date_raw)
    last_update_posted = parse_date_to_iso(last_update_posted_raw)
    results_first_posted = parse_date_to_iso(results_first_posted_raw)

    conditions = find_all_text(root, ".//condition")
    keywords = find_all_text(root, ".//keyword")
    condition_mesh_terms = find_all_text(root, ".//condition_browse/mesh_term")
    intervention_mesh_terms = find_all_text(root, ".//intervention_browse/mesh_term")

    interventions = parse_interventions(root)
    arms = parse_arm_groups(root)

    primary_outcomes = parse_outcomes(root, "primary_outcome")
    secondary_outcomes = parse_outcomes(root, "secondary_outcome")

    # references = parse_references(root)

    # -----------------------------
    # <clinical_results> block: raw XML, flattened plain text, compact summaries.
    # Stored in _source (mapped index=False below), so it never affects BM25
    # ranking but can still be returned to the reader for grounding/QA.
    # -----------------------------
    clinical_results_el = root.find(".//clinical_results")
    has_clinical_results = clinical_results_el is not None

    if clinical_results_el is not None:
        clinical_results_raw_xml = ET.tostring(clinical_results_el, encoding="unicode")
        clinical_results_plain_text = flatten_xml_plain_text(clinical_results_el)
        clinical_results_outcomes_compact = build_outcomes_compact(clinical_results_el)
        clinical_results_safety_compact = build_safety_compact(clinical_results_el)
    else:
        clinical_results_raw_xml = ""
        clinical_results_plain_text = ""
        clinical_results_outcomes_compact = ""
        clinical_results_safety_compact = ""

    retrieval_text_parts = [
        brief_title,
        official_title,
        brief_summary,
        detailed_description,
        " ".join(conditions),
        " ".join(keywords),
        " ".join(condition_mesh_terms),
        " ".join(intervention_mesh_terms),
        interventions["interventions_text"],
        arms["arms_text"],
        primary_outcomes["text"],
        secondary_outcomes["text"],
        eligibility_text,
    ]

    retrieval_text = " ".join([x for x in retrieval_text_parts if x])

    doc = {
        "nct_id": nct_id,
        "org_study_id": find_text(root, ".//id_info/org_study_id"),
        "secondary_ids": find_all_text(root, ".//id_info/secondary_id"),
        "url": find_text(root, ".//required_header/url"),
        "xml_path": str(xml_path),

        "brief_title": brief_title,
        "official_title": official_title,
        "brief_summary": brief_summary,
        "detailed_description": detailed_description,

        "overall_status": overall_status,
        "study_type": study_type,
        "phase": phase,

        "start_date": start_date,
        "start_date_raw": start_date_raw,

        "completion_date": completion_date,
        "completion_date_raw": completion_date_raw,
        "completion_date_type": get_attr(root, ".//completion_date", "type"),

        "primary_completion_date": primary_completion_date,
        "primary_completion_date_raw": primary_completion_date_raw,
        "primary_completion_date_type": get_attr(root, ".//primary_completion_date", "type"),

        "last_update_posted": last_update_posted,
        "last_update_posted_raw": last_update_posted_raw,
        "last_update_posted_type": get_attr(root, ".//last_update_posted", "type"),

        "results_first_posted": results_first_posted,
        "results_first_posted_raw": results_first_posted_raw,
        "results_first_posted_type": get_attr(root, ".//results_first_posted", "type"),

        "has_clinical_results": has_clinical_results,

        "conditions": conditions,
        "keywords": keywords,
        "condition_mesh_terms": condition_mesh_terms,
        "eligibility_text": eligibility_text,

        "intervention_names": interventions["intervention_names"],
        "intervention_types": interventions["intervention_types"],
        "intervention_descriptions": interventions["intervention_descriptions"],
        "intervention_arm_labels": interventions["intervention_arm_labels"],
        "intervention_mesh_terms": intervention_mesh_terms,
        "interventions_text": interventions["interventions_text"],

        "arm_group_labels": arms["arm_group_labels"],
        "arm_group_types": arms["arm_group_types"],
        "arm_group_descriptions": arms["arm_group_descriptions"],
        "arms_text": arms["arms_text"],

        "primary_outcome_measures": primary_outcomes["measures"],
        "primary_outcome_time_frames": primary_outcomes["time_frames"],
        "primary_outcome_descriptions": primary_outcomes["descriptions"],
        "primary_outcomes_text": primary_outcomes["text"],

        "secondary_outcome_measures": secondary_outcomes["measures"],
        "secondary_outcome_time_frames": secondary_outcomes["time_frames"],
        "secondary_outcome_descriptions": secondary_outcomes["descriptions"],
        "secondary_outcomes_text": secondary_outcomes["text"],

        # "reference_pmids": references["reference_pmids"],
        # "reference_citations": references["reference_citations"],

        # Full <clinical_results> payload: kept out of BM25 scoring (index=False
        # in the mapping) but available in _source for display/grounding.
        "clinical_results_raw_xml": clinical_results_raw_xml,
        "clinical_results_plain_text": clinical_results_plain_text,
        "clinical_results_outcomes_compact": clinical_results_outcomes_compact,
        "clinical_results_safety_compact": clinical_results_safety_compact,
        "clinical_results_raw_xml_chars": len(clinical_results_raw_xml),
        "clinical_results_plain_text_chars": len(clinical_results_plain_text),

        "retrieval_text": retrieval_text,
        # "all_text": retrieval_text,
    }

    # Remove None values. Empty lists/strings are okay.
    return {k: v for k, v in doc.items() if v is not None}


# ---- 1000 documents limit, delete later -------------------
def generate_actions(xml_folder, limit=None):
    xml_folder = Path(xml_folder)
    count = 0

    for xml_path in xml_folder.rglob("*.xml"):
        if limit is not None and count >= limit:
            break

        try:
            doc = parse_trial_xml(xml_path)

            if not doc:
                print(f"Skipping file without NCT ID: {xml_path}")
                continue

            count += 1

            yield {
                "_index": index_name,
                "_id": doc["nct_id"],
                "_source": doc,
            }

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")

# -----------------------------------------------------------------


# -----------------------------
# 1. Connect to OpenSearch
# -----------------------------

host = os.getenv("OPENSEARCH_HOST", "localhost")
port = int(os.getenv("OPENSEARCH_PORT", "9200"))

user = os.getenv("OPENSEARCH_USER", "admin")

password = os.getenv("OPENSEARCH_PASSWORD")
if not password:
    raise RuntimeError("Missing OpenSearch password. Set the OPENSEARCH_PASSWORD environment variable.")

index_name = os.getenv("OPENSEARCH_INDEX_NAME", "clinical_trials_bm25_v1")
url_prefix = os.getenv("OPENSEARCH_URL_PREFIX", "")
use_ssl = os.getenv("OPENSEARCH_USE_SSL", "false").lower() in {"1", "true", "yes"}
verify_certs = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() in {"1", "true", "yes"}

client = OpenSearch(
    hosts=[{"host": host, "port": port}],
    http_auth=(user, password),
    use_ssl=use_ssl,
    url_prefix=url_prefix,
    verify_certs=verify_certs,
    ssl_assert_hostname=verify_certs,
    ssl_show_warn=verify_certs,
    timeout=120,
    max_retries=3,
    retry_on_timeout=True,
)


# -----------------------------
# 2. Helper field definitions
# -----------------------------

TEXT_FIELD = {
    "type": "text",
    "analyzer": "standard",
    "similarity": "BM25"
}

TEXT_WITH_KEYWORD = {
    "type": "text",
    "analyzer": "standard",
    "similarity": "BM25",
    "fields": {
        "keyword": {
            "type": "keyword",
            "ignore_above": 1024
        }
    }
}

# Stored in _source but never scored/searched. Used for the raw/plain/compact
# clinical_results fields, which exist for display and grounding, not ranking.
STORED_TEXT_NOT_SEARCHABLE = {
    "type": "text",
    "index": False,
}

STORED_INTEGER_NOT_SEARCHABLE = {
    "type": "integer",
    "index": False,
}


# -----------------------------
# 3. Index body
# -----------------------------

index_body = {
    "settings": {
        "index": {
            # Keep this low. You already had shard-limit issues before.
            "number_of_shards": 1,
            "number_of_replicas": 0,

            # Faster bulk indexing. Put it back to "1s" after indexing.
            "refresh_interval": "-1"
        }
    },

    "mappings": {
        # Strict is good while debugging because it catches parser mistakes.
        "dynamic": "strict",

        # These big artificial fields are searchable, but not stored in _source.
        # This saves space because retrieval_text/all_text duplicate many other fields.
        "_source": {
            "excludes": [
                "retrieval_text"
            ]
        },

        "properties": {
            # -----------------------------
            # Identifiers / metadata
            # -----------------------------
            "nct_id": {"type": "keyword"},
            "org_study_id": {"type": "keyword"},
            "secondary_ids": {"type": "keyword"},
            "url": {"type": "keyword"},
            "xml_path": {"type": "keyword"},

            # -----------------------------
            # Titles and descriptions
            # -----------------------------
            "brief_title": TEXT_WITH_KEYWORD,
            "official_title": TEXT_WITH_KEYWORD,
            "brief_summary": TEXT_FIELD,
            "detailed_description": TEXT_FIELD,

            # -----------------------------
            # Trial status / type / dates
            # -----------------------------
            "overall_status": {"type": "keyword"},
            "study_type": {"type": "keyword"},
            "phase": {"type": "keyword"},

            "start_date": {"type": "date"},
            "start_date_raw": {"type": "keyword"},

            "completion_date": {"type": "date"},
            "completion_date_raw": {"type": "keyword"},
            "completion_date_type": {"type": "keyword"},

            "primary_completion_date": {"type": "date"},
            "primary_completion_date_raw": {"type": "keyword"},
            "primary_completion_date_type": {"type": "keyword"},

            "last_update_posted": {"type": "date"},
            "last_update_posted_raw": {"type": "keyword"},
            "last_update_posted_type": {"type": "keyword"},

            "results_first_posted": {"type": "date"},
            "results_first_posted_raw": {"type": "keyword"},
            "results_first_posted_type": {"type": "keyword"},

            # Boolean telling you whether the XML has a clinical_results block.
            "has_clinical_results": {"type": "boolean"},

            # -----------------------------
            # Population / condition fields
            # -----------------------------
            "conditions": TEXT_WITH_KEYWORD,
            "keywords": TEXT_WITH_KEYWORD,
            "condition_mesh_terms": TEXT_WITH_KEYWORD,

            # Only the textblock from eligibility criteria.
            # No gender, min age, max age, healthy_volunteers, etc.
            "eligibility_text": TEXT_FIELD,

            # -----------------------------
            # Interventions
            # -----------------------------
            "intervention_names": TEXT_WITH_KEYWORD,
            "intervention_types": {"type": "keyword"},
            "intervention_descriptions": TEXT_FIELD,
            "intervention_arm_labels": TEXT_WITH_KEYWORD,
            "intervention_mesh_terms": TEXT_WITH_KEYWORD,
            "interventions_text": TEXT_FIELD,

            # -----------------------------
            # Arms / comparators
            # -----------------------------
            "arm_group_labels": TEXT_WITH_KEYWORD,
            "arm_group_types": {"type": "keyword"},
            "arm_group_descriptions": TEXT_FIELD,
            "arms_text": TEXT_FIELD,

            # -----------------------------
            # Outcomes
            # -----------------------------
            "primary_outcome_measures": TEXT_WITH_KEYWORD,
            "primary_outcome_time_frames": TEXT_FIELD,
            "primary_outcome_descriptions": TEXT_FIELD,
            "primary_outcomes_text": TEXT_FIELD,

            "secondary_outcome_measures": TEXT_WITH_KEYWORD,
            "secondary_outcome_time_frames": TEXT_FIELD,
            "secondary_outcome_descriptions": TEXT_FIELD,
            "secondary_outcomes_text": TEXT_FIELD,

            # -----------------------------
            # Optional publication/reference metadata
            # Not super important for retrieval, but useful for context/debugging.
            # -----------------------------
            # "reference_pmids": {"type": "keyword"},
            # "reference_citations": TEXT_FIELD,

            # -----------------------------
            # <clinical_results> raw / plain / compact payload.
            # Stored in _source for display and grounding, but index=False so
            # none of it is ever scored by BM25 or changes retrieval ranking.
            # -----------------------------
            "clinical_results_raw_xml": STORED_TEXT_NOT_SEARCHABLE,
            "clinical_results_plain_text": STORED_TEXT_NOT_SEARCHABLE,
            "clinical_results_outcomes_compact": STORED_TEXT_NOT_SEARCHABLE,
            "clinical_results_safety_compact": STORED_TEXT_NOT_SEARCHABLE,
            "clinical_results_raw_xml_chars": STORED_INTEGER_NOT_SEARCHABLE,
            "clinical_results_plain_text_chars": STORED_INTEGER_NOT_SEARCHABLE,

            # -----------------------------
            # Combined search fields
            # These are indexed/searchable but excluded from _source above.
            # -----------------------------
            "retrieval_text": TEXT_FIELD,
            # "all_text": TEXT_FIELD
        }
    }
}


# -----------------------------
# 4. Create index
# -----------------------------

if client.indices.exists(index=index_name):
    print(f"Index already exists: {index_name}")
else:
    response = client.indices.create(
        index=index_name,
        body=index_body
    )
    print(f"Created index: {index_name}")
    print(response)


test_file = next(XML_FOLDER.rglob("*.xml"))
test_doc = parse_trial_xml(test_file)

print("Example XML:", test_file)
print("NCT ID:", test_doc.get("nct_id"))
print("Title:", test_doc.get("brief_title"))
print("Has clinical results:", test_doc.get("has_clinical_results"))
print("Interventions:", test_doc.get("intervention_names"))
print("Conditions:", test_doc.get("conditions"))
print("Clinical results raw XML chars:", test_doc.get("clinical_results_raw_xml_chars"))
print("Clinical results plain text chars:", test_doc.get("clinical_results_plain_text_chars"))

success, errors = bulk(
    client,
    generate_actions(XML_FOLDER),
    chunk_size=500,
    request_timeout=120,
    raise_on_error=False,
)

print("Successfully indexed:", success)
print("Errors:", len(errors) if errors else 0)


client.indices.put_settings(
    index=index_name,
    body={
        "index": {
            "refresh_interval": "1s"
        }
    },
)

client.indices.refresh(index=index_name)

print(client.count(index=index_name))

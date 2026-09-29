import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import re
import tempfile
import shutil
from datetime import datetime


NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


def normalize_nct_id(value):
    if not value:
        return None

    match = NCT_RE.search(str(value))
    if not match:
        return None

    return match.group(0).upper()


def unique_preserve_order(values):
    seen = set()
    output = []

    for value in values:
        value = normalize_nct_id(value)
        if not value:
            continue

        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_retrieved_ncts_from_trace(trace_path, exclude_raw_queries=None):
    """
    Reads NCT IDs from the search trace.

    exclude_raw_queries: optional set of exact raw_query strings to skip.
    Used to exclude non-functional "planning" searches -- specifically the
    top-level planning search that generate_research_plan() runs once per
    review, on the raw unmodified review question, purely to help write the
    <=2 "search directions" text. Its results are never pooled into the
    dedup/rerank/context pipeline, so counting them toward retrieval
    recall/precision overstates what the funnel can actually act on. (There
    is a second, similar per-branch planning search inside plan_research(),
    but its raw_query text isn't reliably distinguishable from a real
    candidate-gathering search after the fact, so it isn't filtered here.)

    Returns:
      - unique NCT IDs in retrieval order
      - total trace rows with valid NCT IDs
    """
    if not trace_path.exists():
        return [], 0

    exclude_raw_queries = exclude_raw_queries or set()
    ncts = []
    total_rows_with_nct = 0

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("raw_query") in exclude_raw_queries:
                continue

            nct_id = normalize_nct_id(row.get("nct_id"))

            if nct_id:
                total_rows_with_nct += 1
                ncts.append(nct_id)

    return unique_preserve_order(ncts), total_rows_with_nct


def find_report_file(pmid_dir):
    """
    Finds the most likely generated report file.
    Important: do NOT parse run.log, because it may contain retrieved NCT IDs
    and would contaminate the reference extraction.
    """
    candidates = []

    for path in pmid_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.name in {"run.log", "open_search_trace.jsonl", "ground_truth.json", "metrics.json"}:
            continue

        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        candidates.append(path)

    if not candidates:
        return None

    # Use newest report-like file.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_reference_section(text):
    """
    Extracts text after a References heading.
    If no References heading is found, returns an empty string.
    This avoids counting NCT IDs mentioned in the body as references.
    """
    lines = text.splitlines()
    start = None

    for i, line in enumerate(lines):
        if re.match(r"^\s*(?:#{1,6}\s*)?references\s*:?\s*$", line, flags=re.IGNORECASE):
            start = i + 1
            break

    if start is None:
        return ""

    end = len(lines)

    for j in range(start, len(lines)):
        # Stop at next markdown heading.
        if re.match(r"^\s*#{1,6}\s+\S+", lines[j]):
            end = j
            break

    return "\n".join(lines[start:end])


def read_reference_ncts_from_report(report_path):
    """
    Extracts NCT IDs from the References section of the generated report.
    """
    if report_path is None or not report_path.exists():
        return []

    text = report_path.read_text(encoding="utf-8", errors="replace")
    references_text = extract_reference_section(text)

    if not references_text:
        return []

    return unique_preserve_order(NCT_RE.findall(references_text))


def compute_metrics(ground_truth_ncts, predicted_ncts):
    """
    Computes set metrics and R-precision.
    R = number of ground-truth NCT IDs.
    R-precision = relevant NCT IDs in top R predicted / R.
    """
    gt_ordered = unique_preserve_order(ground_truth_ncts)
    pred_ordered = unique_preserve_order(predicted_ncts)

    gt_set = set(gt_ordered)
    pred_set = set(pred_ordered)

    matched = sorted(gt_set & pred_set)

    gt_count = len(gt_set)
    pred_count = len(pred_set)
    matched_count = len(matched)

    precision = matched_count / pred_count if pred_count else 0.0
    recall = matched_count / gt_count if gt_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    R = gt_count
    top_r = pred_ordered[:R]
    r_precision = len(set(top_r) & gt_set) / R if R else 0.0

    return {
        "ground_truth_count": gt_count,
        "predicted_count": pred_count,
        "matched_count": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r_precision": r_precision,
        "matched_nct_ids": matched,
        "predicted_nct_ids": pred_ordered,
    }


def other_section_ncts(item, included_ncts):
    """
    Returns the NCT IDs from a review's excluded / awaiting_assessment /
    ongoing Cochrane sections, with anything already present in
    `included_ncts` removed first -- a trial can legitimately appear in
    more than one section for the same review (confirmed: ~1085 cases
    across the full ground-truth file), and those should only ever count
    toward the main included-set recall, not toward this one.

    Reviews loaded from a ground-truth file with no "sections" key (e.g.
    the older flat ground_truth.json) simply produce empty lists for all
    three -- this degrades gracefully rather than erroring.
    """
    sections = item.get("sections") or {}
    included_set = set(unique_preserve_order(included_ncts))

    def section_minus_included(key):
        raw = sections.get(key) or []
        return unique_preserve_order(
            nct for nct in raw if normalize_nct_id(nct) not in included_set
        )

    return {
        "excluded": section_minus_included("excluded_nct"),
        "awaiting_assessment": section_minus_included("awaiting_assessment_nct"),
        "ongoing": section_minus_included("ongoing_nct"),
    }


def compute_section_recall(target_ncts, predicted_ncts):
    """
    Recall-only metric against a non-included Cochrane section: what
    fraction of target_ncts (e.g. this review's "ongoing" trials) also
    appear in predicted_ncts (retrieved_nct_ids or references_nct_ids).

    Unlike compute_metrics(), there's no meaningful "precision" here --
    predicted_ncts is scored against the *included* set for precision
    already; this is purely "does the system also surface trials Cochrane
    did NOT include," which is a recall-shaped question by nature.
    """
    target_set = set(unique_preserve_order(target_ncts))
    pred_set = set(unique_preserve_order(predicted_ncts))
    matched = sorted(target_set & pred_set)
    count = len(target_set)

    return {
        "count": count,
        "matched_count": len(matched),
        "recall": (len(matched) / count) if count else 0.0,
        "matched_nct_ids": matched,
    }


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports either a list directly, or a dict containing a list.
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["reviews", "data", "items", "results"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Ground truth file must be a list of review objects, or a dict containing a list.")


def has_nct_ids(item):
    nct_ids = item.get("nct_ids", [])
    return isinstance(nct_ids, list) and len(nct_ids) > 0


def wait_for_search_api(port, timeout=60):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)

    return False

def build_query_retrieval_map(trace_path):
    grouped = {}
    order = []

    if not trace_path.exists():
        return []

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_query = row.get("raw_query") or row.get("search_query")
            if not raw_query:
                continue

            if raw_query not in grouped:
                grouped[raw_query] = {
                    "raw_query": raw_query,
                    "search_query": row.get("search_query"),
                    "loose_query": row.get("loose_query"),
                    "documents_all": [],
                    "documents_sent_to_gpt_researcher": [],
                }
                order.append(raw_query)

            doc = {
                "nct_id": row.get("nct_id"),
                "title": row.get("title"),
                "xml_path": row.get("xml_path"),
                "local_url": row.get("local_url"),
                "clinicaltrials_url": row.get("clinicaltrials_url"),
                "raw_rank": row.get("raw_rank"),
                "final_rank": row.get("final_rank"),
                "filter_reason": row.get("filter_reason"),
                "sent_to_gpt_researcher": row.get("sent_to_gpt_researcher"),
            }

            grouped[raw_query]["documents_all"].append(doc)

            if row.get("sent_to_gpt_researcher"):
                grouped[raw_query]["documents_sent_to_gpt_researcher"].append(doc)

    return [grouped[q] for q in order]


def start_search_api(
    search_api_path,
    port,
    trace_path,
    search_log_path=None,
):
    env = os.environ.copy()
    env["OPENSEARCH_TRACE_PATH"] = str(trace_path)

    search_log_file = None
    stdout_target = subprocess.DEVNULL

    if search_log_path is not None:
        search_log_file = open(search_log_path, "w", encoding="utf-8")
        search_log_file.write("COMMAND:\n")
        search_log_file.write(
            f"{sys.executable} {search_api_path} --port {port}\n\n"
        )
        search_log_file.write("SEARCH API ENVIRONMENT CONFIG:\n")
        search_log_file.write(f"OPENSEARCH_TRACE_PATH={env['OPENSEARCH_TRACE_PATH']}\n\n")
        search_log_file.write("OUTPUT:\n")
        search_log_file.flush()
        stdout_target = search_log_file

    process = subprocess.Popen(
        [
            sys.executable,
            str(search_api_path),
            "--port",
            str(port),
        ],
        env=env,
        stdout=stdout_target,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    process.search_log_file = search_log_file

    return process


def stop_process(process):
    if process.poll() is None:
        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    log_file = getattr(process, "search_log_file", None)
    if log_file is not None and not log_file.closed:
        log_file.flush()
        log_file.close()


def run_research_agent(
    research_agent_path,
    query,
    output_dir,
    pmid=None,
    title=None,
    skip_report=False,
    final_context_debug_dir=None,
    final_context_debug_label=None,
):
    cmd = [
        sys.executable,
        str(research_agent_path),
        "--query",
        query,
        "--output-dir",
        str(output_dir),
    ]

    if pmid:
        cmd.extend(["--pmid", str(pmid)])
    if title:
        cmd.extend(["--title", str(title)])

    if skip_report:
        cmd.append("--skip-report")

    env = os.environ.copy()
    if final_context_debug_dir is not None:
        env["FINAL_CONTEXT_DEBUG_DIR"] = str(final_context_debug_dir)
    if final_context_debug_label:
        env["FINAL_CONTEXT_DEBUG_LABEL"] = str(final_context_debug_label)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )


def average_metric(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def average_nested(rows, keys):
    """Same as average_metric, for a value nested under a tuple of dict keys,
    e.g. keys=("other_sections", "ongoing", "retrieved", "recall")."""
    values = []
    for row in rows:
        node = row
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if node is not None:
            values.append(node)
    if not values:
        return 0.0
    return sum(values) / len(values)


OTHER_SECTION_NAMES = ("excluded", "awaiting_assessment", "ongoing")


def _empty_other_section_summary():
    summary = {}
    for section in OTHER_SECTION_NAMES:
        summary[f"avg_{section}_retrieved_recall"] = 0.0
        summary[f"avg_{section}_references_recall"] = 0.0
    return summary


def build_run_summary(evaluation_rows):
    if not evaluation_rows:
        return {
            "query_count": 0,
            "ok_count": 0,
            "failed_count": 0,
            "avg_retrieved_precision": 0.0,
            "avg_retrieved_recall": 0.0,
            "avg_retrieved_f1": 0.0,
            "avg_retrieved_r_precision": 0.0,
            "avg_retrieved_actual_precision": 0.0,
            "avg_retrieved_actual_recall": 0.0,
            "avg_retrieved_actual_f1": 0.0,
            "avg_retrieved_actual_r_precision": 0.0,
            "avg_references_precision": 0.0,
            "avg_references_recall": 0.0,
            "avg_references_f1": 0.0,
            "avg_references_r_precision": 0.0,
            **_empty_other_section_summary(),
        }

    ok_count = sum(1 for row in evaluation_rows if row.get("status") == "ok")

    summary = {
        "query_count": len(evaluation_rows),
        "ok_count": ok_count,
        "failed_count": len(evaluation_rows) - ok_count,
        "avg_retrieved_precision": average_metric(evaluation_rows, "retrieved_precision"),
        "avg_retrieved_recall": average_metric(evaluation_rows, "retrieved_recall"),
        "avg_retrieved_f1": average_metric(evaluation_rows, "retrieved_f1"),
        "avg_retrieved_r_precision": average_metric(evaluation_rows, "retrieved_r_precision"),
        "avg_retrieved_actual_precision": average_metric(evaluation_rows, "retrieved_actual_precision"),
        "avg_retrieved_actual_recall": average_metric(evaluation_rows, "retrieved_actual_recall"),
        "avg_retrieved_actual_f1": average_metric(evaluation_rows, "retrieved_actual_f1"),
        "avg_retrieved_actual_r_precision": average_metric(evaluation_rows, "retrieved_actual_r_precision"),
        "avg_references_precision": average_metric(evaluation_rows, "references_precision"),
        "avg_references_recall": average_metric(evaluation_rows, "references_recall"),
        "avg_references_f1": average_metric(evaluation_rows, "references_f1"),
        "avg_references_r_precision": average_metric(evaluation_rows, "references_r_precision"),
    }

    for section in OTHER_SECTION_NAMES:
        summary[f"avg_{section}_retrieved_recall"] = average_nested(
            evaluation_rows, ("other_sections", section, "retrieved", "recall")
        )
        summary[f"avg_{section}_references_recall"] = average_nested(
            evaluation_rows, ("other_sections", section, "references", "recall")
        )

    return summary


def write_consolidated_results(output_path, metadata, evaluation_rows):
    payload = {
        "run_metadata": metadata,
        "summary": build_run_summary(evaluation_rows),
        "queries": evaluation_rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_file_stem(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:120].strip("._-") or "item"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON file.")
    parser.add_argument("--output-root", default="outputs/eval_runs", help="Folder where the consolidated run JSON will be written.")
    parser.add_argument("--output-name", default="evaluation_run.json", help="Filename for the single consolidated JSON output.")
    parser.add_argument("--research-agent", default="research_agent.py", help="Path to research_agent.py.")
    parser.add_argument("--search-api", default="search_api.py", help="Path to search_api.py.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of reviews to run.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index of the first review-with-NCTs to run, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="1-based index of the last review-with-NCTs to run, inclusive.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Run GPT Researcher retrieval only and skip final report generation.",
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth)
    output_root = Path(args.output_root)
    research_agent_path = Path(args.research_agent)
    search_api_path = Path(args.search_api)

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / args.output_name
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = output_root / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    query_generation_dir = output_root / "query_generation"
    query_maps_dir = output_root / "query_retrieval_maps"
    query_generation_dir.mkdir(parents=True, exist_ok=True)
    query_maps_dir.mkdir(parents=True, exist_ok=True)

    # whoosh_trace/ and research_artifacts/ are intentionally not persisted
    # (2026-08-19): the trace is still written to a temp file and used to
    # compute retrieval metrics + query_retrieval_maps/ below, same as
    # before -- it just no longer gets copied out permanently, since
    # query_retrieval_maps/ already carries everything anything downstream
    # reads from it. research_artifacts/ duplicated contexts/*.txt plus a
    # "costs" field that isn't real for a local vLLM model (see costs.py --
    # it prices every call at fixed OpenAI per-token rates regardless of
    # which model actually ran).
    contexts_dir = output_root / "contexts"
    final_context_debug_dir = output_root / "final_context_debug"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    final_context_debug_dir.mkdir(parents=True, exist_ok=True)

    items = load_ground_truth(ground_truth_path)
    all_items_with_ncts = [item for item in items if has_nct_ids(item)]

    if args.start_index < 1:
        raise ValueError("--start-index must be >= 1")

    total_items_with_ncts = len(all_items_with_ncts)
    end_index = args.end_index if args.end_index is not None else total_items_with_ncts

    if end_index < args.start_index:
        raise ValueError("--end-index must be >= --start-index")
    if args.start_index > total_items_with_ncts:
        raise ValueError(
            f"--start-index {args.start_index} is beyond the {total_items_with_ncts} reviews with NCT IDs"
        )

    end_index = min(end_index, total_items_with_ncts)
    items_with_ncts = all_items_with_ncts[args.start_index - 1 : end_index]

    if args.limit is not None:
        items_with_ncts = items_with_ncts[: args.limit]

    print(f"Total reviews in ground truth: {len(items)}")
    print(f"Reviews with NCT IDs: {total_items_with_ncts}")
    print(f"Running index range: {args.start_index}-{end_index}")
    print(f"Running: {len(items_with_ncts)}")

    evaluation_rows = []
    run_metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ground_truth_path": str(ground_truth_path),
        "research_agent_path": str(research_agent_path),
        "search_api_path": str(search_api_path),
        "output_path": str(output_path),
        "port": args.port,
        "limit": args.limit,
        "start_index": args.start_index,
        "end_index": end_index,
        "total_reviews_in_ground_truth": len(items),
        "reviews_with_nct_ids": total_items_with_ncts,
        "reviews_in_this_shard": len(items_with_ncts),
    }

    for i, item in enumerate(items_with_ncts, start=args.start_index):
        pmid = str(item.get("pmid") or f"no_pmid_{i}")
        title = item.get("title")
        query = item.get("research_query")

        if not query:
            print(f"\n[{i}/{len(items_with_ncts)}] Skipping PMID {pmid}: missing research_query")
            continue

        print("\n" + "=" * 80)
        print(f"[{i}/{len(items_with_ncts)}] PMID: {pmid}")
        print(f"Title: {title}")
        print(f"Query: {query}")
        file_stem = f"{i:03d}_{safe_file_stem(pmid)}"
        debug_run_id = f"{file_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        search_process = None
        returncode = None
        saved_report_path = reports_dir / f"{file_stem}_generated_report.md"
        saved_run_log_path = run_logs_dir / f"{file_stem}_run.log"
        saved_query_generation_path = query_generation_dir / f"{file_stem}_query_generation_trace.json"
        saved_query_map_path = query_maps_dir / f"{file_stem}_query_retrieval_map.json"
        saved_context_path = contexts_dir / f"{file_stem}_research_context.txt"
        saved_final_context_debug_dir = final_context_debug_dir / debug_run_id

        with tempfile.TemporaryDirectory(prefix=f"eval_{pmid}_", dir=output_root) as tmp_dir:
            tmp_path = Path(tmp_dir)
            trace_path = tmp_path / "open_search_trace.jsonl"
            run_log_path = tmp_path / "run.log"
            search_log_path = tmp_path / "search_api.log"
            temp_final_context_debug_dir = tmp_path / "final_context_debug"

            try:
                search_process = start_search_api(
                    search_api_path=search_api_path,
                    port=args.port,
                    trace_path=trace_path,
                    search_log_path=search_log_path,
                )

                ready = wait_for_search_api(port=args.port, timeout=60)

                if not ready:
                    print(f"Search API did not become ready on port {args.port}. Skipping PMID {pmid}.")
                    status = "search_api_failed"
                else:
                    result = run_research_agent(
                        research_agent_path=research_agent_path,
                        query=query,
                        output_dir=tmp_path,
                        pmid=pmid,
                        title=title,
                        skip_report=args.skip_report,
                        final_context_debug_dir=temp_final_context_debug_dir,
                        final_context_debug_label=debug_run_id,
                    )
                    returncode = result.returncode
                    status = "ok" if returncode == 0 else "research_agent_failed"

                    search_log_text = ""
                    if search_log_path.exists():
                        search_log_text = search_log_path.read_text(encoding="utf-8", errors="replace")

                    with open(run_log_path, "w", encoding="utf-8") as logf:
                        logf.write(f"PMID: {pmid}\n")
                        logf.write(f"Title: {title}\n")
                        logf.write(f"Query: {query}\n")
                        logf.write(f"Status: {status}\n")
                        logf.write(f"Return code: {returncode}\n")
                        logf.write("=" * 80 + "\n")
                        logf.write("SEARCH API LOG\n")
                        logf.write("=" * 80 + "\n")
                        logf.write(search_log_text)
                        logf.write("\n" + "=" * 80 + "\n")
                        logf.write("RESEARCH AGENT STDOUT\n")
                        logf.write("=" * 80 + "\n")
                        logf.write(result.stdout or "")
                        logf.write("\n" + "=" * 80 + "\n")
                        logf.write("RESEARCH AGENT STDERR\n")
                        logf.write("=" * 80 + "\n")
                        logf.write(result.stderr or "")

            finally:
                if search_process is not None:
                    stop_process(search_process)

            trace_lines = 0
            if trace_path.exists():
                with open(trace_path, "r", encoding="utf-8") as f:
                    trace_lines = sum(1 for _ in f)

            ground_truth_ncts = unique_preserve_order(item.get("nct_ids", []))
            retrieved_ncts, trace_rows_with_nct = read_retrieved_ncts_from_trace(trace_path)
            retrieved_ncts_actual, trace_rows_with_nct_actual = (
                read_retrieved_ncts_from_trace(trace_path, exclude_raw_queries={query})
            )

            report_path = find_report_file(tmp_path)
            reference_ncts = read_reference_ncts_from_report(report_path)

            if report_path is not None and report_path.exists():
                shutil.copy2(report_path, saved_report_path)

            if run_log_path.exists():
                shutil.copy2(run_log_path, saved_run_log_path)

            query_generation_path = tmp_path / "query_generation_trace.json"
            context_path = tmp_path / "research_context.txt"

            if context_path.exists():
                shutil.copy2(context_path, saved_context_path)

            if temp_final_context_debug_dir.exists():
                shutil.copytree(
                    temp_final_context_debug_dir,
                    saved_final_context_debug_dir,
                    dirs_exist_ok=True,
                )

            if query_generation_path.exists():
                shutil.copy2(query_generation_path, saved_query_generation_path)

            if trace_path.exists():
                write_json(saved_query_map_path, build_query_retrieval_map(trace_path))

        retrieved_metrics = compute_metrics(
            ground_truth_ncts=ground_truth_ncts,
            predicted_ncts=retrieved_ncts,
        )

        retrieved_metrics_actual = compute_metrics(
            ground_truth_ncts=ground_truth_ncts,
            predicted_ncts=retrieved_ncts_actual,
        )

        reference_metrics = compute_metrics(
            ground_truth_ncts=ground_truth_ncts,
            predicted_ncts=reference_ncts,
        )

        other_sections = other_section_ncts(item, ground_truth_ncts)
        other_sections_row = {}
        for section in OTHER_SECTION_NAMES:
            section_ncts = other_sections[section]
            other_sections_row[section] = {
                "nct_count": len(section_ncts),
                "nct_ids": section_ncts,
                "retrieved": compute_section_recall(section_ncts, retrieved_ncts),
                "references": compute_section_recall(section_ncts, reference_ncts),
            }

        metrics_row = {
            "pmid": pmid,
            "title": title,
            "query": query,
            "status": status,

            "ground_truth_nct_count": len(ground_truth_ncts),
            "ground_truth_nct_ids": ground_truth_ncts,

            "retrieved_unique_count": retrieved_metrics["predicted_count"],
            "retrieved_matched_count": retrieved_metrics["matched_count"],
            "retrieved_precision": retrieved_metrics["precision"],
            "retrieved_recall": retrieved_metrics["recall"],
            "retrieved_f1": retrieved_metrics["f1"],
            "retrieved_r_precision": retrieved_metrics["r_precision"],
            "retrieved_nct_ids": retrieved_metrics["predicted_nct_ids"],
            "retrieved_matched_nct_ids": retrieved_metrics["matched_nct_ids"],

            # Same as retrieved_* above, but excluding trace rows from the
            # top-level planning search (generate_research_plan(), run once
            # per review on the raw unmodified question, only to help write
            # search directions -- never pooled into the real candidate
            # pipeline). See read_retrieved_ncts_from_trace()'s docstring.
            "retrieved_actual_unique_count": retrieved_metrics_actual["predicted_count"],
            "retrieved_actual_matched_count": retrieved_metrics_actual["matched_count"],
            "retrieved_actual_precision": retrieved_metrics_actual["precision"],
            "retrieved_actual_recall": retrieved_metrics_actual["recall"],
            "retrieved_actual_f1": retrieved_metrics_actual["f1"],
            "retrieved_actual_r_precision": retrieved_metrics_actual["r_precision"],
            "retrieved_actual_nct_ids": retrieved_metrics_actual["predicted_nct_ids"],
            "retrieved_actual_matched_nct_ids": retrieved_metrics_actual["matched_nct_ids"],
            "planning_search_trace_rows_excluded": trace_rows_with_nct - trace_rows_with_nct_actual,

            "references_unique_count": reference_metrics["predicted_count"],
            "references_matched_count": reference_metrics["matched_count"],
            "references_precision": reference_metrics["precision"],
            "references_recall": reference_metrics["recall"],
            "references_f1": reference_metrics["f1"],
            "references_r_precision": reference_metrics["r_precision"],
            "references_nct_ids": reference_metrics["predicted_nct_ids"],
            "references_matched_nct_ids": reference_metrics["matched_nct_ids"],

            # Recall against NCTs Cochrane explicitly put in excluded /
            # awaiting_assessment / ongoing (never in the included set for
            # this review) -- see other_section_ncts()/compute_section_recall().
            # Only populated when the ground-truth file has a "sections" key
            # (e.g. ground_truth_correct_nct_conclusions_FINAL_v2.json);
            # degrades to zero counts otherwise.
            "other_sections": other_sections_row,

            "saved_report_path": str(saved_report_path) if saved_report_path.exists() else None,
            "saved_run_log_path": str(saved_run_log_path) if saved_run_log_path.exists() else None,
            "final_context_debug_dir": str(saved_final_context_debug_dir) if saved_final_context_debug_dir.exists() else None,
            "trace_lines": trace_lines,
            "trace_rows_with_nct": trace_rows_with_nct,
            "returncode": returncode,
        }

        evaluation_rows.append(metrics_row)
        write_consolidated_results(output_path, run_metadata, evaluation_rows)

        print(f"Status: {status}")
        print(f"Trace rows saved: {trace_lines}")
        print(f"Ground truth NCTs: {len(ground_truth_ncts)}")
        print(f"Retrieved unique NCTs: {retrieved_metrics['predicted_count']}")
        print(f"Retrieved matched NCTs: {retrieved_metrics['matched_count']}")
        print(f"Retrieved recall: {retrieved_metrics['recall']:.3f}")
        print(f"Retrieved R-precision: {retrieved_metrics['r_precision']:.3f}")
        print(
            f"Actual retrieval recall (excl. planning search): {retrieved_metrics_actual['recall']:.3f} "
            f"({trace_rows_with_nct - trace_rows_with_nct_actual} planning-search trace rows excluded)"
        )
        print(f"Reference unique NCTs: {reference_metrics['predicted_count']}")
        print(f"Reference matched NCTs: {reference_metrics['matched_count']}")
        print(f"Reference recall: {reference_metrics['recall']:.3f}")
        for section in OTHER_SECTION_NAMES:
            row = other_sections_row[section]
            if row["nct_count"]:
                print(
                    f"{section}: {row['nct_count']} NCTs | "
                    f"retrieved recall {row['retrieved']['recall']:.3f} | "
                    f"references recall {row['references']['recall']:.3f}"
                )

    print("\nDone.")
    print(f"Consolidated results saved to: {output_path}")


if __name__ == "__main__":
    main()

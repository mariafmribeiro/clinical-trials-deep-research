#!/usr/bin/env python3
"""Regenerate reports from saved trial blocks without generated learnings."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRIAL_BLOCK_RE = re.compile(
    r"^### Trial\s+\d+:\s*(NCT\d{8})\b.*?(?=^### Trial\s+\d+:\s*NCT\d{8}\b|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use saved pre-trim contexts to generate evidence-only reports. "
            "Retrieval, reranking, and learning extraction are not rerun."
        )
    )
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-glob", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--representation",
        choices=("evidence_only", "result_aware"),
        default="evidence_only",
        help="How saved trial blocks are labelled before report generation.",
    )
    parser.add_argument("--max-context-words", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_pmids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("pmids", []) if isinstance(payload, dict) else payload
    pmids = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("pmid")
        if value is not None and str(value).strip():
            pmids.append(str(value).strip())
    if not pmids:
        raise ValueError(f"No PMIDs found in {path}")
    return list(dict.fromkeys(pmids))


def latest_path(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No matching path was found")
    return max(paths, key=lambda path: path.stat().st_mtime)


def locate_saved_context(
    source_root: Path,
    source_glob: str,
    pmid: str,
) -> tuple[Path, Path]:
    contexts: list[Path] = []
    for run_dir in source_root.glob(source_glob):
        contexts.extend(
            run_dir.glob(
                f"final_context_debug/*_{pmid}_*/*_{pmid}_*_pre_trim_context.txt"
            )
        )

    context_path = latest_path(contexts)
    debug_files = list(context_path.parent.glob(f"*_{pmid}_*_final_context_debug.json"))
    debug_path = latest_path(debug_files)
    return context_path, debug_path


def extract_trial_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    seen: set[str] = set()

    for match in TRIAL_BLOCK_RE.finditer(text):
        nct_id = match.group(1).upper()
        if nct_id in seen:
            continue

        block = re.sub(r"\s*---\s*$", "", match.group(0).strip()).strip()
        if not block:
            continue

        seen.add(nct_id)
        blocks.append((nct_id, block))

    if not blocks:
        raise ValueError("No trial blocks were found in the saved context")
    return blocks


def make_result_aware(block: str) -> str:
    """Expose the registry boundary between planned and observed evidence."""
    has_outcome_results = bool(
        re.search(r"^Posted outcome results:\s*\S", block, re.IGNORECASE | re.MULTILINE)
    )
    has_safety_results = bool(
        re.search(r"^Posted safety results:\s*\S", block, re.IGNORECASE | re.MULTILINE)
    )

    transformed = re.sub(
        r"^Posted outcome results:",
        "OBSERVED POSTED OUTCOME RESULTS:",
        block,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    transformed = re.sub(
        r"^Posted safety results:",
        "OBSERVED POSTED SAFETY RESULTS:",
        transformed,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    transformed = re.sub(
        r"^Primary outcomes:",
        "PLANNED PRIMARY OUTCOMES (NOT OBSERVED RESULTS):",
        transformed,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    transformed = re.sub(
        r"^Secondary outcomes:",
        "PLANNED SECONDARY OUTCOMES (NOT OBSERVED RESULTS):",
        transformed,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    transformed = re.sub(
        r"^Brief summary:",
        "STUDY AIM/DESCRIPTION (NOT AN OBSERVED RESULT):",
        transformed,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    status = (
        "RESULT STATUS: "
        f"OUTCOME={'POSTED' if has_outcome_results else 'NOT POSTED'}; "
        f"SAFETY={'POSTED' if has_safety_results else 'NOT POSTED'}."
    )
    lines = transformed.splitlines()
    insert_at = 1 if lines else 0
    lines.insert(insert_at, status)
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(text.split())


def pack_blocks(
    blocks: list[tuple[str, str]],
    max_words: int,
) -> tuple[list[tuple[str, str]], int]:
    kept: list[tuple[str, str]] = []
    used = 0

    # Match the original packer: skip a non-fitting block and continue.
    for nct_id, block in blocks:
        words = word_count(block)
        if used + words > max_words:
            continue
        kept.append((nct_id, block))
        used += words

    return kept, used


def build_prompt(question: str, context: str, representation: str) -> str:
    # This is the report prompt used by the current final runs. Keeping it
    # unchanged isolates the effect of removing generated learnings.
    evidence_rules = ""
    if representation == "result_aware":
        evidence_rules = """
Apply these evidence-interpretation rules strictly:

- Only content under OBSERVED POSTED OUTCOME RESULTS may support a directional claim about effectiveness.
- Only content under OBSERVED POSTED SAFETY RESULTS may support a directional claim about safety.
- PLANNED OUTCOMES, study titles, and STUDY AIM/DESCRIPTION fields describe what a trial intended to investigate; they are not findings and must never be presented as observed effects.
- When the relevant result status is NOT POSTED, state only that the trial was registered or designed to evaluate the outcome. Do not say that it found, showed, demonstrated, improved, reduced, or increased anything.
- Preserve the group labels and numerical direction exactly. Do not call an event rate lower when its reported count or rate is higher, and do not claim statistical significance unless it is explicitly reported.
- Do not infer that an intervention belongs to the class named in the research question unless that relationship is explicit in the supplied record.
"""

    return f""" You are a clinical evidence synthesis assistant. Write balanced and cautious summaries using only the supplied ClinicalTrials.gov evidence.

Research question:

"{question}"

Retrieved context:

<context>
{context}
</context>

{evidence_rules}

Write a concise synthesis of the supplied evidence that directly answers the research question and integrates the findings across the relevant trial records.

Use three to five connected paragraphs without bullet points or internal subheadings. Begin by describing the body of directly relevant registry evidence. Then synthesise what this evidence indicates about the main comparisons, benefits, and harms addressed by the research question. Focus on the overall pattern of evidence rather than describing each trial separately. The final paragraph should provide a brief and balanced answer to the research question, reflecting the completeness and limitations of the available evidence and avoiding clinical recommendations.

Use numerical findings only when they are explicitly reported in the supplied records. Clearly distinguish posted results from planned outcomes and study-design information. When the registry evidence is limited, reflect that limitation in the strength and wording of the conclusion.

Ignore records that are clearly unrelated to the research question. Support the main findings with grouped NCT citations, citing only the records used in the synthesis.

Use exactly these sections:

## Summary

## References

In the References section, list each cited record once:

- NCT ID - Brief title
"""


def call_vllm(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer dummy_key",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))

    content = result["choices"][0]["message"].get("content") or ""
    content = THINK_RE.sub("", content).strip()
    if not content:
        raise ValueError("The model returned an empty report")
    return content, result


def request_with_retries(args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            return call_vllm(
                base_url=args.base_url,
                model=args.served_model_name,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
            )
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
            last_error = error
            if attempt < args.max_attempts:
                print(f"  attempt {attempt} failed: {error}; retrying...", flush=True)
                time.sleep(5 * attempt)
    raise RuntimeError(f"No valid report after {args.max_attempts} attempts: {last_error}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    pmids = load_pmids(args.selection)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.output_dir / "reports"
    contexts_dir = args.output_dir / "contexts"
    metadata_dir = args.output_dir / "metadata"
    raw_dir = args.output_dir / "raw_responses"
    for directory in (reports_dir, contexts_dir, metadata_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolved: list[dict[str, Any]] = []
    for pmid in pmids:
        context_path, debug_path = locate_saved_context(
            args.source_root, args.source_glob, pmid
        )
        debug = json.loads(debug_path.read_text(encoding="utf-8"))
        question = str(debug.get("root_query") or "").strip()
        if not question:
            raise ValueError(f"No root_query found for PMID {pmid} in {debug_path}")

        blocks = extract_trial_blocks(context_path.read_text(encoding="utf-8"))
        if args.representation == "result_aware":
            blocks = [(nct_id, make_result_aware(block)) for nct_id, block in blocks]
        kept, used_words = pack_blocks(blocks, args.max_context_words)
        if not kept:
            raise ValueError(f"No trial block fits the context budget for PMID {pmid}")

        resolved.append(
            {
                "pmid": pmid,
                "question": question,
                "source_context": str(context_path.resolve()),
                "source_debug": str(debug_path.resolve()),
                "available_trial_blocks": len(blocks),
                "kept_trial_blocks": len(kept),
                "context_words": used_words,
                "kept_nct_ids": [nct_id for nct_id, _ in kept],
                "context": "\n\n".join(block for _, block in kept),
            }
        )

    print(f"Model: {args.model_label}", flush=True)
    print(f"Selected PMIDs: {len(resolved)}", flush=True)
    print(f"Source pattern: {args.source_root / args.source_glob}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    for item in resolved:
        print(
            f"  PMID {item['pmid']}: {item['kept_trial_blocks']}/"
            f"{item['available_trial_blocks']} blocks, {item['context_words']} words",
            flush=True,
        )

    if args.dry_run:
        print("Dry run complete; no model requests were sent.", flush=True)
        return 0

    manifest_path = args.output_dir / "manifest.jsonl"
    successes = 0
    failures = 0

    for index, item in enumerate(resolved, start=1):
        pmid = item["pmid"]
        report_path = reports_dir / f"{pmid}_generated_report.md"
        context_out = contexts_dir / f"{pmid}_evidence_only_context.txt"
        metadata_path = metadata_dir / f"{pmid}_metadata.json"
        raw_path = raw_dir / f"{pmid}_response.json"

        if report_path.exists() and not args.overwrite:
            print(f"[{index}/{len(resolved)}] PMID {pmid}: already complete", flush=True)
            successes += 1
            continue

        print(f"[{index}/{len(resolved)}] PMID {pmid}", flush=True)
        context_out.write_text(item["context"] + "\n", encoding="utf-8")
        prompt = build_prompt(item["question"], item["context"], args.representation)

        try:
            report, raw_response = request_with_retries(args, prompt)
            report_path.write_text(report + "\n", encoding="utf-8")
            write_json(raw_path, raw_response)
            metadata = {
                key: value for key, value in item.items() if key != "context"
            }
            metadata.update(
                {
                    "condition": (
                        "result_aware_trial_blocks_without_learnings"
                        if args.representation == "result_aware"
                        else "trial_blocks_without_learnings"
                    ),
                    "representation": args.representation,
                    "model_label": args.model_label,
                    "served_model_name": args.served_model_name,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "max_context_words": args.max_context_words,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "report_path": str(report_path.resolve()),
                    "context_path": str(context_out.resolve()),
                }
            )
            write_json(metadata_path, metadata)
            with manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            successes += 1
        except Exception as error:  # Continue so one review does not end the pilot.
            failures += 1
            error_path = metadata_dir / f"{pmid}_error.json"
            write_json(
                error_path,
                {
                    "pmid": pmid,
                    "model_label": args.model_label,
                    "error": str(error),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"  ERROR: {error}", flush=True)

    print(f"Finished with {successes} successes and {failures} failures.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate reports from fixed, previously prepared result-aware contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_evidence_only_reports import (
    build_prompt,
    load_pmids,
    request_with_retries,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reports with a new model while keeping the prepared "
            "result-aware Qwen contexts fixed."
        )
    )
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_fixed_inputs(
    prepared_root: Path,
    pmids: list[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    missing: list[str] = []

    for pmid in pmids:
        context_path = prepared_root / "contexts" / f"{pmid}_evidence_only_context.txt"
        metadata_path = prepared_root / "metadata" / f"{pmid}_metadata.json"
        if not context_path.is_file() or not metadata_path.is_file():
            missing.append(pmid)
            continue

        context = context_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("representation") != "result_aware":
            raise ValueError(
                f"PMID {pmid} is not a result-aware context: {metadata_path}"
            )

        question = str(metadata.get("question") or "").strip()
        if not question:
            raise ValueError(f"No research question found for PMID {pmid}")

        items.append(
            {
                "pmid": pmid,
                "question": question,
                "context": context,
                "source_context": str(context_path.resolve()),
                "source_metadata": str(metadata_path.resolve()),
                "source_model_label": metadata.get("model_label"),
                "source_context_sha256": sha256_text(context),
                "available_trial_blocks": metadata.get("available_trial_blocks"),
                "kept_trial_blocks": metadata.get("kept_trial_blocks"),
                "context_words": metadata.get("context_words"),
                "kept_nct_ids": metadata.get("kept_nct_ids", []),
            }
        )

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing prepared context or metadata for {len(missing)} PMIDs: "
            f"{preview}{suffix}"
        )

    return items


def main() -> int:
    args = parse_args()
    pmids = load_pmids(args.selection)
    items = load_fixed_inputs(args.prepared_root, pmids)

    reports_dir = args.output_dir / "reports"
    contexts_dir = args.output_dir / "contexts"
    metadata_dir = args.output_dir / "metadata"
    raw_dir = args.output_dir / "raw_responses"
    for directory in (reports_dir, contexts_dir, metadata_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_label}", flush=True)
    print(f"Selected PMIDs: {len(items)}", flush=True)
    print(f"Fixed context source: {args.prepared_root.resolve()}", flush=True)
    print(f"Output: {args.output_dir.resolve()}", flush=True)
    print("Representation: result-aware trial blocks without learnings", flush=True)

    for item in items:
        print(
            f"  PMID {item['pmid']}: {item['kept_trial_blocks']} blocks, "
            f"{item['context_words']} words, sha256={item['source_context_sha256'][:12]}",
            flush=True,
        )

    if args.dry_run:
        print("Dry run complete; no model requests were sent.", flush=True)
        return 0

    manifest_path = args.output_dir / "manifest.jsonl"
    successes = 0
    failures = 0

    for index, item in enumerate(items, start=1):
        pmid = item["pmid"]
        report_path = reports_dir / f"{pmid}_generated_report.md"
        context_path = contexts_dir / f"{pmid}_evidence_only_context.txt"
        metadata_path = metadata_dir / f"{pmid}_metadata.json"
        raw_path = raw_dir / f"{pmid}_response.json"

        if report_path.exists() and not args.overwrite:
            print(f"[{index}/{len(items)}] PMID {pmid}: already complete", flush=True)
            successes += 1
            continue

        print(f"[{index}/{len(items)}] PMID {pmid}", flush=True)
        context_path.write_text(item["context"], encoding="utf-8")
        prompt = build_prompt(item["question"], item["context"], "result_aware")

        try:
            report, raw_response = request_with_retries(args, prompt)
            report_path.write_text(report + "\n", encoding="utf-8")
            write_json(raw_path, raw_response)

            metadata = {key: value for key, value in item.items() if key != "context"}
            metadata.update(
                {
                    "condition": "result_aware_without_learnings_fixed_qwen_context",
                    "representation": "result_aware",
                    "learnings_included": False,
                    "model_label": args.model_label,
                    "served_model_name": args.served_model_name,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "report_path": str(report_path.resolve()),
                    "context_path": str(context_path.resolve()),
                    "context_sha256": sha256_text(item["context"]),
                }
            )
            write_json(metadata_path, metadata)
            with manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            successes += 1
        except Exception as error:  # Keep the remaining reviews running.
            failures += 1
            write_json(
                metadata_dir / f"{pmid}_error.json",
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

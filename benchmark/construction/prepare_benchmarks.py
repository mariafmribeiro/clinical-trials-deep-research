#!/usr/bin/env python3
"""Create the complete and higher-confidence benchmark files."""

import argparse
import json
from pathlib import Path


PUBLIC_FIELDS = (
    "title",
    "pmid",
    "url",
    "rct_count",
    "research_query",
    "nct_ids",
    "doi",
    "sections",
)


def public_record(record):
    """Keep only fields required to reproduce the benchmark and experiments."""
    return {field: record.get(field) for field in PUBLIC_FIELDS}


def prepare_records(records):
    complete = [
        public_record(record) for record in records if record.get("nct_ids")
    ]
    higher_confidence = [
        record
        for record in complete
        if record.get("rct_count") == len(record["nct_ids"])
    ]
    return complete, higher_confidence


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the complete mapped benchmark and its higher-confidence "
            "51-review subset."
        )
    )
    parser.add_argument("input", type=Path, help="Complete extraction JSON file")
    parser.add_argument(
        "--complete-output",
        type=Path,
        default=Path("benchmark/benchmark_complete.json"),
    )
    parser.add_argument(
        "--higher-confidence-output",
        type=Path,
        default=Path("benchmark/benchmark.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open(encoding="utf-8") as input_file:
        records = json.load(input_file)

    complete, higher_confidence = prepare_records(records)
    write_json(args.complete_output, complete)
    write_json(args.higher_confidence_output, higher_confidence)

    print(
        f"Wrote {len(complete)} complete and "
        f"{len(higher_confidence)} higher-confidence records."
    )


if __name__ == "__main__":
    main()

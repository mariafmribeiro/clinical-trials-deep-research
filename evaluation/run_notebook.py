#!/usr/bin/env python3
"""Execute notebook code cells without modifying the notebook file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "gpt_researcher").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the repository root.")


def execute_notebook(path: Path) -> None:
    path = path.resolve()
    root = repository_root(path.parent)
    with path.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    namespace = {
        "__name__": "__notebook__",
        "__file__": str(path),
    }
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        os.chdir(root)
        sys.path.insert(0, str(path.parent))
        sys.path.insert(0, str(root))

        code_cells = [
            cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
        ]
        for number, cell in enumerate(code_cells, start=1):
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            print(f"[{path.name}] cell {number}/{len(code_cells)}", flush=True)
            try:
                exec(compile(source, f"{path}:cell-{number}", "exec"), namespace)
            except Exception:
                print(f"Failed in {path.name}, code cell {number}.", file=sys.stderr)
                raise
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for notebook in args.notebooks:
        execute_notebook(notebook)


if __name__ == "__main__":
    main()

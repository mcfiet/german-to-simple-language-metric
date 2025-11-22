#!/usr/bin/env python3
"""
Input CSV format:
    original,leicht

Jede Zeile enthält derzeit einen einfachen Satz („leicht“) pro normalem Satz („Original“).
Dieses Skript gruppiert die Zeilen nach dem normalen Satz und schreibt eine JSONL-Datei, in der jeder
Eintrag alle eindeutigen einfachen Sätze enthält, die diesem normalen Satz zugeordnet sind.
Das vorangegangene Problem war nämlich, dass ein deutscher Satz als mehrfache Samples auftauchte und immer einen anderen Simple German Satz zugeteilt bekommen hat. 
Jetzt sind alle Simple German Übersetzungen einem normal deutschem Satz zugeordnet.

Default inputs: data/output/train.csv, val.csv, test.csv (if present).
Outputs: data/output/<name>_grouped.jsonl

Usage:
    python data/group_output_by_normal.py
    python data/group_output_by_normal.py --input data/output/train.csv data/output/val.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DATA_DIR = Path(__file__).resolve().parent / "output"


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def group_file(path: Path, output_path: Path) -> Tuple[int, int]:
    buckets: Dict[str, List[str]] = {}

    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "original" not in reader.fieldnames or "leicht" not in reader.fieldnames:
            raise ValueError(f"Missing required columns in {path}: {reader.fieldnames}")

        for row in reader:
            normal = (row.get("original") or "").strip()
            simple = (row.get("leicht") or "").strip()
            if not normal or not simple:
                continue
            buckets.setdefault(normal, []).append(simple)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for normal, simples in buckets.items():
            record = {
                "original": normal,
                "simple_sentences": dedupe_preserve_order(simples),
            }
            json.dump(record, out_f, ensure_ascii=False)
            out_f.write("\n")

    return len(buckets), sum(len(v) for v in buckets.values())


def default_inputs() -> List[Path]:
    candidates = [DATA_DIR / name for name in ("train.csv", "val.csv", "test.csv")]
    return [p for p in candidates if p.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group simple sentences per normal sentence for CSVs in data/output/"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=default_inputs(),
        help="Input CSV files (default: data/output/train.csv val.csv test.csv if they exist)",
    )
    parser.add_argument(
        "--suffix",
        default="_grouped.jsonl",
        help="Suffix for grouped output files (default: _grouped.jsonl)",
    )
    args = parser.parse_args()

    if not args.input:
        raise SystemExit("No input files found. Specify with --input.")

    for in_path in args.input:
        if not in_path.exists():
            print(f"Skip missing file: {in_path}")
            continue
        out_path = in_path.with_name(in_path.stem + args.suffix)
        grouped_count, total_simples = group_file(in_path, out_path)
        print(
            f"{in_path.name} -> {out_path.name}: "
            f"{grouped_count} grouped normals from {total_simples} simple rows"
        )


if __name__ == "__main__":
    main()

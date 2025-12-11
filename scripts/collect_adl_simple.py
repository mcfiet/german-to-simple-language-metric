"""
Collect all sentences from the 'simple' column in ADL train/val/test CSVs.

Writes one sentence per line to the specified output file.
"""

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def collect_simple(csv_paths: Iterable[Path]) -> List[str]:
    sentences: List[str] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "leicht" not in df.columns:
            raise ValueError(f"'leicht' column not found in {path}")
        sentences.extend(
            str(val).strip() for val in df["leicht"].fillna("") if str(val).strip()
        )
    return sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect sentences from the 'leicht' column in ADL CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/adl"),
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/adl/simple_sentences.txt"),
        help="Output text file (one sentence per line).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_paths = [
        args.input_dir / "train.csv",
        args.input_dir / "val.csv",
        args.input_dir / "test.csv",
    ]
    missing = [p for p in csv_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSVs: {', '.join(map(str, missing))}")

    sentences = collect_simple(csv_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sentences), encoding="utf-8")
    print(f"[done] wrote {len(sentences)} sentences to {args.output}")


if __name__ == "__main__":
    main()

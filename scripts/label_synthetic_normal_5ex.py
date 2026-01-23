"""
Label synthetic simple/normal sentence pairs with multiple normal variants.

Reads a TSV with two columns:
1) simple sentence
2) JSON list of normal/complex sentence variants

Emits a CSV with one sentence per row and a label column:
1 for simple, 0 for not simple.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten a simple/normal TSV with variants into sentence,label CSV rows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic_normal_ollama_5ex.tsv"),
        help="Input TSV with two columns: simple<TAB>json_list_of_normal.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_normal_ollama_5ex_labeled.csv"),
        help="Output CSV with columns: sentence,label.",
    )
    parser.add_argument(
        "--simple-label",
        type=int,
        default=1,
        help="Label value to assign to simple sentences.",
    )
    parser.add_argument(
        "--normal-label",
        type=int,
        default=0,
        help="Label value to assign to normal/complex sentences.",
    )
    return parser.parse_args()


def strip_prefix(text: str) -> str:
    """Remove leading markers like 'Normal:' or 'Einfach:' if present."""
    lowered = text.lower()
    for prefix in ("normal:", "einfach:", "einfacher text:", "einfacher:", "einfach:"):
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
    return text.strip()


def parse_variants(raw: str, path: Path, line_no: int) -> List[str]:
    try:
        variants = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Expected JSON list in {path} at line {line_no}, got: {raw!r}"
        ) from exc
    if not isinstance(variants, list):
        raise ValueError(
            f"Expected JSON list in {path} at line {line_no}, got: {type(variants).__name__}"
        )
    return [strip_prefix(str(item)) for item in variants if str(item).strip()]


def read_pairs(path: Path) -> List[Tuple[str, List[str]]]:
    pairs: List[Tuple[str, List[str]]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(
                f"Expected 2 columns separated by TAB in {path} at line {line_no}, got {len(parts)}"
            )
        simple_raw, normals_raw = parts
        simple = strip_prefix(simple_raw)
        normals = parse_variants(normals_raw, path, line_no)
        pairs.append((simple, normals))
    return pairs


def flatten(
    pairs: Iterable[Tuple[str, List[str]]], simple_label: int, normal_label: int
):
    for simple, normals in pairs:
        if simple:
            yield simple, simple_label
        for normal in normals:
            if normal:
                yield normal, normal_label


def write_labeled(rows: Iterable[Tuple[str, int]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sentence", "label"])
        writer.writerows(rows)


def main():
    args = parse_args()
    pairs = read_pairs(args.input)
    labeled_rows = list(flatten(pairs, args.simple_label, args.normal_label))
    write_labeled(labeled_rows, args.output)
    print(f"[done] wrote {len(labeled_rows)} labeled sentences to {args.output}")


if __name__ == "__main__":
    main()

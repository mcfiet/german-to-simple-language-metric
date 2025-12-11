"""
Remove duplicate sentences from a text file (one sentence per line).
Writes a new file with first occurrence preserved.
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Set


def dedupe(lines: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    unique: List[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove duplicate sentences from a text file.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input text file (one sentence per line).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file for deduplicated sentences.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lines = args.input.read_text(encoding="utf-8").splitlines()
    unique = dedupe(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(unique), encoding="utf-8")
    print(f"[done] removed {len(lines) - len(unique)} duplicates; wrote {len(unique)} sentences to {args.output}")


if __name__ == "__main__":
    main()

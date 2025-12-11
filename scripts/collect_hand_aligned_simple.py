"""
Concatenate all *.simple files from the hand_aligned folder into one text file.

Each input file is appended verbatim; the output contains one line per sentence
as stored in the source files.
"""

import argparse
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_ALIGNED = REPO_ROOT / "Simple-German-Corpus" / "results" / "hand_aligned"


def find_simple_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.simple"))


def collect(files: List[Path]) -> str:
    parts: List[str] = []
    for path in files:
        parts.append(path.read_text(encoding="utf-8").strip())
    return "\n".join(p for p in parts if p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect all *.simple files in hand_aligned into one text file."
    )
    parser.add_argument(
        "--hand-aligned-dir",
        type=Path,
        default=DEFAULT_HAND_ALIGNED,
        help="Directory containing hand_aligned data (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hand_aligned_simple.txt"),
        help="Output text file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = find_simple_files(args.hand_aligned_dir)
    if not files:
        print(f"[warn] no .simple files found under {args.hand_aligned_dir}")
        return
    combined = collect(files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(combined, encoding="utf-8")
    print(f"[done] wrote {len(files)} files into {args.output}")


if __name__ == "__main__":
    main()

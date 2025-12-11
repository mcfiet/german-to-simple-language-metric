"""
Collect all easy-language URLs from the Simple-German-Corpus Datasets.

This scans each domain folder for `header.json` entries marked with `"easy": true`
and writes the URLs to a text file (one per line).
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIR = REPO_ROOT / "Simple-German-Corpus" / "Datasets"


def dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def collect_easy_urls(datasets_dir: Path) -> List[str]:
    urls: List[str] = []
    for domain_dir in sorted(datasets_dir.iterdir()):
        header_path = domain_dir / "header.json"
        if not header_path.exists():
            continue
        try:
            with header_path.open("r", encoding="utf-8") as fp:
                header = json.load(fp)
        except Exception as exc:
            print(f"[warn] failed to read {header_path}: {exc}")
            continue
        for entry in header.values():
            if entry.get("easy"):
                urls.append(entry.get("url", ""))
    return [u for u in urls if u]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List all easy-language URLs from SGC Datasets headers."
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help="Path to Simple-German-Corpus/Datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/simple_urls.txt"),
        help="Where to write the URLs (one per line).",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate URLs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    urls = collect_easy_urls(args.datasets_dir)
    if not args.no_dedupe:
        urls = dedupe(urls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(urls), encoding="utf-8")
    print(f"[done] wrote {len(urls)} URLs to {args.output}")


if __name__ == "__main__":
    main()

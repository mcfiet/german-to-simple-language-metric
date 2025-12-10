"""
Run the Simple-German-Corpus crawlers/parsers and collect sentences.

This script reuses the cloned `Simple-German-Corpus` project in the repo root.
It will:
1) crawl each site module (unless --skip-crawl is set),
2) parse the downloaded HTML into sentence files using each site's parser,
3) aggregate all parsed sentences into a single output file.

Example:
    python scripts/run_simple_corpus_pipeline.py --output data/simple_german_sentences.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "Simple-German-Corpus"
sys.path.insert(0, str(CORPUS_ROOT))

import crawler  # type: ignore
import crawler.utilities as utl  # type: ignore
import defaultvalues  # type: ignore


def dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def run_crawlers(module_names: Sequence[str], skip_crawl: bool, delay: float):
    """Run each crawler to download pages and populate headers."""
    import inspect

    for name in module_names:
        module = getattr(crawler, name)
        if skip_crawl or not hasattr(module, "crawling"):
            print(f"[skip] crawl {name}")
            continue
        print(f"[crawl] {name} ({module.base_url})")
        try:
            sig = inspect.signature(module.crawling)
            if len(sig.parameters) >= 1:
                module.crawling(module.base_url)
            else:
                module.crawling()
        except Exception as exc:
            print(f"[warn] crawl failed for {name}: {exc}")
            continue
        if delay:
            import time

            time.sleep(delay)


def run_parsers(module_names: Sequence[str]):
    """Parse downloaded pages into sentence files."""
    for name in module_names:
        module = getattr(crawler, name)
        if not hasattr(module, "parser"):
            print(f"[skip] parse {name} (no parser)")
            continue
        print(f"[parse] {name}")
        utl.parse_soups(module.base_url, module.parser)


def collect_sentences(module_names: Sequence[str]) -> List[str]:
    sentences: List[str] = []
    for name in module_names:
        module = getattr(crawler, name)
        folder, _ = utl.get_names_from_url(module.base_url)
        parsed_dir = Path(utl.dataset_location) / folder / "parsed"
        if not parsed_dir.exists():
            print(f"[warn] no parsed dir for {name}: {parsed_dir}")
            continue
        for path in sorted(parsed_dir.glob("*.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    sentences.append(line)
    return dedupe(sentences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the Simple-German-Corpus crawlers to build a sentence dataset."
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=None,
        help="Subset of crawler module names to run (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/simple_german_sentences.txt"),
        help="Where to write aggregated sentences.",
    )
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip downloading; only parse existing crawled data.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between crawler runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure paths inside the corpus point to the cloned repo.
    defaultvalues.repository_location = str(CORPUS_ROOT)
    defaultvalues.dataset_location = str(CORPUS_ROOT / "Datasets")
    defaultvalues.results_location = str(CORPUS_ROOT / "results")
    # Keep utilities in sync with the updated locations.
    utl.dataset_location = defaultvalues.dataset_location
    utl.results_location = defaultvalues.results_location

    module_names = args.modules if args.modules else list(crawler.__all__)

    run_crawlers(module_names, skip_crawl=args.skip_crawl, delay=args.delay)
    run_parsers(module_names)

    sentences = collect_sentences(module_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sentences), encoding="utf-8")
    print(f"[done] wrote {len(sentences)} sentences to {args.output}")


if __name__ == "__main__":
    main()

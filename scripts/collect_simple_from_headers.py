import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "Simple-German-Corpus"
DEFAULT_DATASETS_DIR = CORPUS_ROOT / "Datasets"

sys.path.insert(0, str(CORPUS_ROOT))
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


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_from_html(path: Path) -> List[str]:
    """Fallback extraction from raw HTML if no parsed file exists."""
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return []
    soup = BeautifulSoup(html, "html.parser")
    texts = [t.get_text(" ", strip=True) for t in soup.find_all(["p", "li", "h1", "h2", "h3"])]
    return split_sentences(" ".join(texts))


def collect_easy_from_header(header_path: Path, prefer_parsed: bool) -> List[str]:
    if not header_path.exists():
        return []
    with header_path.open("r", encoding="utf-8") as fp:
        header = json.load(fp)

    sentences: List[str] = []
    for entry in header.values():
        if not entry.get("easy"):
            continue
        parsed_path = Path(utl.get_parsed_path_from_url(entry["url"]))
        crawled_path = Path(utl.get_crawled_path_from_url(entry["url"]))

        content: List[str] = []
        if prefer_parsed and parsed_path.exists():
            content = [
                line.strip()
                for line in parsed_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if not content:
            # Fallback to HTML extraction
            content = extract_from_html(crawled_path)
        sentences.extend(content)
    return sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect easy-language files listed in header.json into one text file."
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help="Path to the corpus Datasets directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/simple_german_sentences.txt"),
        help="Output text file (one sentence per line).",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate sentences (default: dedupe).",
    )
    parser.add_argument(
        "--no-prefer-parsed",
        action="store_true",
        help="Do not prefer parsed files; always try HTML extraction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Point utilities to the provided datasets dir.
    defaultvalues.repository_location = str(args.datasets_dir.parent)
    defaultvalues.dataset_location = str(args.datasets_dir)
    defaultvalues.results_location = str(args.datasets_dir.parent / "results")
    utl.dataset_location = defaultvalues.dataset_location
    utl.results_location = defaultvalues.results_location

    sentences: List[str] = []
    for domain_dir in sorted(args.datasets_dir.iterdir()):
        header_path = domain_dir / "header.json"
        if not header_path.exists():
            continue
        sentences.extend(
            collect_easy_from_header(
                header_path, prefer_parsed=not args.no_prefer_parsed
            )
        )

    if not args.no_dedupe:
        sentences = dedupe(sentences)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sentences), encoding="utf-8")
    print(f"[done] wrote {len(sentences)} sentences to {args.output}")


if __name__ == "__main__":
    main()

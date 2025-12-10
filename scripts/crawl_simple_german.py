import argparse
import re
import time
from pathlib import Path
from typing import Iterable, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": "SimpleGermanCrawler/1.0 (+github.com/example)"
}


def load_urls(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle.readlines()
            if line.strip() and not line.startswith("#")
        ]


def fetch(url: str, timeout: int = 15) -> str:
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as exc:  # pragma: no cover - network failure handling
        print(f"[warn] failed to fetch {url}: {exc}")
        return ""


def extract_links(base_url: str, soup: BeautifulSoup, max_links: int) -> List[str]:
    """Return same-domain links to crawl, capped by max_links."""
    base_netloc = urlparse(base_url).netloc
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(base_url, anchor["href"]).split("#")[0]
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc != base_netloc:
            continue
        if absolute in links:
            continue
        links.append(absolute)
        if len(links) >= max_links:
            break
    return links


def extract_sentences(
    soup: BeautifulSoup,
    min_words: int,
    max_words: int,
    max_word_length: int = 25,
) -> List[str]:
    text_parts: List[str] = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        text = tag.get_text(" ", strip=True)
        if text:
            text_parts.append(text)
    raw_text = re.sub(r"\s+", " ", " ".join(text_parts))
    candidates = re.split(r"(?<=[.!?])\s+", raw_text)
    sentences: List[str] = []
    for candidate in candidates:
        cleaned = candidate.strip().replace("\xa0", " ")
        if not cleaned:
            continue
        words = cleaned.split()
        if not (min_words <= len(words) <= max_words):
            continue
        if any(len(word) > max_word_length for word in words):
            continue
        sentences.append(cleaned)
    return sentences


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    unique: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def crawl(
    sources_path: Path,
    output_path: Path,
    max_links_per_site: int,
    min_words: int,
    max_words: int,
    delay: float,
):
    urls = load_urls(sources_path)
    sentences: List[str] = []
    visited: Set[str] = set()
    for source in urls:
        print(f"[info] crawling seed {source}")
        source_html = fetch(source)
        if not source_html:
            continue
        source_soup = BeautifulSoup(source_html, "html.parser")
        page_links = [source] + extract_links(
            source, source_soup, max_links_per_site - 1
        )
        for page_url in page_links:
            if page_url in visited:
                continue
            visited.add(page_url)
            html = fetch(page_url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            sentences.extend(extract_sentences(soup, min_words, max_words))
            if delay:
                time.sleep(delay)
    unique_sentences = dedupe_preserve_order(sentences)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(unique_sentences), encoding="utf-8")
    try:
        display_path = output_path.resolve().relative_to(Path.cwd())
    except ValueError:
        display_path = output_path
    print(f"[done] wrote {len(unique_sentences)} sentences to {display_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl news pages for simple German sentences."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/simple_german_sources.txt"),
        help="Text file with newline-separated URLs to use as crawl seeds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/simple_german_sentences.txt"),
        help="Path to write extracted sentences.",
    )
    parser.add_argument(
        "--max-links-per-site",
        type=int,
        default=5,
        help="How many pages to fetch per seed domain (including the seed).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Minimum words per sentence to keep.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=18,
        help="Maximum words per sentence to keep.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between page requests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    crawl(
        sources_path=args.sources,
        output_path=args.output,
        max_links_per_site=args.max_links_per_site,
        min_words=args.min_words,
        max_words=args.max_words,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()

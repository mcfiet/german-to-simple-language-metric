"""
Apply the Simple-German-Corpus matching.utilities.preprocess pipeline to text.

This wraps the exact preprocess function from Simple-German-Corpus/matching/utilities.py
so you can run it on your aggregated sentences and write the transformed output
to a new file (one sentence per line).
"""

import argparse
import sys
from pathlib import Path


# Point imports at the cloned Simple-German-Corpus project.
REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "Simple-German-Corpus"
sys.path.insert(0, str(CORPUS_ROOT))

from matching.utilities import preprocess  # type: ignore
import matching.utilities as match_utils  # type: ignore


def run_preprocess(text: str, args: argparse.Namespace) -> list[str]:
    # Avoid spaCy max_length errors on large corpora.
    match_utils.nlp.max_length = max(match_utils.nlp.max_length, len(text) + 1000)
    docs = preprocess(
        text=text,
        remove_hyphens=args.remove_hyphens,
        lowercase=args.lowercase,
        remove_gender=args.remove_gender,
        lemmatization=args.lemmatization,
        spacy_sentences=args.spacy_sentences,
        remove_stopwords=args.remove_stopwords,
        remove_punctuation=args.remove_punctuation,
    )
    return [doc.text.strip() for doc in docs if doc.text.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Simple-German-Corpus matching.utilities.preprocess on a text file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/simple_german_sentences.txt"),
        help="Input text file (content treated as one corpus).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/simple_german_sentences_preprocessed.txt"),
        help="Output file; one preprocessed sentence per line.",
    )
    parser.add_argument(
        "--no-remove-hyphens",
        dest="remove_hyphens",
        action="store_false",
        help="Keep hyphens like '-Wort'.",
    )
    parser.add_argument(
        "--no-lowercase",
        dest="lowercase",
        action="store_false",
        help="Keep original casing.",
    )
    parser.add_argument(
        "--no-remove-gender",
        dest="remove_gender",
        action="store_false",
        help="Keep gendered endings (*in, :innen, _innen, Binnen-I).",
    )
    parser.add_argument(
        "--lemmatization",
        action="store_true",
        help="Enable lemmatization (off by default to match corpus defaults).",
    )
    parser.add_argument(
        "--no-spacy-sentences",
        dest="spacy_sentences",
        action="store_false",
        help="Do not let spaCy split sentences; keep input newlines.",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove German stopwords (defaults to keeping them).",
    )
    parser.add_argument(
        "--remove-punctuation",
        action="store_true",
        help="Strip punctuation after preprocessing.",
    )
    parser.set_defaults(
        remove_hyphens=True,
        lowercase=True,
        remove_gender=True,
        lemmatization=False,
        spacy_sentences=True,
        remove_stopwords=False,
        remove_punctuation=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    sentences = run_preprocess(text, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sentences), encoding="utf-8")
    print(f"[done] wrote {len(sentences)} sentences to {args.output}")


if __name__ == "__main__":
    main()

"""
Generate synthetic normal/complex German sentences from simple inputs using a local Ollama model.

Reads a text file with one simple sentence per line, prompts an Ollama model
with a few-shot template, and writes paired (simple, normal) sentences to a TSV.

Requirements:
- Ollama installed locally and the chosen model pulled (e.g., `ollama pull llama3:8b`).
"""

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple


FEW_SHOT = """Du wandelst einfache deutsche Sätze in normale, flüssige Schriftsprache um.
Der Inhalt bleibt erhalten, die Länge soll ähnlich bleiben (±20 % Wörter).
Gib genau einen Satz zurück, keine Erklärungen, keine Anführungszeichen.

Beispiele:
Einfach: Wir treffen uns morgen um 10 Uhr vor dem Rathaus.
Normal: Wir verabreden uns morgen um 10 Uhr vor dem Rathaus.
Einfach: Die Tür ist zu. Bitte klopfen.
Normal: Die Tür ist geschlossen, bitte klopfen Sie.
Einfach: Anna fährt heute nicht mit, weil sie krank ist.
Normal: Anna kommt heute nicht mit, weil sie krank ist."""


def build_prompt(simple_sentence: str) -> str:
    return f"Einfach: {simple_sentence.strip()}\nNormal:"


def run_ollama(model: str, prompt: str, temperature: float, max_tokens: int, timeout: int) -> str:
    cmd = [
        "ollama",
        "run",
        model,
        f"[[SYSTEM]]\n{FEW_SHOT}\n[[USER]]\n{prompt}\n[[/USER]]",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ollama command not found. Is Ollama installed and on PATH?") from exc
    if result.returncode != 0:
        raise RuntimeError(f"Ollama failed ({result.returncode}): {result.stderr.strip()}")
    # Ollama streams chunks; join stdout lines.
    output = result.stdout.strip()
    return output.splitlines()[-1].strip()


def read_sentences(path: Path, limit: int | None) -> List[str]:
    sentences: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sentences.append(line.strip())
        if limit and len(sentences) >= limit:
            break
    return sentences


def write_tsv(pairs: Iterable[Tuple[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{simple}\t{normal}" for simple, normal in pairs]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic normal German sentences from simple ones using Ollama."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input TXT file with one simple sentence per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_normal.tsv"),
        help="Output TSV file with columns: simple \\t normal.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama model name to use (must be available locally).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="Max tokens to generate per sentence.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of input sentences to process.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sentences = read_sentences(args.input, args.limit)
    pairs: List[Tuple[str, str]] = []
    for idx, simple in enumerate(sentences, 1):
        prompt = build_prompt(simple)
        try:
            normal = run_ollama(
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"[warn] skipping line {idx} due to error: {exc}")
            continue
        pairs.append((simple, normal))
        if idx % 10 == 0:
            print(f"[info] processed {idx}/{len(sentences)}")
    write_tsv(pairs, args.output)
    print(f"[done] wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()

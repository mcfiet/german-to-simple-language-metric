"""
Generate synthetic normal/complex German sentences from simple inputs using an HTTP-accessible OSS LLM.

Reads a text file with one simple sentence per line, prompts the model with a few-shot template,
and writes paired (simple, normal) sentences to a TSV. The endpoint must accept a JSON payload
and return the model completion as plain text.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import requests

FEW_SHOT = """Du wandelst einfache deutsche Sätze (Leichte Sprache) in normales, flüssiges, eher gehobenes Schriftdeutsch um.

Wichtige Anforderungen:
- Bedeutung und Fakten bleiben exakt erhalten, nichts dazuerfinden.
- Länge ähnlich halten (±20 % Wörter).
- Gib genau EINEN Satz zurück.
- Keine Erklärungen, keine Listen, keine Anführungszeichen, keine zusätzlichen Zeilen.

Stilziel: Nicht Leichte Sprache, sondern der Gegenpol: natürlich, idiomatisch, ggf. etwas formeller/komplexer – aber weiterhin klar.

Leitplanken (nur anwenden, wenn es sinnvoll ist):
- W2 (Gegenteil): Verwende bei Bedarf präzisere/fachlichere oder allgemeinere Sammelbegriffe statt sehr konkreter Umschreibungen (z. B. „Bus und Bahn“ → „öffentlicher Nahverkehr“), ohne neue Infos einzuführen.
- W5 (Gegenteil): Kurze Wörter dürfen durch längere/gehobenere Synonyme ersetzt werden (z. B. „Bus“ → „Omnibus“), wenn es natürlich klingt.
- W6 (Gegenteil): Abkürzungen sind erlaubt (z. B. „das heißt“ → „d. h.“), aber nur wenn üblich und ohne Mehrdeutigkeit.
- W7 (Gegenteil): Nominalstil ist erlaubt (z. B. „wir wählen“ → „die Wahl findet statt“), aber nicht übertreiben.
- W8 (Gegenteil): Passiv ist erlaubt (z. B. „wir wählen“ → „es wird gewählt“), wenn es stilistisch passt.
- W9 (Gegenteil): Genitiv ist erlaubt und bevorzugt, wenn natürlich (z. B. „das Haus vom Lehrer“ → „das Haus des Lehrers“).
- W10 (Gegenteil): Konjunktiv ist erlaubt (z. B. „vielleicht regnet es“ → „es könnte regnen“).
- W11 (Gegenteil): Negative Formulierungen sind erlaubt, wenn sie im Ausgangssatz angelegt sind oder idiomatischer wirken; nicht künstlich ins Negative drehen.

Beispiele:
Einfach: Wir treffen uns morgen um 10 Uhr vor dem Rathaus.
Normal: Morgen um 10 Uhr ist das Treffen vor dem Rathaus vorgesehen.
Einfach: Die Tür ist zu. Bitte klopfen.
Normal: Die Tür ist geschlossen; bitte klopfen Sie.
Einfach: Anna fährt heute nicht mit, weil sie krank ist.
Normal: Anna nimmt heute wegen Krankheit nicht teil.
Einfach: Wir fahren mit Bus und Bahn in die Stadt.
Normal: Wir fahren mit dem öffentlichen Nahverkehr in die Stadt.
Einfach: Du darfst hier nicht parken. Das heißt: Stell das Auto woanders hin.
Normal: Sie dürfen hier nicht parken, d. h., stellen Sie das Auto bitte anderswo ab.
Einfach: Das ist das Zimmer von dem Chef.
Normal: Das ist das Zimmer des Chefs.
Einfach: Vielleicht regnet es morgen.
Normal: Morgen könnte es regnen.
Einfach: Morgen wählen wir den Heim-Beirat.
Normal: Morgen wird der Heim-Beirat gewählt.
"""


def build_prompt(simple_sentence: str) -> str:
    return f"Einfach: {simple_sentence.strip()}\nNormal:"


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


def call_model(
    url: str,
    prompt: str,
    system: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": None, 
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected response schema: {json.dumps(data)[:500]}") from exc
    return content.strip().splitlines()[-1].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic normal German sentences from simple ones using an HTTP OSS model."
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
        "--url",
        type=str,
        required=True,
        help="HTTP endpoint of the locally hosted model (OpenAI-compatible chat completion).",
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
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep in seconds between requests to avoid overloading the server.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name to include in the payload if your endpoint requires it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sentences = read_sentences(args.input, args.limit)
    pairs: List[Tuple[str, str]] = []
    for idx, simple in enumerate(sentences, 1):
        prompt = build_prompt(simple)
        try:
            normal = call_model(
                url=args.url,
                prompt=prompt,
                system=FEW_SHOT,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"[warn] skipping line {idx} due to error: {exc}")
            continue
        pairs.append((simple, normal))
        if args.sleep:
            time.sleep(args.sleep)
        if idx % 10 == 0:
            print(f"[info] processed {idx}/{len(sentences)}")
    write_tsv(pairs, args.output)
    print(f"[done] wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()

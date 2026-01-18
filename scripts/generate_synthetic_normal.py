"""
Generate synthetic normal/complex German sentences from simple inputs using a local Ollama model.

Reads a text file with one simple sentence per line, prompts an Ollama model
with a few-shot template, and writes per-input JSON-lists of multiple variants
to a TSV: simple \\t ["normal1", "normal2", ...].

Requirements:
- Ollama installed locally and the chosen model pulled (e.g., `ollama pull llama3:8b`).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Tuple


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

Antwortformat:
- Gib ausschließlich ein gültiges JSON-Objekt: {"normal": "<ein Satz>"}.
- Keine Zusatztexte, keine Erklärungen, keine Codeblöcke, keine Anführungszeichen außerhalb des JSON.

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


def build_prompt(simple_sentence: str, variant: int, total_variants: int) -> str:
    return (
        f"Variante {variant} von {total_variants} (abwechslungsreich gestalten, unterschiedliche Leitplanken anwenden, aber immer faktentreu).\n"
        f"Einfach: {simple_sentence.strip()}\nNormal:"
    )


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
    # Ollama streams chunks; parse JSON from the combined stdout to enforce the schema.
    output = result.stdout.strip()
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"Model returned no JSON object: {output[:200]}")
    try:
        data = json.loads(output[start : end + 1])
        return data["normal"].strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON response: {output[start : end + 1][:200]}") from exc


def read_sentences(path: Path, limit: int | None) -> List[str]:
    sentences: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sentences.append(line.strip())
        if limit and len(sentences) >= limit:
            break
    return sentences


def write_tsv(records: Iterable[Tuple[str, List[str]]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{simple}\t{json.dumps(variants, ensure_ascii=False)}" for simple, variants in records]
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
    parser.add_argument(
        "--variants",
        type=int,
        default=5,
        help="How many alternative normal sentences to generate per input.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per sentence when parsing/requests fail (in addition zum ersten Versuch).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    sentences = read_sentences(args.input, args.limit)
    total = len(sentences)
    records: List[Tuple[str, List[str]]] = []
    skipped_all = 0
    partial = 0
    for idx, simple in enumerate(sentences, 1):
        variants: List[str] = []
        for variant_idx in range(1, args.variants + 1):
            prompt = build_prompt(simple, variant_idx, args.variants)
            normal: str | None = None
            attempts = args.retries + 1
            for attempt in range(1, attempts + 1):
                try:
                    normal = run_ollama(
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    break
                except Exception as exc:
                    if attempt >= attempts:
                        print(
                            f"[warn] skipping variant {variant_idx} on line {idx} after {attempt} attempts: {exc}"
                        )
                    else:
                        print(
                            f"[warn] attempt {attempt}/{attempts} failed on line {idx} variant {variant_idx}: {exc}; retrying..."
                        )
            if normal is not None:
                variants.append(normal)
        if not variants:
            skipped_all += 1
            continue
        if len(variants) < args.variants:
            partial += 1
            print(
                f"[warn] line {idx} produced only {len(variants)}/{args.variants} variants; keeping partial result."
            )
        records.append((simple, variants))
        if idx % 10 == 0:
            print(f"[info] processed {idx}/{len(sentences)}")
    write_tsv(records, args.output)
    duration = time.time() - start_time
    print(f"[done] wrote {len(records)} rows to {args.output}")
    print(
        f"[stats] total input: {total}, full: {len(records) - partial}, partial: {partial}, skipped: {skipped_all}, duration: {duration:.1f}s"
    )


if __name__ == "__main__":
    main()

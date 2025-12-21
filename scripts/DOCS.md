# Übersicht

Übersicht über die Funktion der einzelnen Skripte und Datenverzeichnisse.

## Skripte & Datengenerierung

| Skript | Beschreibung | Generierte Daten (`data/`) |
| :--- | :--- | :--- |
| `collect_hand_aligned_simple.py` | Fügt alle `.simple`-Dateien aus dem hand-aligned Korpus zusammen. | `hand_aligned_simple.txt` |
| `collect_simple_from_headers.py` | Extrahiert Einträge aus SGC-Headern, die als „easy“ markiert sind. | `simple_only_sentences_simple_sgc.txt` |
| `crawl_simple_german.py` | Crawlt Seed-URLs, extrahiert Sätze und entfernt Duplikate. | `simple_german_sentences_crawl.txt` |
| `dedup_sentences.py` | Entfernt doppelte Zeilen aus einer Satzliste. | `simple_german_sentences_from_urls_2_preprocessed_deduped.txt` |
| `extract_ls_docx.py` | Extrahiert Sätze rekursiv aus Lebenshilfe DOCX-Dateien. (Aus Datenschutzgründen nicht mit comitted) | `ls-translations-lebenshilfe/simpel_german_sentences_lebenshilfe.txt` |
| `generate_synthetic_normal.py` | Erzeugt mittels lokalem Ollama-Modell Paare aus einfachen/normalen Sätzen. | `synthetic_normal.tsv` |
| `list_simple_urls.py` | Listet alle URLs aus den SGC-Headern, die als „easy“ markiert sind. | `data/simple_urls.txt` |
| `run_from_easy_url_list.py` | Ruft Seiten einer URL-Liste ab, extrahiert Sätze und speichert sie. | `simple_german_sentences_from_urls.txt` |
| `run_simple_corpus_pipeline.py` | Orchestriert die Crawler/Parser des Simple-German-Corpus. | `simple_german_sentences_pipeline.txt` |
| `group_output_by_normal.py` | Gruppiert CSV-Zeilen nach Originaltext in JSONL. | `data/output/[set]_grouped.jsonl` |

## Daten-Verzeichnisse (`data/`)

- **`data/`**
  Zentraler Sammelordner für alle Korpora, Crawl-Ergebnisse und generierte Artefakte.

- **`data/adl/`**
  ADL-Datensatz-Splits (`train.csv`, `val.csv`, `test.csv`) mit der Spalte `leicht` (einfache Sätze); dient als Input für `collect_adl_simple.py`.

- **`data/output/`**
  Ergebnisse der Experimente (CSV-Paare und gruppierte JSONL-Dateien aus `group_output_by_normal.py`).

- **`data/ls-translations-lebenshilfe/`**
  Lebenshilfe DOCX-Übersetzungen und extrahierte Satz-Listen (`sentences.txt`, `simpel_german_sentences_lebenshilfe.txt`). Die DOCX-Eingabedateien liegen im Unterordner `ls/`.

- **`data/results/`**
  Evaluations-Artefakte aus dem Simple-German-Corpus:
  - `hand_aligned/`: Zeilenweise ausgerichtete Paare (`.simple` und `.normal`).
  - `evaluated/`: Bewertungen durch Annotatoren für Match-Dateien (`*.results`).
  - `matched/`: Ausgabedateien der Matcher (`*.matches`), sortiert nach Algorithmus/Schwellenwert.

- **Dateien direkt in `data/`**
  Aggregierte Satzlisten (`simple_german_sentences_*.txt`) und synthetische TSV-Dateien (`synthetic_normal*.tsv`), die von den oben genannten Skripten generiert wurden.

## Extern

- **`Simple-German-Corpus/`**
  Das Upstream-Projekt, das von mehreren Skripten verwendet wird (für Crawler, Parser, Preprocessing-Tools). Dieses Verzeichnis muss vorhanden sein, damit die Befehle funktionieren.

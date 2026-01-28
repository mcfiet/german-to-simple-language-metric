# genai-project

# **README – Metrik zur Bewertung von Übersetzungen in Leichte Sprache**

## **Kurzüberblick**

Kurze, praktische Übersicht zum Projektaufbau, den Skripten und Notebooks. Die inhaltliche Ausarbeitung steht in der Thesis.

---

## **Schnellstart**

```bash
pip install -r requirements.txt
```

Hinweis: Einige Skripte benötigen den Ordner `Simple-German-Corpus/` sowie nicht eingecheckte DOCX‑Daten (Lebenshilfe).

---

## **Skripte (Datenerzeugung & Utilities)**

Die meisten Skripte liegen in `scripts/`, zusätzlich gibt es `data/group_output_by_normal.py`. Details zu Inputs/Outputs stehen in `scripts/DOCS.md`.

| Skript | Zweck (kurz) | Output (data/) |
| :--- | :--- | :--- |
| `collect_hand_aligned_simple.py` | sammelt `.simple`‑Dateien | `hand_aligned_simple.txt` |
| `collect_simple_from_headers.py` | extrahiert „easy“ aus SGC‑Headern | `simple_only_sentences_simple_sgc.txt` |
| `crawl_simple_german.py` | crawlt Seed‑URLs | `simple_german_sentences_crawl.txt` |
| `dedup_sentences.py` | dedupliziert Satzlisten | `*_deduped.txt` |
| `extract_ls_docx.py` | extrahiert Lebenshilfe‑DOCX | `ls-translations-lebenshilfe/...` |
| `generate_synthetic_normal.py` | synthetische Paare (lokales LLM) | `synthetic_normal.tsv` |
| `generate_synthetic_normal_http.py` | synthetische Paare (HTTP/Remote) | `synthetic_normal_http.tsv` |
| `label_synthetic_normal.py` | Labeling‑Variante | `*_labeled.csv` |
| `label_synthetic_normal_5ex.py` | Labeling mit 5 Beispielen | `*_5ex_labeled.csv` |
| `list_simple_urls.py` | URLs aus SGC‑Headern | `data/simple_urls.txt` |
| `run_from_easy_url_list.py` | extrahiert Sätze aus URL‑Liste | `simple_german_sentences_from_urls.txt` |
| `run_simple_corpus_pipeline.py` | Pipeline für Simple‑German‑Corpus | `simple_german_sentences_pipeline.txt` |
| `data/group_output_by_normal.py` | gruppiert CSVs nach Originalsatz | `data/output/[set]_grouped.jsonl` |

Beispiel:

```bash
python scripts/run_simple_corpus_pipeline.py
```

---

## **Notebooks**

Trainings‑ und Analyse‑Notebooks liegen in `notebooks/`:

- `baseline_tfidf_logreg.ipynb` – Baseline mit TF‑IDF + LogReg
- `lstm_training.ipynb` – BiLSTM‑Training
- `mlp_training.ipynb` – MLP‑Kopf auf Embeddings
- `sbert_finetune_frozen.ipynb` – SBERT, Encoder eingefroren
- `sbert_finetune_unfrozen.ipynb` – SBERT, volles Finetuning
- `sbert_fintetune_unfrozen_last_n_layers.ipynb` – nur letzte N‑Layer
- `sbert_finetune_lora.ipynb` – LoRA‑Finetuning
- `mpnet_finetune_unfrozen.ipynb` – MPNet‑Vergleich
- `learning_curve_analysis.ipynb` – Lernkurven‑Analyse

---

## **Daten/Outputs**

Relevante Strukturen:

```
data/
 ├── results/
 │   ├── evaluated/    # Label: Verständlichkeit / Qualität
 │   ├── matched/      # gepaarte Sätze (Original ↔ Leichte Sprache)
 │   └── hand_aligned/ # manuell ausgerichtete Paare
 ├── output/           # Train/Val/Test + gruppierte JSONL
 ├── ls-translations-lebenshilfe/ # interne DOCX-Daten (nicht eingecheckt)
 └── ...               # aggregierte Satzlisten + synthetische TSV/CSV
```

Details zu Outputs: `data/output/README.md` und `data/results/README.md`.

---

## **Ordnerstruktur**

```
.
├── data/
│   ├── results/
│   ├── output/
│   └── ...
│
├── scripts/            # Datengenerierung, Crawler, Labeling (Details in scripts/DOCS.md)
├── notebooks/          # Trainings-/Analyse-Notebooks
├── models/             # gespeicherte/finetuned Modelle
├── Simple-German-Corpus/ # Upstream-Abhängigkeit für mehrere Skripte
├── thesis/             # LaTeX-Arbeit + Artefakte
├── ui/                 # lokaler UI-Ordner (derzeit nur Platzhalter)
└── venv/               # lokale Python-Umgebung
│
└── README.md
```

---

## **Zusätzliche Doku**

- `scripts/DOCS.md`: Übersicht über alle Skripte und generierte Daten.
- `data/output/README.md`: Beschreibung der experimentellen Outputs.
- `data/results/README.md`: Beschreibung der Evaluationsdaten und Ground Truths.

## **Hinweise**

- Einige Skripte erwarten, dass `Simple-German-Corpus/` vorhanden ist.
- Lebenshilfe-DOCX-Daten sind aus Datenschutzgründen nicht eingecheckt.

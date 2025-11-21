# genai-project

# **README – Metrik zur Bewertung von Übersetzungen in Leichte Sprache**

## **Projektbeschreibung**

Dieses Projekt entwickelt eine Metrik zur Bewertung von Übersetzungen **von deutscher Standardsprache in Leichte Sprache**.
Ziel ist es, automatisch zu bestimmen:

1. **Wie inhaltlich übereinstimmend** ein Originaltext und seine Übersetzung in Leichte Sprache sind.
2. **Wie verständlich** die Leichte-Sprache-Version ist.
3. **Wie hoch die semantische Qualität** der Übersetzung ausfällt.
4. **Ob und wie stark** der übersetzte Text tatsächlich den Kriterien der Leichten Sprache entspricht.

Die Metrik soll später als **Bewertungsfunktion (Reward-Funktion)** in einem Übersetzungsmodell oder in einer generativen Pipeline genutzt werden.

---

## **Motivation**

Übersetzungen in Leichte Sprache sind für Barrierefreiheit essenziell.
Maschinelle Übersetzungssysteme existieren, aber:

- sie sind oft inkonsistent,
- verlieren Inhalte,
- oder liefern sprachlich zu komplexe Sätze.

Eine robuste Metrik zur automatischen Bewertung dieser Übersetzungen fehlt bisher.
Dieses Projekt schließt diese Lücke, indem es eine datenbasierte, lernbare Bewertungsfunktion entwickelt – vermutlich ein **Regressionsmodell**, das auf verschiedenen Qualitätsdimensionen lernt.

---

## **Ziele**

- Aufbau einer **automatischen Qualitätsmetrik** für Leichte-Sprache-Übersetzungen
- Bewertung entlang zweier Kernaspekte:

  - **Semantische Übereinstimmung** (Meaning Preservation)
  - **Verständlichkeit / Einfachheitsgrad** (Readability, Guidelines der Leichten Sprache)

- Nutzung als:

  - **Evaluationsmetrik**,
  - **Reward-Funktion** für RL oder GAN-basierte Ansätze,
  - oder als Qualitätssignal in einem Übersetzungsmodell.

---

## **Datenbasis**

Der Hauptdatensatz ist derselbe wie im ADL-Projekt:
Lamarr-Institut Datensatz zu Leichter Sprache (Übersetzungen in verschiedene Schwierigkeitsgrade)

Falls Zeit bleibt:

- **Eigene Daten** aus den Texten/Übersetzungen der Lebenshilfe Kiel.
- Optionale Erweiterung um Daten, die aus generierten Leichte-Sprache-Übersetzungen bestehen.

Alle Daten liegen im Ordner `data/` in drei Strukturen:

```
data/
 ├── evaluated/        # enth. Label: Verständlichkeit / Qualität
 ├── matched/          # gepaarte Sätze (Original ↔ Leichte Sprache)
 └── hand_aligned/     # manuell ausgerichtete Paare
```

---

## **Methodik & Modelle**

### **1. Baseline**

Ein erstes **Regressionsmodell**, das einfache Textmerkmale nutzt:

- Satzlänge,
- Wortkomplexität,
- Embedding-Ähnlichkeit,
- syntaktische Merkmale.

### **2. Semantische Qualitätsmetrik**

Verwendung moderner Sprachmodelle für:

- Text-Embedding-Ähnlichkeit (z. B. Sentence Transformers)
- Alignment zwischen Original und Leichter Sprache
- Bewertung semantischer Konsistenz

### **3. Verständlichkeitsmetrik**

Ableitung von Messwerten wie:

- Lesbarkeitsindizes
- Regelkonformität nach Leichte-Sprache-Standards
- linguistische Einfachheit

### **4. Erweiterte Methoden (optional)**

- Einsatz eines **GAN**, um Qualitätsmetriken oder Bewertungsscores zu generieren (Diskriminator als “Quality Judge”)
- Human-Evaluation als Benchmark

---

## **Workflow / Projektzeitplan**

Aus der Projektplanung (8-Wochen-Plan) :

| Woche | Aufgabe                                                           |
| ----- | ----------------------------------------------------------------- |
| 1     | Projektsetup, Zieldefinition                                      |
| 2     | Datensammlung, -aufbereitung, Dataloader bauen                    |
| 3     | Fertigstellung der Datenpipeline, Baseline-Metrik (Regression)    |
| 4–5   | Evaluation, erste Modellverbesserungen                            |
| 5–6   | Entwicklung der Semantik-Qualitätsmetrik                          |
| 7     | Kombination, Validierung                                          |
| 8     | Human Evaluation, Analyse, Visualisierung, Bericht & Präsentation |

---

## **Ordnerstruktur**

```
.
├── data/
│   ├── evaluated/
│   ├── matched/
│   └── hand_aligned/
│
├── src/
│   ├── dataloader.py
│   ├── preprocess.py
│   ├── baseline_regression.py
│   ├── semantic_metric.py
│   └── evaluation.py
│
└── README.md
```

---

## **Installation & Setup**

```bash
git clone <repository-url>
cd <repo>

pip install -r requirements.txt
```

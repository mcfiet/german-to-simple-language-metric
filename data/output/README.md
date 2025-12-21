# data/output

Zwischengespeicherte CSV- und gruppierte JSONL-Dateien, die während der Experimente verwendet werden.

- `train.csv`, `val.csv`, `test.csv`: Satzpaare mit den Spalten `original` und `leicht`.
- `*_grouped.jsonl`: Erstellt von `data/group_output_by_normal.py`. Gruppiert alle einfachen Varianten unter dem jeweiligen normalen Satz.

Die Datensätze stammen noch aus dem ADL-Projekt

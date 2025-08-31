# Primes

Dies ist ein Repository für eine praktische Implementierung von Primzahltests in Python.

---

## Projektüberblick

- Git-Repository wurde erstellt und initial strukturiert.
  - `src/primality` für alle Implementierungen für die Primzahltests
  - `src/analysis` für die Analyse durch Plots und Datenexports
  - `tests/` für Unittests
  - `data/` für die Datensammlung der Testpools
  - `results/` für die Analyseauswertung der Daten 
  - `requirements.txt` für alle benötigten Pakete


---

## Installation
### Installationsschritte
Repo klonen:
```bash
git clone https://github.com/sp8cky/Primes && cd Primes
```

### Dependencies installieren
```bash
pip install -r requirements.txt
```
### Skript ausführen
Aus dem root directory:
```bash
python src/analysis/prime-analysis
```

---
## Details zur Implementierung

### Testablauf
- Testdaten werden initialisiert
- Für jeden Test wird ein Pool an Zahlen n erzeugt 
- Die Laufzeitmessung wird über t Wiederholungen durchgeführt
- Die Protokollversionen aller Tests werden folgend durchgeführt, um weitere Daten zu protokollieren (siehe [Datenerhebnung pro Test](#Datenerhebnung-pro-Test))
- Die Ergebnisse werden über Plots visualisiert und per csv-Datei exportiert

### Datenerhebnung pro Test
Felder (werden für jeden Test ergänzt, wenn verfügbar):
- `Zahl`: Die getestete Zahl n
- `Test`: Name des Tests (z. B. Fermat, Lucas, Proth…)
- `Ergebnis`: Ob der Test n als Primzahl erkannt hat (`True`/`False`)
- `best_time`:	Kürzeste gemessene Laufzeit über alle Wiederholungen
- `avg_time`:	Durchschnittlich gemessene Laufzeit über alle Wiederholungen
- `worst_time`:	Längste gemessene Laufzeit über alle Wiederholungen
- `std_dev`:	Standardabweichung der Laufzeit in Millisekunden (Stabilität der Messung)
- `a_values`: Liste der verwendeten Zufallsbasen a
- `reason`: Begründung, falls der Test nicht durchführbar war
- `other_fields`: Testspezifische Angaben von Zwischenergebnissen

Folgende Tests dokumentieren folgende Angaben:

| Gr | Test                        | n   | res | TP | FP  | TN  | EC | ER  | Best  | Avg   | Worst | Std  | a_values                             | Other_fields                                      | Reason |
|----|-----------------------------|-----|-----|----|-----|-----|----|-----|-------|-------|-------|------|--------------------------------------|--------------------------------------------------|--------|
|    | Fermat (k=)                 |     |     |    |     |     |    |     |       |       |       |      | [(a1, result), (...)]                |                                                  |        |
|    | Miller-Rabin (k=)           |     |     |    |     |     |    |     |       |       |       |      | [(a1, result), (...)]                |                                                  |        |
|    | Solovay-Strassen (k=)       |     |     |    |     |     |    |     |       |       |       |      | [(a1, cond1, cond2), (...)]          |                                                  |        |
|    | Initial Lucas               |     |     |    |     |     |    |     |       |       |       |      | [(a1, cond1, cond2), (...)]          |                                                  |        |
|    | Lucas                       |     |     |    |     |     |    |     |       |       |       |      | [(a1, cond1, cond2), (...)]          |                                                  |        |
|    | Optimized Lucas             |     |     |    |     |     |    |     |       |       |       |      | {q1: (a1, cond1, cond2), q2: (...)}  |                                                  |        |
|    | Pepin                       |     |     |    |     |     |    |     |       |       |       |      |                                      |                                                  |        |
|    | Lucas-Lehmer                |     |     |    |     |     |    |     |       |       |       |      |                                      | [p, sequence, S]                                 |        |
|    | Wilson                      |     |     |    |     |     |    |     |       |       |       |      |                                      |                                                  |        |
|    | AKS10                       |     |     |    |     |     |    |     |       |       |       |      |                                      | [initial_check, find_r, prime_divisor_check, polynomial_check] |        |
|    | Proth Variant               |     |     |    |     |     |    |     |       |       |       |      | [(a_1, result), (...)]               |                                                  |        |
|    | Pocklington                 |     |     |    |     |     |    |     |       |       |       |      |                                      |                                                  |        |
|    | Optimized Pocklington       |     |     |    |     |     |    |     |       |       |       |      | {q1: (a_1, cond1, cond2), q2: (...)} |                                                  |        |
|    | Optimized Pocklington Var.  |     |     |    |     |     |    |     |       |       |       |      | {q1: (a_1, cond1, cond2), q2: (...)} | [b, pow(b, (n - 1) // F, n)]                     |        |
|    | Generalized Pocklington     |     |     |    |     |     |    |     |       |       |       |      | [(a1, cond1, cond2), (...)]          | [K, p, n]                                        |        |
|    | Rao                         |     |     |    |     |     |    |     |       |       |       |      | [(a1, cond1, cond2), (...)]          | [p, 2, n_exp]                                    |        |
|    | Ramzy                       |     |     |    |     |     |    |     |       |       |       |      | [(a1, cond1, cond2), (...)]          | [K, p, n_exp]                                    |        |

### Ergebnisse
- Daten werden im CSV-Format und die Plots im PNG-Format gespeichert
- Die CSV-Dateien sind Grundlage für die statistische Auswertung

### Auswertung
- Auswertung erfolgt pro Testpool über `src/analysis/analyse-data.py`
- Ergebnisse werden als csv-Datei exportiert

---
## Credits
This project was created by sp8cky.


---
---
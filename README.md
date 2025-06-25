# Primes

Repo for runtime analysis of prime tests.

## 01 – Einleitung und aktueller Stand

In diesem Notebook wird der aktuelle Stand der Arbeit dokumentiert. Ziel ist es, die Implementierung und Analyse von Primzahlkriterien und Primzahltests im Kontext der Kryptographie vorzubereiten.
pip-25.1.1
Python 3.13.3
---

### ✅ Projektüberblick

- Git-Repository wurde erstellt und initial strukturiert.
- Projektstruktur orientiert sich an einem modularen Aufbau mit:
  - `src/` für die Implementierung
  - `tests/` für Unittests
  - `requirements.txt` für alle benötigten Pakete
- Diese Jupyter-Datei dient als Dokumentation und explorative Umgebung.


### Theoretischer Stand

#### Allgemeine Primkriterien
Folgende Tests wurden bereits implementiert und dokumentieren folgende Angaben:
Folgende Tests wurden bereits implementiert und dokumentieren folgende Angaben:

| Ergebnis | Test                         | Eingabe | Best Time | Avg Time | Worst Time | Std.abweichung | a_values                             | Other_fields                                      | Reason |
|----------|------------------------------|---------|-----------|----------|------------|----------------|--------------------------------------|--------------------------------------------------|--------|
|          | Fermat (k=)                  |         |           |          |            |                | [(a_1, result), (...)]               |                                                  |        |
|          | Miller-Rabin (k=)            |         |           |          |            |                | [(a_1, result), (...)]               |                                                  |        |
|          | Solovay-Strassen (k=)        |         |           |          |            |                | [(a_1, result), (...)]               |                                                  |        |
|          | Initial Lucas                |         |           |          |            |                | [(a_1, cond1, cond2), (...)]         |                                                  |        |
|          | Lucas                        |         |           |          |            |                | [(a_1, cond1, cond2), (...)]         |                                                  |        |
|          | Optimized Lucas              |         |           |          |            |                | {q_1: (a_1, cond1, cond2), q2: (...)} |                                                 |        |
|          | Pepin                        |         |           |          |            |                |                                      |                                                  |        |
|          | Lucas-Lehmer                 |         |           |          |            |                |                                      | [p, sequence, S]                                 |        |
|          | Wilson                       |         |           |          |            |                |                                      |                                                  |        |
|          | AKS                          |         |           |          |            |                |                                      | [initial_check, find_r, prime_divisor_check, polynomial_check] |        |
|          | Proth Variant                |         |           |          |            |                | [(a_1, result), (...)]               |                                                  |        |
|          | Pocklington                  |         |           |          |            |                |                                      |                                                  |        |
|          | Optimized Pocklington        |         |           |          |            |                |                                      |                                                  |        |
|          | Optimized Pocklington Variant |        |           |          |            |                | {q_1: (a_1, cond1, cond2), q2: (...)}| [b, pow(b, (n - 1) // F, n)]                     |        |
|          | Generalized Pocklington      |         |           |          |            |                | [(a_1, cond1, cond2), (...)]         | [K, p, n]                                        |        |
|          | Grau                         |         |           |          |            |                | [a_1]                                | [K, p, n, phi]                                   |        |
|          | Grau Probability             |         |           |          |            |                | [a_1]                                | [K, p, n, phi, j]                                |        |


---

### Implementierung
#### DIC Struktur
# Gespeicherte Testdaten pro Primzahltest

Felder (werden für jeden Test ergänzt, wenn verfügbar):
- `Zahl`: Die getestete Zahl n
- `Test`: Name des Tests (z. B. Fermat, Lucas, Proth…)
- `Ergebnis`: Ob der Test n als Primzahl erkannt hat (`True`/`False`)
- `best_time`:	Kürzeste gemessene Laufzeit über alle Wiederholungen
- `worst_time`:	Längste gemessene Laufzeit über alle Wiederholungen
- `std_dev`:	Standardabweichung der Laufzeit in Millisekunden (Stabilität der Messung)
- `a_values`: Liste der verwendeten Zufallsbasen a
- `reason`: Begründung, falls der Test nicht durchführbar war
- `other_fields`: Testspezifische Angaben von Zwischenergebnissen


### Nächste Schritte
- Implementierung erster Laufzeitanalysen (`time`, `timeit`, `cProfile`)
- Vergleich der Laufzeiten für verschiedene `n`
- Visualisierung der Laufzeitkomplexität
- Erweiterung um probabilistische Tests zur Gegenüberstellung (z. B. Miller–Rabin)

---

```bash
pip install -r requirements.txt

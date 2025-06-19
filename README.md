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
  - `notebooks/` für interaktive Analyse
  - `tests/` für Unittests
  - `requirements.txt` für alle benötigten Pakete
- Diese Jupyter-Datei dient als Dokumentation und explorative Umgebung.

Primes/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 pyproject.toml                  # Optional, falls du Poetry o.Ä. verwendest
│
├── 📁 src/                            # Hauptcode
│   ├── __init__.py
│   ├── primality/                    # Module für Tests & Kriterien
│   │   ├── __init__.py
│   │   ├── criteria.py               # Primzahlkriterien (z. B. Euklid, Wilson, Fermat-Kriterium)
│   │   ├── tests.py                  # Primality Tests (Fermat, Miller-Rabin, AKS, etc.)
│   │   ├── helpers.py                # Hilfsfunktionen (z. B. Modexp, gcd, witness-Erzeugung)
│   │
│   ├── analysis/                     # Laufzeitanalyse & Auswertung
│   │   ├── __init__.py
│   │   ├── timing.py                 # Zeitmessung (timeit, Messreihen)
│   │   ├── dataset.py                # Generierung & Speicherung von Testdaten
│   │   ├── plot.py                   # Visualisierung mit matplotlib/seaborn
│
├── 📁 notebooks/                     # Jupyter-Notebooks für Analyse & Präsentation
│   ├── 01_Theorie.ipynb             # Mathematische Einführung (Texte + Tests)
│   ├── 02_Laufzeit_MillerRabin.ipynb
│   ├── 03_Vergleich_Tests.ipynb
│
├── 📁 tests/                         # Unit-Tests mit pytest
│   ├── test_criteria.py
│   ├── test_tests.py
│   ├── test_helpers.py
│
├── 📁 data/                          # (optional) CSV/JSON für gespeicherte Messwerte
│   ├── miller_rabin_results.csv
│
└── 📁 reports/                       # (optional) generierte PDFs/LaTeX-Bausteine
    ├── laufzeitanalyse_miller.pdf

---

### Theoretischer Stand

#### Allgemeine Primkriterien

Folgende mathematische Kriterien wurden formal definiert und teilweise bereits implementiert:

- **Fermat-Kriterium**  
- **Wilson-Kriterium**  
- **Lucas-Test (Vorläufer, Lucas-Test, Optimierung)**

---

#### Deterministische Primzahltests
Diese Tests wurden anhand der zugrundeliegenden Theorien erarbeitet:
- **MSRT-Test** (Miller–Rabin–Solovay–Test in deterministischer Variante für kleine `n`)
- **SST-Test** (Solovay–Strassen-Test)
- **AKS-Test** (Agrawal–Kayal–Saxena, vollständig deterministisch)

---

### Implementierung
#### DIC Struktur
# Gespeicherte Testdaten pro Primzahltest

Allgemeine Felder (werden für jeden Test ergänzt, wenn verfügbar):
- `Zahl`: Die getestete Zahl n
- `Test`: Name des Tests (z. B. Fermat, Lucas, Proth…)
- `Ergebnis`: Ob der Test n als Primzahl erkannt hat (`True`/`False`)
- `time`: Laufzeit des Tests in Millisekunden (z. B. `"0.173 ms"`)

## Fermat
- `a_values`: Liste der verwendeten Zufallsbasen a
- `results`: Liste der booleschen Teilergebnisse je Durchlauf

## Wilson
- `result`: Ergebnis des Wilson-Kriteriums

## Initial Lucas
- `a`: Zufällig gewählte Basis a
- `condition1`: Ergebnis der Bedingung a^{n-1} \equiv 1 \mod n
- `early_break`: Frühzeitiger Abbruch bei kleinem Teiler m von n-1
- `result`: Endergebnis

## Lucas
- `a`: Zufällig gewählte Basis a
- `condition1`: Ergebnis der Bedingung a^{n-1} \equiv 1 \mod n
- `early_break`: Frühzeitiger Abbruch bei kleinem Teiler m von n-1
- `result`: Endergebnis

## Optimized Lucas
- `factors`: Primfaktoren von n-1
- `tests`: Dict: \{q: [(a, Bedingung erfüllt?)]\}
- `result`: Endergebnis

## Pepin
- `k`: Wert k bei Fermat-Zahl F_k = 2^{2^k} + 1
- `calculation`: Klartext der Rechnung 3^{(n-1)/2} \mod n
- `reason`: Begründung, falls der Test nicht durchführbar war
- `result`: Endergebnis

## Lucas-Lehmer
- `p`: Wert aus n = 2^p - 1
- `sequence`: Berechnete Folge S_i
- `final_S`: Letzter Wert S_{p-2}
- `reason`: Begründung bei Scheitern
- `result`: Endergebnis

## Proth
- `a_values`: Liste der getesteten Basen
- `results`: Ergebnis pro Basis
- `reason`: Erklärung bei Abbruch
- `result`: Endergebnis

## Pocklington
- `a_values`: getestete Basen
- `condition1`: Liste der Ergebnisse a^{n-1} \equiv 1 \mod n
- `condition2`: Liste der Ergebnisse für die Nebenbedingung (ggT-Test)
- `reason`: Begründung bei Scheitern
- `result`: Endergebnis

## Optimized Pocklington
- `tests`: Dict: \{q: [(a, Bedingung erfüllt?)]\}
- `reason`: Begründung bei Scheitern
- `result`: Endergebnis

## Proth Variant
- `a`: verwendete Basis
- `reason`: Begründung bei Scheitern
- `result`: Endergebnis

## Optimized Pocklington Variant
- `tests`: Dict: \{q: (a, Bedingung erfüllt?)\}
- `b_test`: b-Wert und ob er Bedingung erfüllt
- `reason`: Begründung bei Scheitern
- `result`: Endergebnis

## Generalized Pocklington
- `K`, `p`, `n`: Werte aus Zerlegung N = Kp^n + 1
- `a`: gefundene Basis
- `attempts`: Liste mit Tupeln (a, condition1, condition2)
- `reason`: Erklärung bei Fehlschlag
- `result`: Endergebnis

## Grau
- `K`, `p`, `n`: wie oben
- `a`: quadratische Nicht-Residue modulo p
- `phi_p`: Wert des zyklotomischen Polynoms modulo N
- `exponent`: verwendeter Exponent
- `reason`: Erklärung bei Fehlschlag
- `result`: Endergebnis

## Grau Probability
- `K`, `p`, `n`: wie oben
- `a`: getestete Basis
- `j`: Index zur Berechnung
- `attempts`: Liste von Versuchen mit (a, j, condition1, condition2)
- `reason`: Erklärung bei Fehlschlag
- `result`: Endergebnis

## Miller-Rabin
- `repeats`: Liste von (a, bestanden?) pro Runde
- `results`: Liste von booleschen Ergebnissen pro Versuch

## Solovay-Strassen
- `repeats`: Liste von (a, bestanden?) pro Runde
- `results`: wie oben

## AKS
- `steps.initial_check`: Gilt Basisannahme? (nicht Potenz etc.)
- `steps.find_r`: gefundener Wert r mit ord_r(n) > log²(n)
- `steps.prime_divisor_check`: Info über gefundene Teiler
- `steps.polynomial_check`: Liste mit (a, Test erfolgreich?) für Polynombedingung
- `result`: Endergebnis



### Nächste Schritte
- Implementierung erster Laufzeitanalysen (`time`, `timeit`, `cProfile`)
- Vergleich der Laufzeiten für verschiedene `n`
- Visualisierung der Laufzeitkomplexität
- Erweiterung um probabilistische Tests zur Gegenüberstellung (z. B. Miller–Rabin)


---

### Setup-Hinweise

Dieses Notebook läuft in einer isolierten Umgebung. Stelle sicher:

```bash
pip install -r requirements.txt

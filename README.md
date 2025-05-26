# Primes

Repo for runtime analysis of prime tests.

## 01 – Einleitung und aktueller Stand

In diesem Notebook wird der aktuelle Stand der Arbeit dokumentiert. Ziel ist es, die Implementierung und Analyse von Primzahlkriterien und Primzahltests im Kontext der Kryptographie vorzubereiten.

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
- **Lucas-Test (einfacher Vorläufer, keine vollständige Lucas-Reihe)**

---

#### Deterministische Primzahltests

Diese Tests wurden anhand der zugrundeliegenden Theorien erarbeitet:
- **MSRT-Test** (Miller–Rabin–Solovay–Test in deterministischer Variante für kleine `n`)
- **SST-Test** (Solovay–Strassen-Test)
- **AKS-Test** (Agrawal–Kayal–Saxena, vollständig deterministisch)

---

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

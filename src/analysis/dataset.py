import os, json, csv
from datetime import datetime
from typing import Dict, Any
from src.primality.tests import test_data

# creates data directory relative to the src directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# get timestamped filename for saving results
def get_timestamped_filename(basename: str, ext: str = "json"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{basename}.{ext}"


def export_test_data_to_csv(test_data: dict, filename: str):
    """
    Exportiert die Testdaten in eine CSV-Datei.
    
    Felder in der CSV (dynamisch, Beispielhafte Erklärung):
    - Test: Name des durchgeführten Primzahltests (z.B. "Fermat", "Lucas", "Proth" etc.)
    - Zahl: Die getestete Zahl (int)
    - time: Laufzeit des Tests in Millisekunden (z.B. "0.123 ms")
    - Ergebnis: boolscher Wert (True/False), ob die Zahl als prim erkannt wurde
    - a: Ein Wert a, der für manche Tests genutzt wird (z.B. Lucas-Test)
    - a_values: Liste von a-Werten, die bei iterativen Tests geprüft wurden (z.B. Fermat)
    - attempts: Liste von Versuchen oder Zwischenergebnissen
    - b_test: Hilfswert oder Zwischenergebnis in manchen Tests (z.B. Pocklington)
    - calculation: String mit Berechnungsschritten oder Zwischenergebnissen
    - condition1, condition2: boolsche Bedingungen in Tests, z.B. Lucas-Bedingungen
    - early_break: Frühzeitiger Abbruchwert (z.B. m bei Lucas-Test)
    - exponent: Exponent in Potenzierungen
    - factors: Faktorenzerlegung (z.B. dict mit Primfaktoren)
    - final_S: Letzter Wert einer Sequenz (z.B. Lucas-Lehmer-Test)
    - j, k, n, p: Parameter je nach Test (z.B. Indizes, Primzahlen, Exponenten)
    - phi_p: Wert der Eulerschen Phi-Funktion bei p
    - reason: Grund für Abbruch oder Ergebnis (String)
    - repeats: Anzahl der Wiederholungen bei probabilistischen Tests
    - results: Liste von boolschen Testergebnissen einzelner Teiltests
    - sequence: Liste mit Zwischenergebnissen einer Zahlenfolge
    - steps: Schritte oder Phasen bei komplexen Tests (z.B. AKS)
    - tests: Untertests (z.B. bei optimierten Varianten)
    
    Hinweis: Nicht alle Felder sind in jedem Test gesetzt. Die Spalten werden dynamisch erzeugt.
    """

    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)

    if not test_data:
        print("⚠️ Keine Testdaten vorhanden zum Export.")
        return

    rows = []

    for testname, entries in test_data.items():
        for number, details in entries.items():
            flat_row = {
                "Test": testname,
                "Zahl": number,
                "time": f"{details.get('time', 0)*1000:.3f} ms",
                "Ergebnis": details.get("result"),
            }
            # Füge alle weiteren Felder hinzu, außer 'time', 'Zahl', 'Test' und 'result'
            for key, value in details.items():
                if key not in {"result", "time"}:
                    flat_row[key] = str(value)
            rows.append(flat_row)

    if not rows:
        print("⚠️ Testdaten sind leer.")
        return

    # Sortiere die Zeilen nach Zahl, sodass alle Tests für eine Zahl zusammenstehen
    rows.sort(key=lambda r: r["Zahl"])

    # Spalten dynamisch sammeln
    fieldnames = sorted({key for row in rows for key in row.keys()})

    # 'time' direkt hinter 'Zahl' einsortieren
    if "time" in fieldnames and "Zahl" in fieldnames:
        fieldnames.remove("time")
        zahl_index = fieldnames.index("Zahl")
        fieldnames.insert(zahl_index + 1, "time")

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Testdaten erfolgreich exportiert nach {path}")

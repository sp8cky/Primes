import os, json, csv
from datetime import datetime
from typing import Dict, Any
from src.primality.test_protocoll import test_data

# creates data directory relative to the src directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# get timestamped filename for saving results
def get_timestamped_filename(basename: str, ext: str = "json"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{basename}.{ext}"


def export_test_data_to_csv(test_data: dict, filename: str):

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

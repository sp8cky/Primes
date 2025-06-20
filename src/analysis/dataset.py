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
                "Ergebnis": details.get("result"),
                "Test": testname,
                "Zahl": number,
                "best_time": f"{details.get('best_time', 0)*1000:.3f} ms",
                "avg_time": f"{details.get('avg_time', 0)*1000:.3f} ms",
                "worst_time": f"{details.get('worst_time', 0)*1000:.3f} ms",
                "std_dev": f"{details.get('std_dev', 0)*1000:.3f} ms",
            }
            # Füge alle weiteren Felder hinzu, außer 'time', 'Zahl', 'Test' und 'result'
            for key, value in details.items():
                if key not in {"result", "best_time", "avg_time", "worst_time", "std_dev"}:
                    flat_row[key] = str(value)
            rows.append(flat_row)

    if not rows:
        print("⚠️ Testdaten sind leer.")
        return

    # Sortiere die Zeilen nach Zahl, sodass alle Tests für eine Zahl zusammenstehen
    rows.sort(key=lambda r: r["Zahl"])

    # Spalten dynamisch sammeln
    fieldnames = [
    "Ergebnis", "Test", "Zahl",
    "best_time", "avg_time", "worst_time", "std_dev",
    "a_values", "other_fields", "reason"
]
    # Ergänze dynamisch alle übrigen Keys, die noch fehlen
    all_keys = {key for row in rows for key in row.keys()}
    for key in all_keys:
        if key not in fieldnames:
            fieldnames.append(key)

        # 'time' direkt hinter 'Zahl' einsortieren
        if "time" in fieldnames and "Zahl" in fieldnames:
            fieldnames.remove("time")
            zahl_index = fieldnames.index("Zahl")
            fieldnames.insert(zahl_index + 1, "time")

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Testdaten erfolgreich exportiert.")

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
                "Ergebnis": details.get("result"),
            }

            # Füge alle weiteren Felder dynamisch hinzu
            for key, value in details.items():
                if key != "result":
                    flat_row[key] = str(value)

            rows.append(flat_row)  # Füge Zeile hinzu

    if not rows:
        print("⚠️ Testdaten sind leer.")
        return

    # Schreibe dynamisch alle Keys als Spaltenüberschriften
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Testdaten erfolgreich nach {path} exportiert.")

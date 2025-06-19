import os, json, csv
from datetime import datetime
from typing import Dict
from src.primality.tests import test_data

# creates data directory relative to the src directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# get timestamped filename for saving results
def get_timestamped_filename(basename: str, ext: str = "json"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{basename}.{ext}"

# save data to a JSON file
def save_json(data, filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# load data from a JSON file
def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)

# export datasets to a CSV file
def export_to_csv(datasets, filename):
    path = os.path.join(DATA_DIR, filename)
    all_rows = []
    for method, results in datasets.items():
        for row in results:
            row_copy = row.copy()
            row_copy["Methode"] = method
            all_rows.append(row_copy)
    if not all_rows:
        return
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

#######################################################################
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

            rows.append(flat_row)

    if not rows:
        print("⚠️ Testdaten sind leer.")
        return

    # Schreibe dynamisch alle Keys als Spaltenüberschriften
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Testdaten erfolgreich exportiert nach {path}")

# ✅ NEU: Debug-Funktion zur Ausgabe
def print_test_data_summary():
    print("\n🔍 Überblick über test_data:")
    for testname, numbers in test_data.items():
        print(f"\n📌 {testname} ({len(numbers)} Zahlen):")
        for i, (n, info) in enumerate(numbers.items()):
            print(f"  {n}: {info}")
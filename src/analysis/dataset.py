import os, json, csv
from datetime import datetime
from typing import Dict, Any
from src.primality.test_protocoll import test_data
from src.primality.test_config import *

# creates data directory relative to the src directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# get timestamped filename for saving results
def get_timestamped_filename(basename: str, ext: str = "json"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{basename}.{ext}"

# Entfernt z.B. '(k = 3)' oder andere Klammerzusätze vom Label
def extract_base_label(label: str) -> str:
    if label is None:
        return None
    return label.split(" (")[0].strip()

def export_test_data_to_csv(test_data: dict, filename: str, test_config: dict, numbers_per_test: dict, metadata: dict = None):
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)

    if not test_data:
        print("⚠️ Keine Testdaten vorhanden zum Export.")
        return

    rows = []

    for testname, entries in test_data.items():
        full_label = test_config[testname]["label"]
        base_label = extract_base_label(full_label)

        if base_label not in TEST_GROUPS:
            print(f"⚠️ Kein Gruppeneintrag für Test '{full_label}' (Basis: '{base_label}'), übersprungen.")
            continue

        group = TEST_GROUPS[base_label]
        label = full_label

        # Hole die gültigen Zahlen für diesen Test (nur diese exportieren!)
        valid_numbers = set(numbers_per_test.get(testname, []))

        for number, details in entries.items():
            # Nur wenn die Zahl zu den generierten Zahlen für den Test gehört
            if number not in valid_numbers:
                continue

            flat_row = {
                "Gruppe": group,
                "Test": label,
                "Zahl": number,
                "Ergebnis": details.get("result"),
                "true_prime": details.get("true_prime"),
                "is_error": details.get("is_error"),
                "false_positive": details.get("false_positive"),
                "false_negative": details.get("false_negative"),
                "best_time": f"{details.get('best_time', 0)*1000:.3f} ms",
                "avg_time": f"{details.get('avg_time', 0)*1000:.3f} ms",
                "worst_time": f"{details.get('worst_time', 0)*1000:.3f} ms",
                "std_dev": f"{details.get('std_dev', 0)*1000:.3f} ms",
                "a_values": details.get("a_values"),
                "other_fields": details.get("other_fields"),
                "reason": details.get("reason"),
            }
            rows.append(flat_row)

    if not rows:
        print("⚠️ Testdaten sind leer.")
        return

    # Sortiere nach Gruppe, Test, Zahl
    rows.sort(key=lambda r: (TEST_ORDER.index(extract_base_label(r["Test"])), int(r["Zahl"])))

    # Definierte Spaltenreihenfolge:
    fieldnames = [
        "Gruppe", "Test", "Zahl", "Ergebnis", "true_prime", "is_error",
        "false_positive", "false_negative",
        "best_time", "avg_time", "worst_time", "std_dev",
        "a_values", "other_fields", "reason"
    ]

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Metadaten als Header-Zeilen einfügen
        if metadata:
            writer.writerow(["--- Konfiguration ---"])
            for key, value in metadata.items():
                writer.writerow([key, value])

        writer.writerow([])  # Leerzeile zur Trennung
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Testdaten erfolgreich exportiert: {path}")
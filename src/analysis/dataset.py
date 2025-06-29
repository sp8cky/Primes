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

# formated output of group ranges
def format_group_ranges(group_ranges: dict) -> str:
    return "\n".join(
        f"{group}: n={cfg['n']}, start={cfg['start']}, end={cfg['end']}"
        for group, cfg in group_ranges.items()
    )

def export_test_data_to_csv(test_data: dict, filename: str, test_config: dict, numbers_per_test: dict, metadata: dict = None):
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)

    if not test_data:
        print("⚠️ Keine Testdaten vorhanden zum Export.")
        return

    rows = []

    # Reihenfolge anhand der TEST_GROUPS
    test_order = list(TEST_GROUPS.keys())

    for test_label in test_order:
        group = TEST_GROUPS[test_label]
        # Finde testnamen aus config mit passendem base label
        matching_tests = [
            testname for testname, cfg in test_config.items()
            if extract_base_label(cfg["label"]) == test_label
        ]

        for testname in matching_tests:
            label = test_config[testname]["label"]
            group = TEST_GROUPS[test_label]
            entries = test_data.get(testname, {})  # leeres dict statt None
            valid_numbers = set(numbers_per_test.get(testname, []))

            if not valid_numbers:
                print(f"⚠️ Keine gültigen Daten für Test '{label}', übersprungen.")
                continue

            for number in valid_numbers:
                details = entries.get(number, {})  # leere Details wenn keine Messdaten da

                flat_row = {
                    "Gruppe": group,
                    "Test": label,
                    "Zahl": number,
                    "Ergebnis": details.get("result"),
                    "true_prime": details.get("true_prime"),
                    "is_error": details.get("is_error"),
                    "false_positive": details.get("false_positive"),
                    "false_negative": details.get("false_negative"),
                    "error_count": details.get("error_count", ""),
                    "error_rate": round(details.get("error_rate", 0), 3) if "error_rate" in details else "",
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
        print("⚠️ Keine exportierbaren Zeilen gefunden.")
        return

    # Sortierung optional nach Gruppe, Test, Zahl
    # Mapping: Testname → Reihenindex
    test_rank = {
        (TEST_GROUPS[t], t): i for i, t in enumerate(TEST_GROUPS.keys())
    }

    # Sortieren nach definierter Reihenfolge, dann nach Zahl
    rows.sort(
        key=lambda r: (
            test_rank.get((r["Gruppe"], extract_base_label(r["Test"])), float("inf")),
            int(r["Zahl"])
        )
    )

    fieldnames = [
        "Gruppe", "Test", "Zahl", "Ergebnis", "true_prime", "is_error",
        "false_positive", "false_negative", "error_count", "error_rate",
        "best_time", "avg_time", "worst_time", "std_dev",
        "a_values", "other_fields", "reason"
    ]

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        if metadata:
            writer.writerow(["--- Konfiguration ---"])
            for key, value in metadata.items():
                if key == "group_ranges":
                    # Hier auf den Wert in metadata zugreifen
                    group_ranges_value = value
                    # Falls group_ranges als dict übergeben wird, iterieren:
                    if isinstance(group_ranges_value, dict):
                        for group, cfg in group_ranges_value.items():
                            writer.writerow([
                                "group_range",
                                group,
                                f"n={cfg['n']}, start={cfg['start']}, end={cfg['end']}"
                            ])
                    else:
                        # Falls als String (z.B. format_group_ranges) gespeichert, einfach schreiben
                        writer.writerow([key, value])
                else:
                    writer.writerow([key, value])
            writer.writerow([])

        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Testdaten erfolgreich exportiert: {path}")
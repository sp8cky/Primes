import os, csv
from src.primality.test_protocoll import test_data
from src.primality.test_config import *

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# Entfernt z.B. '(k = 3)' oder andere Klammerzusätze vom Label
def extract_base_label(label: str) -> str:
    if label is None:
        return None
    return label.split(" (")[0].strip()


# Exportiert die Testdaten als CSV-Datei
def export_test_data_to_csv(test_data: dict, filename: str, test_config: dict, numbers_per_test: dict, metadata: dict = None):
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)

    if not test_data:
        print("⚠️ Keine Testdaten vorhanden zum Export.")
        return

    rows = []

    sorted_tests = sorted(
        test_config.items(),
        key=lambda item: (item[1].get("testgroup", ""), item[1].get("label", item[0]))
    )

    for testname, cfg in sorted_tests:
        label = cfg.get("label", testname)
        group = cfg.get("testgroup", "Unbekannte Gruppe")

        valid_numbers = set(numbers_per_test.get(testname, []))
        if not valid_numbers:
            print(f"⚠️ Keine gültigen Daten für Test '{label}', übersprungen.")
            continue

        entries = test_data.get(testname, {})

        for number in valid_numbers:
            details = entries.get(number, {})

            flat_row = {
                "Gruppe": group,
                "Test": label,
                "Zahl": number,
                "Ergebnis": details.get("result"),
                "true_prime": details.get("true_prime"),
                "is_error": details.get("is_error"),
                "false_positive": details.get("false_positive"),
                "false_negative": details.get("false_negative"),
                "error_rate": round(details.get("error_rate") or 0, 3) if "error_rate" in details else "",
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

    # Sortieren nach Gruppe (testgroup) und Zahl
    test_order = list(test_config.keys())
    test_order_map = {name: i for i, name in enumerate(test_order)}

    # Sortiere rows nach Testreihenfolge und Zahl
    rows.sort(
        key=lambda r: (
            test_order_map.get(extract_base_label(r["Test"]), 9999),
            int(r["Zahl"])
        )
    )

    fieldnames = [
        "Gruppe", "Test", "Zahl", "Ergebnis", "true_prime", "is_error",
        "false_positive", "false_negative", "error_rate",
        "best_time", "avg_time", "worst_time", "std_dev",
        "a_values", "other_fields", "reason"
    ]

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        if metadata:
            for key, value in metadata.items():
                if key == "group_ranges" and isinstance(value, dict):
                    for group, cfg in value.items():
                        f.write(f"group_range, {group}, n={cfg['n']}, start={cfg['start']}, end={cfg['end']}\n")
                else:
                    f.write(f"{key}, {value}\n")
            f.write("\n")

        # Durchschnittswerte pro Test berechnen und einfügen
        test_stats = {}

        numeric_fields = {
            "false_positive": int,
            "false_negative": int,
            "error_rate": float,
            "best_time": lambda v: float(v.replace(" ms", "")),
            "avg_time": lambda v: float(v.replace(" ms", "")),
            "worst_time": lambda v: float(v.replace(" ms", "")),
            "std_dev": lambda v: float(v.replace(" ms", "")),
        }

        for row in rows:
            test = row["Test"]
            if test not in test_stats:
                test_stats[test] = {key: [] for key in numeric_fields.keys()}

            for key, parser in numeric_fields.items():
                value = row.get(key, "")
                if value == "" or value is None:
                    continue
                try:
                    parsed = parser(value)
                    test_stats[test][key].append(parsed)
                except ValueError:
                    continue

        for test in sorted(test_stats.keys(), key=lambda t: test_order_map.get(extract_base_label(t), 9999)):
            values = test_stats[test]
            line = f"test_avg, {test}"
            for key in [
                "false_positive", "false_negative", "error_rate",
                "best_time", "avg_time", "worst_time", "std_dev"
            ]:
                data = values[key]
                if data:
                    avg = sum(data) / len(data)
                    if "time" in key:
                        line += f", avg_{key}={avg:.3f} ms"
                    else:
                        line += f", avg_{key}={avg:.3f}"
            f.write(line + "\n")

        f.write("\n")

        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Testdaten erfolgreich exportiert: {path}")
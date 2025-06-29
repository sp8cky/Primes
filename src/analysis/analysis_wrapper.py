from src.primality.tests import *
from src.primality.test_protocoll import *
from src.primality.generate_primes import *
from src.analysis.timing import *
from src.analysis.plot import *
from src.analysis.dataset import *
from src.primality.test_config import *
import time
from typing import List, Dict


# Zeitmessungshilfe
def measure_section(label: str, func, *args, **kwargs):
    print(f"Starte Abschnitt: {label}...")
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    duration = end - start
    print(f"\nAbschnitt '{label}' abgeschlossen in {duration:.2f} Sekunden\n")
    return result



def run_primetest_analysis(
    n_numbers: int = 100,
    num_type: str = 'g',
    start: int = 100_000,
    end: int = 1_000_000,
    test_repeats: int = 2,
    include_tests: list = None,
    prob_test_repeats: list = None,
    seed: int | None = None,
    protocoll: bool = True,
    save_results: bool = True,
    show_plot: bool = True,
    variant: int = 2,  # NEU: 1 = eine Liste für alle Tests, 2 = eigene Zahlen pro Test
    allow_partial_numbers = False,
    group_ranges: Dict[str, Dict[str, int]] = None
) -> Dict[str, List[Dict]]:

    if seed is not None: random.seed(seed)

    # Test-Konfiguration laden
    test_config = get_test_config(include_tests, prob_test_repeats)

    # Zahlengenerierung
    if variant == 1:
        # 1. Eine gemeinsame Liste für alle Tests
        print(f"Generiere eine gemeinsame Liste von {n_numbers} Zahlen vom Typ '{num_type}' im Bereich [{start}, {end}]...")
        numbers = measure_section(
            f"Zahlengenerierung für alle Tests ({num_type})",
            generate_numbers_for_test,
            n_numbers, start, end, num_type
        )
        numbers_per_test = {test_name: numbers for test_name in test_config.keys()}
    elif variant == 2:
        # 2. Eigene Zahlen pro Test, wie bisher
        numbers_per_test = measure_section(
            "Zahlengenerierung pro Test",
            generate_numbers_per_group,
            n_numbers, start, end, test_config, allow_partial_numbers=allow_partial_numbers, group_ranges=group_ranges
        )
    else:
        raise ValueError("variant muss 1 oder 2 sein")

    # 3. Ausgabe der generierten Zahlen
    for test_name, nums in numbers_per_test.items():
        print(f"→ {test_name}: {len(nums)} Zahlen: {nums}")

    # 4. Testdaten initialisieren für alle Zahlen
    all_numbers = set()
    for nums in numbers_per_test.values():
        all_numbers.update(nums)
    measure_section("Initialisiere Testdaten", init_dictionary_fields, list(all_numbers))

    # Funktionsabbildungen
    runtime_functions = {}
    protocol_functions = {}
    for test_name, config in test_config.items():
        runtime_functions[test_name] = config["runtime_function"]
        protocol_functions[test_name] = config["protocol_function"]

    # Zeitmessung
    datasets = measure_section("Laufzeitmessung", lambda: {
        test_name: (
            print(f"🔍 Messe Laufzeit für: {test_name}") or
            measure_runtime(
                runtime_functions[test_name],
                numbers_per_test[test_name],
                test_name,
                label=(
                    f"{test_name} (k = {test_config[test_name]['prob_test_repeats']})"
                    if "prob_test_repeats" in test_config[test_name]
                    else test_name
                ),
                runs_per_n=test_repeats
            )
        )
        for test_name in runtime_functions
    })

    # Protokolle
    if protocoll:
        measure_section("Protokoll-Tests", lambda: [
            protocol_functions[test_name](n)
            for test_name in protocol_functions
            for n in numbers_per_test[test_name]
        ])

    # Fehleranalyse
    measure_section("Fehleranalyse", lambda: analyze_errors(test_data))

    # Plotten
    if show_plot:
        # Nur Datensätze mit mind. einem Eintrag nehmen
        valid_plot_entries = [
        (test_name, data)
        for test_name, data in datasets.items()
            if isinstance(data, list) and len(data) > 0
        ]

        plot_data = {
            "n_values": [[entry["n"] for entry in data] for _, data in valid_plot_entries],
            "avg_times": [[entry["avg_time"] for entry in data] for _, data in valid_plot_entries],
            "std_devs": [[entry["std_dev"] for entry in data] for _, data in valid_plot_entries],
            "best_times": [[entry["best_time"] for entry in data] for _, data in valid_plot_entries],
            "worst_times": [[entry["worst_time"] for entry in data] for _, data in valid_plot_entries],
            "labels": [data[0]["label"] for _, data in valid_plot_entries],
            "colors": ["#b41f1f", "#d62728", "#e6550d", "#ff7f0e", "#bcbd22", "#2ca02c", "#31a354", "#637939", "#8c6d31", "#17becf", "#3182bd", "#393b79", "#756bb1", "#9467bd", "#e377c2", "#7b4173","#843c39", "#72302e", "#8c564b", "#636363", "#7f7f7f"]
        }
        measure_section("Plotten", plot_runtime_grouped,
            n_lists=plot_data["n_values"],
            time_lists=plot_data["avg_times"],
            std_lists=plot_data["std_devs"],
            best_lists=plot_data["best_times"],
            worst_lists=plot_data["worst_times"],
            labels=plot_data["labels"],
            colors=plot_data["colors"],
            figsize=(18, 9),
            total_numbers=n_numbers,
            runs_per_n=test_repeats,
            group_ranges=group_ranges
        )

    # CSV-Export
    if save_results:
        filename = get_timestamped_filename("test-data", "csv")
        measure_section("Exportiere CSV", lambda: export_test_data_to_csv(
            test_data,
            filename,
            test_config=test_config,
            numbers_per_test=numbers_per_test,
            metadata={
                "n_numbers": n_numbers,
                "start": start,
                "end": end,
                "test_repeats": test_repeats,
                "number_type": num_type,
                "variant": variant,
                "group_ranges": group_ranges
            }
        ))
    return datasets

# Hauptaufruf
if __name__ == "__main__":
    run_tests = ["Ramzy", "Rao"]
    repeat_tests = [5,5,5]
    group_ranges={
        "Probabilistische Tests":   {"n": 10, "start": 100, "end": 10_000},
        "Lucas-Tests":              {"n": 10, "start": 100, "end": 10_000},
        "Langsame Tests":           {"n": 10, "start": 100, "end": 10_000},
        "Proth-Tests":              {"n": 10, "start": 100, "end": 10_000},
        "Pocklington-Tests":        {"n": 10, "start": 100, "end": 10_000},
        "Rao":                {"n": 10, "start": 100, "end": 10_000},
        "Ramzy":              {"n": 10, "start": 100, "end": 10_000},
        "Fermat-Zahlen":            {"n": 5, "start": 0, "end": 10_000},
        "Mersenne-Zahlen":          {"n": 5, "start": 2, "end": 10_000},
    }


    run_primetest_analysis(
        n_numbers=10,
        num_type='p',
        start=10,
        end=10000,
        test_repeats=10,
        #include_tests=run_tests,
        prob_test_repeats=repeat_tests,
        seed=42,
        protocoll=True,
        save_results=True,
        show_plot=True,
        allow_partial_numbers = True,
        variant=2,
        group_ranges=group_ranges
    )
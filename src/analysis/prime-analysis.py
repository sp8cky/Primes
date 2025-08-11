from src.primality.tests import *
from src.primality.constants import *
from src.primality.test_protocoll import *
from src.primality.generate_primes import *
from src.analysis.timing import *
from src.analysis.plot import *
from src.analysis.dataset import *
from src.primality.test_config import *
import time
from typing import List, Dict
from datetime import datetime


# Zeitmessungshilfe
def measure_section(label: str, func, *args, **kwargs):
    #print(f"Starte Abschnitt: {label}...")
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    duration = end - start
    print(f"Abschnitt '{label}' abgeschlossen in {duration:.2f} Sekunden")
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
    group_ranges: Dict[str, Dict[str, int]] = None,
    custom_group_numbers: Dict[str, List[int]] = None
) -> Dict[str, List[Dict]]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Test-Konfiguration laden
    test_config = get_test_config(include_tests, prob_test_repeats, global_seed=seed)

    # Zahlengenerierung
    if variant == 1:
        print(f"Generiere eine gemeinsame Liste von {n_numbers} Zahlen vom Typ '{num_type}' im Bereich [{start}, {end}]...")
        # Hier nochmal explizit num_type ausgeben
        if num_type.startswith("g"):
            if ":" in num_type:
                ratio = num_type.split(":")[1]
                print(f"NumType ist 'g' mit prime_ratio = {ratio}")
            else:
                print(f"NumType ist 'g' mit default prime_ratio = 0.5")
        else:
            print(f"NumType ist '{num_type}' (kein spezielles prime_ratio)")

        numbers = measure_section(
            f"Zahlengenerierung für alle Tests ({num_type})",
            generate_numbers_for_test,
            n_numbers, start, end, num_type
        )
        numbers_per_test = {test_name: numbers for test_name in test_config.keys()}
    elif variant == 2:
        print(f"Generiere eigene Zahlen pro Test, num_type='{num_type}'")
        if num_type.startswith("g"):
            if ":" in num_type:
                ratio = num_type.split(":")[1]
                print(f"NumType ist 'g' mit prime_ratio = {ratio}")
            else:
                print(f"NumType ist 'g' mit default prime_ratio = 0.5")
        else:
            print(f"NumType ist '{num_type}' (kein spezielles prime_ratio)")

        numbers_per_test = {}

        # 1. Benutzerdefinierte Gruppen übernehmen
        if custom_group_numbers:
            for group_name, number_list in custom_group_numbers.items():
                assigned = assign_custom_numbers_to_group(group_name, number_list, test_config)
                numbers_per_test.update(assigned)

        # 2. Für übrige Gruppen: automatische Generierung
        auto_generated = measure_section(
            "Zahlengenerierung pro Test",
            generate_numbers_per_group,
            n_numbers,
            start,
            end,
            test_config,
            allow_partial_numbers=allow_partial_numbers,
            group_ranges=group_ranges,
            seed=seed,
            num_type=num_type
        )

        # 3. Manuelle Gruppen nicht überschreiben
        for test_name, number_list in auto_generated.items():
            if test_name not in numbers_per_test:
                print(f"Test '{test_name}' wurde automatisch generiert: {len(number_list)} Zahlen")
                numbers_per_test[test_name] = number_list
    else:
        raise ValueError("variant muss 1 oder 2 sein")

    # 3. Ausgabe der generierten Zahlen
    for test_name, nums in numbers_per_test.items():
        print(f"→ {test_name}: {len(nums)} Zahlen: {nums}")

    # 4. Testdaten initialisieren für alle Zahlen
    for test_name, numbers in numbers_per_test.items():
        measure_section(f"Initialisiere Testdaten für {test_name}", init_dictionary_fields, numbers, test_name)

    # Funktionsabbildungen
    runtime_functions = {}
    protocol_functions = {}
    for test_name, config in test_config.items():
        runtime_functions[test_name] = config["runtime_function"]
        protocol_functions[test_name] = config["protocol_function"]


    # Zeitmessung
    datasets = measure_section("Laufzeitmessung", lambda: {
        test_name: (
            print(f"🔍 Messe Laufzeit für: {test_name} mit n = {numbers_per_test[test_name]}") or
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
            protocol_functions[test_name](n, seed=seed)
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
            "colors": ["#b41f1f", "#ff7f0e", "#bcbd22", "#3182bd", "#2ca02c",
                       "#31a354", "#d62728", "#e6550d",
                       "#17becf", "#393b79", "#637939",
                       "#8c6d31", "#756bb1",
                       "#9467bd", "#e377c2", "#7b4173", "#843c39", "#72302e", "#8c564b", "#636363", "#7f7f7f", "#1f77b4", "#ff1493"],
        }
        measure_section("Plotten", plot_runtime,
            n_lists=plot_data["n_values"],
            time_lists=plot_data["avg_times"],
            std_lists=plot_data["std_devs"],
            best_lists=plot_data["best_times"],
            worst_lists=plot_data["worst_times"],
            labels=plot_data["labels"],
            colors=plot_data["colors"],
            figsize=(24, 14),
            total_numbers=n_numbers,
            runs_per_n=test_repeats,
            group_ranges=group_ranges,
            seed=seed,
            timestamp=timestamp,
            variant=variant,
            start=start,
            end=end,
            custom_xticks=custom_ticks,
            number_type=num_type,
            #filename=f"d1-plot-seed{seed}-v{variant}.png"
        )
        # Gruppierten Plot aufrufen
        measure_section("Plots",
            plot_grouped_all,
                datasets=datasets,
                test_data=test_data,
                group_ranges=group_ranges,
                timestamp=timestamp,
                seed=seed,
                variant=variant,
                runs_per_n=test_repeats,
                prob_test_repeats=repeat_prob_tests,
                figsize=(24, 16),
                number_type=num_type,
                #filename1=f"d1-group_Probabilistisch-graph-s{seed}-v{variant}.png",
                #filename2=f"d1-group_Probabilistisch-stats-s{seed}-v{variant}.png"
        )

    # CSV-Export
    if save_results:
        #filename = f"d1-testdata-seed{seed}-v{variant}.csv"
        filename = f"{timestamp}-test-data-seed{seed}-v{variant}.csv"
        measure_section("Exportiere CSV", lambda: export_test_data_to_csv(
            test_data,
            filename = filename,
            test_config=test_config,
            numbers_per_test=numbers_per_test,
            metadata={
                "n_numbers": n_numbers,
                "start": start,
                "end": end,
                "test_repeats": test_repeats,
                "number_type": num_type,
                "variant": variant,
                "seed": seed, 
                "group_ranges": group_ranges
            }
        ))

    return datasets

# Hauptaufruf
if __name__ == "__main__":
    #custom_group_numbers = {"Probabilistische Tests": [341, 561, 645, 1105, 1729, 2047, 2465, 2701, 2821, 6601]}
    pseudopimes = {"Probabilistische Tests": [341, 561, 645, 1105, 1387, 1729, 1905, 2047, 2465, 2701, 2821, 3277, 4033, 4369, 4371, 4681, 5461, 6601, 7957, 8321,  8481, 8911, 10261, 10585, 11305, 12801, 13741, 13747, 13981, 14491,
        15709, 15841, 16705, 18705, 29341, 41041, 42799, 46657, 49141, 52633, 57727, 62745, 63973, 65281, 74665, 75361, 80581, 85489, 88357, 90751, 101101, 104653, 115921, 126217, 130561, 162401, 172081, 188461, 196093,
        220729, 252601, 256999, 271951, 278545, 280601, 294409, 314821, 334153, 340561, 357761, 390937, 399001, 410041, 449065, 458989, 476971, 486737, 488881, 490841,
        512461, 530881, 552721, 656601, 825265, 1024651, 1033669, 1045505, 1152271, 1193221, 1299961, 1487161, 1524601, 1588261, 1773289, 1857241, 1909001, 2044501, 2096125]}
    
    h1 = 10**2
    k1 = 10**3
    k10 = 10**4
    k100 = 10**5
 
    custom_ticks = [1, h1, k1, k10, k100]
    #run_tests = ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen"]
    run_tests = ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen", 
              "Initial Lucas", "Lucas", "Optimized Lucas", 
              "Wilson", "AKS10", "Pepin", 
              "Lucas-Lehmer", "Proth", "Proth Variant", 
              "Pocklington", "Optimized Pocklington", "Optimized Pocklington Variant", "Generalized Pocklington", #"Grau", "Grau Probability"
              "Rao", "Ramzy"]
    #repeat_prob_tests = [1,1,1]
    #repeat_prob_tests = [2,2,2]
    repeat_prob_tests = [3,3,3]
    #repeat_prob_tests = [4,4,4]
    #repeat_prob_tests = [5,5,5]

    my_group_ranges={ 
        "Probabilistisch":      {"n": 100, "start": 1,  "end": k100,    "xticks": [1, h1, k1, k10, k100]},
        "Lucas":                {"n": 100, "start": 1,  "end": k100,   "xticks": [1, h1, k1, k10, k100]},
        "Langsam":              {"n": 100, "start": 1,  "end": k100,    "xticks": [1, h1, k1, k10, k100]},
        "Proth-Tests":          {"n": 100, "start": 1,  "end": k100,  "xticks": [1, h1, k1, k10, k100]},
        "Pocklington-Tests":    {"n": 100, "start": 1,  "end": k100,    "xticks": [1, h1, k1, k10, k100]},
        "Rao":                  {"n": 100, "start": 1,  "end": k100,    "xticks": [1, h1, k1, k10, k100]},
        "Ramzy":                {"n": 100, "start": 1,  "end": k100,    "xticks": [1, h1, k1, k10, k100]},
        "Fermat-Zahlen":        {"n": 100, "start": 1,  "end": k100,   "xticks": [1, h1, k1, k10, k100]},
        "Mersenne-Zahlen":      {"n": 100, "start": 1,  "end": k100,  "xticks": [1, h1, k1, k10, k100]},
    }
# TODO: DURCHLAUF 5


    run_primetest_analysis(
        n_numbers=100,
        num_type='g:0.5',
        start=1,
        end=k100,
        test_repeats=10,
        include_tests=run_tests,
        prob_test_repeats=repeat_prob_tests,
        #seed=10176,
        seed=random.randint(1000, 10000),
        protocoll=True,
        save_results=True,
        show_plot=True,
        variant=2,
        allow_partial_numbers = True,
        group_ranges=my_group_ranges,
        #custom_group_numbers=pseudopimes
    )
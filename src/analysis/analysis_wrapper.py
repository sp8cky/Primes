from src.primality.tests import *
from src.primality.test_protocoll import *
from src.analysis.timing import measure_runtime
from src.analysis.plot import plot_runtime
from src.analysis.dataset import *
from src.primality.test_protocoll import test_data
import random
from sympy import isprime, primerange
from typing import List, Dict
from functools import partial
import time

# Zeitmessungshilfe
def measure_section(label: str, func, *args, **kwargs):
    print(f"Starte Abschnitt: {label}...")
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    duration = end - start
    print(f"\nAbschnitt '{label}' abgeschlossen in {duration:.2f} Sekunden\n")
    return result

def run_with_prints(test_fn, test_name, n, test_repeats):
    print(f"  n = {n}: ", end="")
    results = []
    for i in range(test_repeats):
        print(f"⏱️{i+1} ", end="", flush=True)
        results.append(test_fn(n))
    return results[-1]

# Generiert Zahlen im Bereich
def generate_numbers(n: int, start: int = 100, end: int = 1000, num_type: str = 'g') -> List[int]:
    if num_type not in ['p', 'z', 'g']: raise ValueError("num_type muss 'p', 'z' oder 'g' sein")

    primes = list(primerange(start, end))
    composites = [x for x in range(start, end) if not isprime(x)]
    if num_type == 'p':
        numbers = random.sample(primes, min(n, len(primes)))
    elif num_type == 'z':
        numbers = random.sample(composites, min(n, len(composites)))
    else:
        p = random.sample(primes, min(n, len(primes)))
        c = random.sample(composites, min(n, len(composites)))
        numbers = random.sample(p + c, min(n, len(p) + len(c)))
    return sorted(numbers)

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
    show_plot: bool = True
) -> Dict[str, List[Dict]]:

    all_available_tests = [
        "Fermat", "Wilson", "Initial Lucas", "Lucas", "Optimized Lucas", "Pepin", "Lucas-Lehmer", "Proth", "Proth Variant", "Pocklington", "Optimized Pocklington", 
        "Optimized Pocklington Variant", "Generalized Pocklington", "Grau", "Grau Probability", "Miller-Rabin", "Solovay-Strassen", "AKS"]

    if include_tests is None: include_tests = all_available_tests

    prob_tests = ["Fermat", "Miller-Rabin", "Solovay-Strassen"]
    default_repeats = [3, 3, 3]
    prob_test_repeats = prob_test_repeats if prob_test_repeats is not None else default_repeats

    test_config = {}
    for test in include_tests:
        if test in prob_tests:
            idx = prob_tests.index(test)
            test_config[test] = {"prob_test_repeats": prob_test_repeats[idx]}
        else:
            test_config[test] = {}

    # Zahlen generieren
    numbers = measure_section("Zahlengenerierung", generate_numbers, n=n_numbers, start=start, end=end, num_type=num_type)
    print(f"Generierte {len(numbers)} Zahlen im Bereich {start}-{end} ({num_type}): {numbers[:10]}")

    # Testdaten initialisieren
    measure_section("Initialisiere Testdaten", init_dictionary_fields, numbers)

    # Funktionsabbildungen
    runtime_functions = {}
    protocol_functions = {}
    for test, cfg in test_config.items():
        if test == "Fermat":
            runtime_functions[test] = partial(fermat_test, k=cfg["prob_test_repeats"])
            protocol_functions[test] = partial(fermat_test_protocoll, k=cfg["prob_test_repeats"])
        elif test == "Wilson":
            runtime_functions[test] = wilson_criterion
            protocol_functions[test] = wilson_criterion_protocoll
        elif test == "Initial Lucas":
            runtime_functions[test] = initial_lucas_test
            protocol_functions[test] = initial_lucas_test_protocoll
        elif test == "Lucas":
            runtime_functions[test] = lucas_test
            protocol_functions[test] = lucas_test_protocoll
        elif test == "Optimized Lucas":
            runtime_functions[test] = optimized_lucas_test
            protocol_functions[test] = optimized_lucas_test_protocoll
        elif test == "Pepin":
            runtime_functions[test] = pepin_test
            protocol_functions[test] = pepin_test_protocoll
        elif test == "Lucas-Lehmer":
            runtime_functions[test] = lucas_lehmer_test
            protocol_functions[test] = lucas_lehmer_test_protocoll
        elif test == "Proth":
            runtime_functions[test] = proth_test
            protocol_functions[test] = proth_test_protocoll
        elif test == "Proth Variant":
            runtime_functions[test] = proth_test_variant
            protocol_functions[test] = proth_test_variant_protocoll
        elif test == "Pocklington":
            runtime_functions[test] = pocklington_test
            protocol_functions[test] = pocklington_test_protocoll
        elif test == "Optimized Pocklington":
            runtime_functions[test] = optimized_pocklington_test
            protocol_functions[test] = optimized_pocklington_test_protocoll
        elif test == "Optimized Pocklington Variant":
            runtime_functions[test] = optimized_pocklington_test_variant
            protocol_functions[test] = optimized_pocklington_test_variant_protocoll
        elif test == "Generalized Pocklington":
            runtime_functions[test] = generalized_pocklington_test
            protocol_functions[test] = generalized_pocklington_test_protocoll
        elif test == "Grau":
            runtime_functions[test] = grau_test
            protocol_functions[test] = grau_test_protocoll
        elif test == "Grau Probability":
            runtime_functions[test] = grau_probability_test
            protocol_functions[test] = grau_probability_test_protocoll
        elif test == "Miller-Rabin":
            runtime_functions[test] = partial(miller_selfridge_rabin_test, k=cfg["prob_test_repeats"])
            protocol_functions[test] = partial(miller_selfridge_rabin_test_protocoll, k=cfg["prob_test_repeats"])
        elif test == "Solovay-Strassen":
            runtime_functions[test] = partial(solovay_strassen_test, k=cfg["prob_test_repeats"])
            protocol_functions[test] = partial(solovay_strassen_test_protocoll, k=cfg["prob_test_repeats"])
        elif test == "AKS":
            runtime_functions[test] = aks_test
            protocol_functions[test] = aks_test_protocoll


    # Zeitmessung MIT PRINTS
    """datasets = measure_section("Laufzeitmessung", lambda: {
    test_name: (
        measure_runtime(
            lambda n: (
                print(f"\n🔍Starte {test_name} für n = {n}: ", end=""),
                [print(i + 1, end=" ", flush=True) for i in range(test_repeats)],
                test_fn(n)
            )[-1],
            numbers,
            test_name,
            label=test_name + (f" (k={test_config[test_name]['prob_test_repeats']})" if "prob_test_repeats" in test_config[test_name] else ""),
            runs_per_n=1
        )
    )
    for test_name, test_fn in runtime_functions.items()
    })"""

    # Protokolle mit print
    """if protocoll:
        measure_section("Protokoll-Tests", lambda: [
            print(f"\n▶️ Starte {test_name} für n = {n}: ", end="") or
            [print(i+1, end=" ", flush=True) for i in range(test_repeats)] or
            print() or
            test_fn(n)
            for test_name, test_fn in protocol_functions.items()
            for n in numbers
        ])"""


    # Zeitmessung
    datasets = measure_section("Laufzeitmessung", lambda: {
    test_name: (
        print(f"🔍 Messe Laufzeit für: {test_name}") or
        measure_runtime(
            test_fn,
            numbers,
            test_name,
            label=test_name + (f" (k={test_config[test_name]['prob_test_repeats']})" if "prob_test_repeats" in test_config[test_name] else ""),
            runs_per_n=test_repeats
        )
    )
    for test_name, test_fn in runtime_functions.items()
    })
    
    
    # protkolle ohne print
    if protocoll:
        measure_section("Protokoll-Tests", lambda: [
            test_fn(n) for test_name, test_fn in protocol_functions.items() for n in numbers
        ])

    # CSV-Export
    if save_results:
        measure_section("Exportiere CSV", export_test_data_to_csv, test_data, get_timestamped_filename("tests", "csv"))

    # Plotten
    if show_plot:
        plot_data = {
            "n_values": [[entry["n"] for entry in data] for data in datasets.values()],
            "avg_times": [[entry["avg_time"] for entry in data] for data in datasets.values()],
            "std_devs": [[entry["std_dev"] for entry in data] for data in datasets.values()],
            "best_times": [[entry["best_time"] for entry in data] for data in datasets.values()],
            "worst_times": [[entry["worst_time"] for entry in data] for data in datasets.values()],
            "labels": [data[0]["label"] for data in datasets.values()],
            "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#31a354", "#756bb1", "#e6550d", "#636363"]
        }
        measure_section("Plotten", plot_runtime,
            n_lists=plot_data["n_values"],
            time_lists=plot_data["avg_times"],
            std_lists=plot_data["std_devs"],
            best_lists=plot_data["best_times"],
            worst_lists=plot_data["worst_times"],
            labels=plot_data["labels"],
            colors=plot_data["colors"],
            figsize=(7, 7)
        )

    return datasets

# Hauptaufruf
if __name__ == "__main__":
    run_tests = ["Fermat"]
    repeat_tests = [5,5,5]
    run_primetest_analysis(
        n_numbers=2,
        num_type='p',
        start=1_000,
        end=10_000,
        test_repeats=10,
        #include_tests=run_tests,
        prob_test_repeats=repeat_tests,
        #seed=42,
        protocoll=True,
        save_results=True,
        show_plot=True
    )
    
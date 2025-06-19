from src.primality.tests import *
from src.analysis.timing import measure_runtime
from src.analysis.plot import plot_runtime
from src.analysis.dataset import *
from src.primality.tests import test_data
import random
from sympy import isprime, primerange
from typing import List, Dict
from functools import partial

# generates numbers in a given range, either primes, composites or mixed
def generate_numbers(n: int, start: int = 100, end: int = 1000, num_type: str = 'g') -> List[int]:
    if num_type not in ['p', 'z', 'g']:
        raise ValueError("num_type muss 'p', 'z' oder 'g' sein")
    
    primes = list(primerange(start, end))
    composites = [x for x in range(start, end) if not isprime(x)]
    if num_type == 'p':
        numbers = random.sample(primes, min(n, len(primes)))
    elif num_type == 'z':
        numbers = random.sample(composites, min(n, len(composites)))
    else:  # 'g'
        p = random.sample(primes, min(n, len(primes)))
        c = random.sample(composites, min(n, len(composites)))
        numbers = random.sample(p + c, min(n, len(p) + len(c)))
    return sorted(numbers)

def run_primetest_analysis(    
    n_numbers: int = 100,
    num_type: str = 'g',
    start: int = 100_000,
    end: int = 1_000_000,
    include_tests: list = None,
    repeats: list = None,
    save_results: bool = True,
    show_plot: bool = True
) -> Dict[str, List[Dict]]:
    
    # Alle verfügbaren Tests (nur Namen, Funktionen kommen später)
    all_available_tests = [
        "Fermat", "Wilson", "Initial Lucas", "Lucas",
        "Optimized Lucas", "Pepin", "Lucas-Lehmer", "Proth", "Pocklington", "Optimized Pocklington", "Proth Variant", "Optimized Pocklington Variant", "Generalized Pocklington", "Grau", "Grau Probability", "Miller-Rabin", "Solovay-Strassen", "AKS"
    ]
    
    # Wenn keine Tests übergeben wurden, nutze alle
    if include_tests is None:
        include_tests = all_available_tests

    # Konfiguriere Wiederholungen (Standard: 3 für alle probabilistischen Tests)
    prob_tests = ["Fermat", "Miller-Rabin", "Solovay-Strassen"]
    default_repeats = [3, 3, 3]
    repeats = repeats if repeats is not None else default_repeats
    
    # Erzeuge Konfigurations-Dictionary
    test_config = {}
    for test in include_tests:
        if test in prob_tests:
            idx = prob_tests.index(test)
            test_config[test] = {"repeats": repeats[idx]}
        else:
            test_config[test] = {}

    # GENERATION
    numbers = generate_numbers(n=n_numbers, start=start, end=end, num_type=num_type)
    print(f"Generating {len(numbers)} test numbers for prime criteria (Typ '{num_type}')")
    
    # INITIALIZE DATA STRUCTURES 
    init_all_test_data(numbers)

    # MAPPING DER FUNKTIONEN
    test_functions = {}
    for test, cfg in test_config.items():
        if test == "Fermat":
            test_functions[test] = partial(fermat_test, k=cfg["repeats"])
        elif test == "Wilson":
            test_functions[test] = wilson_criterion
        elif test == "Initial Lucas":
            test_functions[test] = initial_lucas_test
        elif test == "Lucas":
            test_functions[test] = lucas_test
        elif test == "Optimized Lucas":
            test_functions[test] = optimized_lucas_test
        elif test == "Pepin":
            test_functions[test] = pepin_test
        elif test == "Lucas-Lehmer":
            test_functions[test] = lucas_lehmer_test
        elif test == "Proth":
            test_functions[test] = proth_test
        elif test == "Pocklington":
            test_functions[test] = pocklington_test
        elif test == "Optimized Pocklington":
            test_functions[test] = optimized_pocklington_test
        elif test == "Proth Variant":
            test_functions[test] = proth_test_variant
        elif test == "Optimized Pocklington Variant":
            test_functions[test] = optimized_pocklington_test_variant
        elif test == "Generalized Pocklington":
            test_functions[test] = generalized_pocklington_test
        elif test == "Grau":
            test_functions[test] = grau_test
        elif test == "Grau Probability":
            test_functions[test] = grau_probability_test
        elif test == "Miller-Rabin":
            test_functions[test] = partial(miller_selfridge_rabin_test, k=cfg["repeats"])
        elif test == "Solovay-Strassen":
            test_functions[test] = partial(solovay_strassen_test, k=cfg["repeats"])
        elif test == "AKS":
            test_functions[test] = aks_test

    # MEASURE 
    print("Measuring runtimes...")
    datasets = {}
    for test_name, test_fn in test_functions.items():
        label = test_name
        if "repeats" in test_config[test_name]:
            label += f" (k={test_config[test_name]['repeats']})"
        datasets[test_name] = measure_runtime(test_fn, numbers, test_name, label=label)
    
    # CALL PROTOCOL
    print_test_protocoll(numbers, datasets, selected_tests=include_tests)

    # SAVE RESTULTS
    if save_results:
        #save_json(datasets, get_timestamped_filename("criteria", "json"))
        #export_to_csv(datasets, get_timestamped_filename("criteria", "csv"))
        export_test_data_to_csv(test_data, get_timestamped_filename("tests", "csv"))

    # PLOTTING
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
        plot_runtime(
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





# CALL ###################################################
if __name__ == "__main__":

    #run_tests = ["Fermat", "Lucas", "Proth", "Pocklington", "Optimized Pocklington"]
    repeat_tests = [3, 5, 3]  # Fermat, MSRT, SST

    results = run_primetest_analysis(
        n_numbers=2,
        num_type='p',
        start=1000, # 100_000,
        end=10_000, #1_000_000,
        #include_tests=run_tests,
        repeats=repeat_tests,
        save_results=True,
        show_plot=True
    )
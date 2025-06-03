from src.primality.criteria import *
from src.primality.tests import *
from src.primality.criteriaProtocoll import *
from src.primality.testsProtocoll import *
from src.analysis.timing import measure_runtime
from src.analysis.plot import plot_runtime
from src.analysis.dataset import *
import random, json, os, csv
from sympy import isprime, primerange
from typing import List, Dict
from datetime import datetime

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

def run_prime_criteria_analysis(n_numbers: int = 100, num_type: str = 'g', start: int = 100_000, end: int = 1_000_000, fermat_k: int = 5,repeats: int = 5, save_results: bool = True, show_plot: bool = True) -> Dict[str, List[Dict]]:
    """
    Führt komplette Primzahltest-Analyse durch (Generierung, Messung, Protokoll, Plot).
    Parameter:
    - n_numbers: Anzahl der Testzahlen
    - num_type: 'p' (Prim), 'z' (Zusammengesetzt), 'g' (gemischt)
    - start/end: Zahlenbereich
    - fermat_k: Iterationen für Fermat-Test
    - repeats: Wiederholungen pro Messung
    - save_results: Ob Ergebnisse gespeichert werden sollen
    - show_plot: Ob Plot angezeigt werden soll
    Returns:
    - Dictionary mit allen Messdaten
    """
    
    # GENERATION
    numbers = generate_numbers(n=n_numbers, start=start, end=end, num_type=num_type)
    print(f"Generiere {len(numbers)} Testzahlen (Typ '{num_type}')")
    
    # MEASURE
    print("Starte Laufzeitmessungen...")
    datasets = {
        "Fermat": measure_runtime(lambda n: fermat_criterion(n, fermat_k), numbers, f"Fermat (k={fermat_k})", repeat=repeats),
        "Wilson": measure_runtime(wilson_criterion, numbers, "Wilson", repeat=repeats),
        "Initial Lucas": measure_runtime(initial_lucas_test, numbers, "Initial Lucas", repeat=repeats),
        "Lucas": measure_runtime(lucas_test, numbers, "Lucas", repeat=repeats),
        "Optimized Lucas": measure_runtime(optimized_lucas_test, numbers, "Optimized Lucas", repeat=repeats)
    }
    
    # SAVE RESTULTS
    if save_results:
        save_json(datasets, get_timestamped_filename("criteria", "json"))
        export_to_csv(datasets, get_timestamped_filename("criteria", "csv"))
    
    # CALL PROTOCOL
    criteria_protocoll(numbers, datasets)

    
    # CREATE DATASETS FOR PLOTTING
    if show_plot:
        plot_data = {
            "n_values": [[entry["n"] for entry in data] for data in datasets.values()],
            "avg_times": [[entry["avg_time"] for entry in data] for data in datasets.values()],
            "std_devs": [[entry["std_dev"] for entry in data] for data in datasets.values()],
            "best_times": [[entry["best_time"] for entry in data] for data in datasets.values()],
            "worst_times": [[entry["worst_time"] for entry in data] for data in datasets.values()],
            "labels": [data[0]["label"] for data in datasets.values()],
            "colors": ["orange", "red", "purple", "blue", "green"]
        }
        plot_runtime(
            n_lists=plot_data["n_values"],
            time_lists=plot_data["avg_times"],
            std_lists=plot_data["std_devs"],
            best_lists=plot_data["best_times"],
            worst_lists=plot_data["worst_times"],
            labels=plot_data["labels"],
            colors=plot_data["colors"],
            figsize=(7, 5)
        )
    return datasets


def run_prime_test_analysis(n_numbers: int = 100, num_type: str = 'g', start: int = 100_000, end: int = 1_000_000, msr_rounds: int = 5, ss_rounds: int = 5, repeats: int = 3, save_results: bool = True, show_plot: bool = True) -> Dict[str, List[Dict]]:

    # GENERATION
    numbers = generate_numbers(n=n_numbers, start=start, end=end, num_type=num_type)
    print(f"Generiere {len(numbers)} Testzahlen (Typ '{num_type}')")
    
    # MEASURE
    print("Starte Laufzeitmessungen für Primzahltests...")
    datasets = {
        "Miller–Rabin": measure_runtime(lambda n: miller_selfridge_rabin_test(n, msr_rounds),numbers,f"Miller–Rabin (r={msr_rounds})",repeat=repeats),
        "Solovay–Strassen": measure_runtime(lambda n: solovay_strassen_test(n, ss_rounds), numbers,f"Solovay–Strassen (r={ss_rounds})",repeat=repeats),
        "AKS": measure_runtime(aks_test, numbers,"AKS", repeat=repeats)
    }
    
    # SAVE RESULTS
    if save_results:
        save_json(datasets, get_timestamped_filename("tests", "json"))
        export_to_csv(datasets, get_timestamped_filename("tests", "csv"))

    # PROTOCOL
    tests_protocoll(numbers, "ms", datasets)


    # PLOTTING
    if show_plot:
        plot_data = {
            "n_values": [[entry["n"] for entry in data] for data in datasets.values()],
            "avg_times": [[entry["avg_time"] for entry in data] for data in datasets.values()],
            "std_devs": [[entry["std_dev"] for entry in data] for data in datasets.values()],
            "best_times": [[entry["best_time"] for entry in data] for data in datasets.values()],
            "worst_times": [[entry["worst_time"] for entry in data] for data in datasets.values()],
            "labels": [data[0]["label"] for data in datasets.values()],
            "colors": ["red", "blue", "green"]
        }
        plot_runtime(
            n_lists=plot_data["n_values"],
            time_lists=plot_data["avg_times"],
            std_lists=plot_data["std_devs"],
            best_lists=plot_data["best_times"],
            worst_lists=plot_data["worst_times"],
            labels=plot_data["labels"],
            colors=plot_data["colors"],
            figsize=(7, 5)
        )
    
    return datasets


################################################
# CALL Criteria ##
if __name__ == "__main__":
    random.seed(42)  # Für Reproduzierbarkeit
    #criteria = run_prime_criteria_analysis(n_numbers=2, num_type='p', start=1000, end=10000, fermat_k=3, repeats=3, save_results=False, show_plot=True)
    tests = run_prime_test_analysis(n_numbers=5, num_type='p', start=10, end=100, repeats=3, save_results=False, show_plot=True)

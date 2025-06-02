from src.primality.criteria import *
from src.primality.tests import *
from src.primality.criteriaProtocoll import *
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

def run_prime_criteria_analysis(n_numbers: int = 100,num_type: str = 'g',start: int = 100_000,end: int = 1_000_000,fermat_k: int = 5,repeats: int = 5,save_results: bool = True,show_plot: bool = True) -> Dict[str, List[Dict]]:
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
    print(f"Generierte {len(numbers)} Testzahlen (Typ '{num_type}')")
    
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        # JSON
        json_file = f"results/results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(datasets, f, indent=4)
        # CSV
        csv_file = f"results/results_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm", "n", "avg_time", "std_dev", "best", "worst"])
            for algo, data in datasets.items():
                for entry in data:
                    writer.writerow([algo,entry["n"],entry["avg_time"],entry["std_dev"],entry["best_time"],entry["worst_time"]])
        print(f"Ergebnisse gespeichert in {json_file} und {csv_file}")
    
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
            figsize=(10, 8)
        )
    return datasets




################################################
# CALL ##
if __name__ == "__main__":
    random.seed(42)  # Für Reproduzierbarkeit
    results = run_prime_criteria_analysis(
        n_numbers=50,
        num_type='p',
        start=100_000,
        end=200_000,
        fermat_k=3,
        repeats=3
    )
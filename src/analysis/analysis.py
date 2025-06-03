from src.primality.criteria import *
from src.primality.tests import *
from src.analysis.timing import measure_runtime
from src.analysis.plot import plot_runtime
from src.analysis.dataset import *
import random
from sympy import isprime, primerange
from typing import List, Dict
from datetime import datetime

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

# whole analysis function for prime criteria, calls generation, measurement, saving and plotting
def run_prime_criteria_analysis(n_numbers: int = 100, num_type: str = 'g', start: int = 100_000, end: int = 1_000_000, fermat_k: int = 5,repeats: int = 5, save_results: bool = True, show_plot: bool = True) -> Dict[str, List[Dict]]: 
    # GENERATION
    numbers = generate_numbers(n=n_numbers, start=start, end=end, num_type=num_type)
    print(f"Generating {len(numbers)} test numbers for prime criteria (Typ '{num_type}')")
    
    # INITIALIZE DATA STRUCTURES 
    init_criteria_data(numbers)
    
    # CALLS
    fermat = lambda n: fermat_criterion(n, fermat_k)
    wilson = wilson_criterion
    initial_lucas = initial_lucas_test
    lucas = lucas_test
    optimized_lucas = optimized_lucas_test
    
    # MEASURE 
    datasets = {
        "Fermat": measure_runtime(fermat, numbers, f"Fermat (k={fermat_k})"),
        "Wilson": measure_runtime(wilson, numbers, "Wilson"),
        "Initial Lucas": measure_runtime(initial_lucas, numbers, "Initial Lucas"),
        "Lucas": measure_runtime(lucas, numbers, "Lucas"),
        "Optimized Lucas": measure_runtime(optimized_lucas, numbers, "Optimized Lucas")
    }
 
    # CALL PROTOCOL
    criteria_protocoll(numbers, datasets)

        # SAVE RESTULTS
    if save_results:
        save_json(datasets, get_timestamped_filename("criteria", "json"))
        export_to_csv(datasets, get_timestamped_filename("criteria", "csv"))

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

# whole analysis function for prime tests, calls generation, measurement, saving and plotting
def run_prime_test_analysis(
    n_numbers: int = 100,
    num_type: str = 'g',
    start: int = 100_000,
    end: int = 1_000_000,
    tests: str = "msa",
    repeats: int = 3,
    save_results: bool = True,
    show_plot: bool = True,
) -> Dict[str, List[Dict]]:
    
    # GENERATION
    numbers = generate_numbers(n=n_numbers, start=start, end=end, num_type=num_type)
    print(f"Generating {len(numbers)} test numbers for prime tests (Typ '{num_type}')")

    # INITIALIZE DATA STRUCTURES 
    init_tests_data(numbers)

    # SELECT TESTS
    test_functions = {}
    test_colors = {}
    
    if 'm' in tests.lower():
        test_functions["Miller-Rabin"] = lambda n: miller_selfridge_rabin_test(n, repeats)
        test_colors["Miller-Rabin"] = "blue"
    if 's' in tests.lower():
        test_functions["Solovay-Strassen"] = lambda n: solovay_strassen_test(n, repeats)
        test_colors["Solovay-Strassen"] = "green"
    if 'a' in tests.lower():
        test_functions["AKS"] = aks_test
        test_colors["AKS"] = "red"
    if not test_functions:
        raise ValueError("Choose at least one test (m/s/a)")
    
    # MEASURE
    datasets = {}
    for test_name, test_fn in test_functions.items():
        datasets[test_name] = measure_runtime(test_fn, numbers, f"{test_name} (k={repeats})" if test_name != "AKS" else test_name)
    
    # SAVE RESULTS
    if save_results:
        save_json(datasets, get_timestamped_filename("tests", "json"))
        export_to_csv(datasets, get_timestamped_filename("tests", "csv"))

    # PROTOCOL
    tests_protocoll(numbers, tests, datasets)

    # PLOTTING
    if show_plot and datasets:
        colors = ["red", "blue", "green"]
        plot_data = {
            "n_values": [[entry["n"] for entry in data] for data in datasets.values()],
            "avg_times": [[entry["avg_time"] for entry in data] for data in datasets.values()],
            "std_devs": [[entry["std_dev"] for entry in data] for data in datasets.values()],
            "best_times": [[entry["best_time"] for entry in data] for data in datasets.values()],
            "worst_times": [[entry["worst_time"] for entry in data] for data in datasets.values()],
            "labels": [data[0]["label"] for data in datasets.values()],
            "colors": colors[:len(datasets)]
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
    
    return {
        "test_data": tests_data,
        "timing_data": datasets,
        "numbers": numbers,
        "selected_tests": list(datasets.keys())
    }


################################################
# CALL
if __name__ == "__main__":
    #random.seed(42)  # Für Reproduzierbarkeit
    criteria = run_prime_criteria_analysis(n_numbers=2, num_type='p', start=10000, end=100000, fermat_k=3, repeats=3, save_results=False, show_plot=True)
    tests = run_prime_test_analysis(n_numbers=4, num_type='p', start=10, end=100, tests="msa", repeats=5, save_results=False, show_plot=True)

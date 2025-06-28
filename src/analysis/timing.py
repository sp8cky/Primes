import time
from typing import Callable, List, Dict, Any
from src.analysis import dataset
import statistics, timeit
from statistics import mean, stdev
from src.primality.test_protocoll import test_data
from sympy import isprime

def measure_runtime(fn: Callable[[int], bool], inputs: List[int], test_name: str, label: str = "", runs_per_n: int = 5) -> List[Dict]:
    results = []

    for n in inputs:
        runtimes = []

        for _ in range(runs_per_n):
            start = time.perf_counter()
            fn(n)
            end = time.perf_counter()
            runtimes.append(end - start)

        avg_t = mean(runtimes)
        best_t = min(runtimes)
        worst_t = max(runtimes)
        std_t = stdev(runtimes) if len(runtimes) > 1 else 0.0

        entry = test_data[test_name][n]
        entry["avg_time"] = avg_t
        entry["best_time"] = best_t
        entry["worst_time"] = worst_t
        entry["std_dev"] = std_t

        results.append({
            "n": n,
            "avg_time": avg_t,
            "std_dev": std_t,
            "best_time": best_t,
            "worst_time": worst_t,
            "label": label
        })

    return results

# check for errors in the test data
def analyze_errors(test_data: Dict[str, Dict[int, Dict[str, Any]]]) -> None:
    total_tests = 0
    total_errors = 0
    for testname, numbers in test_data.items():
        for n, data in numbers.items():
            result = data.get("result")
            if result is None:
                continue

            true_prime = isprime(n)
            data["true_prime"] = true_prime
            is_error = (result != true_prime)
            data["is_error"] = is_error
            data["false_positive"] = (not true_prime and result is True)
            data["false_negative"] = (true_prime and result is False)

            total_tests += 1
            if is_error:
                total_errors += 1

    print(f"Fehleranalyse abgeschlossen: {total_errors} Fehler bei {total_tests} Tests.")
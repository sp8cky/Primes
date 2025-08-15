import time
from src.primality.constants import *
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
        repeat_results = []
        true_prime = isprime(n)

        for _ in range(runs_per_n):
            try:
                start = time.perf_counter()
                result = fn(n)
                end = time.perf_counter()
                runtimes.append(end - start)
                repeat_results.append(result)
            except ValueError as e:
                test_data[test_name][n] = {
                    "was_value_error": True,
                    "reason": str(e),  # optional, für Logging oder Debug
                    "repeat_results": [],
                    "repeat_count": 0,
                    "result": None
                }
                break

        if not repeat_results:
            continue  # keine gültige Wiederholung für n

        avg_t = mean(runtimes)
        best_t = min(runtimes)
        worst_t = max(runtimes)
        std_t = stdev(runtimes) if len(runtimes) > 1 else 0.0

        entry = test_data[test_name].setdefault(n, {})
        entry["avg_time"] = avg_t
        entry["best_time"] = best_t
        entry["worst_time"] = worst_t
        entry["std_dev"] = std_t
        entry["true_prime"] = true_prime
        entry["result"] = repeat_results[-1]
        entry["repeat_count"] = runs_per_n
        entry["repeat_results"] = repeat_results

        results.append({
            "n": n,
            "avg_time": avg_t,
            "std_dev": std_t,
            "best_time": best_t,
            "worst_time": worst_t,
            "label": label
        })

    return results




def analyze_errors(test_data: Dict[str, Dict[int, Dict[str, Any]]]) -> None:
    total_n = 0
    total_tests = 0
    total_errors = 0

    print("\nFehleranalyse pro Test:\n")
    for testname, numbers in test_data.items():
        test_errors = 0
        test_runs = 0
        test_n = 0
        excluded_inputs = 0

        for n, data in numbers.items():
            true_prime = isprime(n)
            data["true_prime"] = true_prime

            repeat_results = data.get("repeat_results", [data.get("result")])
            repeat_count = len(repeat_results)

            # Nur PRIME/COMPOSITE zählen zur Fehleranalyse
            valid_results = [r for r in repeat_results if r in {PRIME, COMPOSITE}]
            excluded = repeat_count - len(valid_results)

            if not valid_results:
                excluded_inputs += 1
                continue

            # Fehler zählen (abweichend vom wahren Primstatus)
            error_count = sum(
                1 for res in valid_results
                if (res == PRIME and not true_prime) or (res == COMPOSITE and true_prime)
            )

            error_rate = error_count / len(valid_results) if valid_results else 0.0

            data["error_rate"] = error_rate
            data["is_error"] = (error_count > 0)
            data["false_positive"] = (not true_prime and any(r == PRIME for r in valid_results))
            data["false_negative"] = (true_prime and all(r == COMPOSITE for r in valid_results))

            test_errors += error_count
            test_runs += len(valid_results)
            test_n += 1

    print(f"Fehleranalyse abgeschlossen")
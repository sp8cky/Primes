import time
from src.primality.constants import *
from typing import Callable, List, Dict, Any
from statistics import mean, stdev
from src.primality.test_protocoll import test_data
from sympy import isprime

# Misst die Laufzeiten der Funktion fn für verschiedene Eingabewerte.
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
                    "reason": str(e),
                    "repeat_results": [],
                    "repeat_count": 0,
                    "result": None
                }
                break

        if not repeat_results: continue  # keine gültige Wiederholung für n

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



# Führt eine Fehleranalyse der Testergebnisse durch.
def analyze_errors(test_data: Dict[str, Dict[int, Dict[str, Any]]]) -> None:
    print("\nFehleranalyse pro Test:\n")
    for testname, numbers in test_data.items():
        for n, data in numbers.items():
            result = data.get("result")
            true_prime = isprime(n)
            data["true_prime"] = true_prime

            # NOT_APPLICABLE überspringen
            if result == NOT_APPLICABLE or INVALID is None:
                data["error_rate"] = None
                data["is_error"] = False
                data["false_positive"] = False
                data["false_negative"] = False
                data["repeat_count"] = 0
                data["error_count"] = 0
                continue

            repeat_results = data.get("repeat_results", [result])

            # PRIME/COMPOSITE zählen
            valid_results = [r for r in repeat_results if r in {PRIME, COMPOSITE}]
            if not valid_results:
                data["error_rate"] = None
                data["is_error"] = False
                data["false_positive"] = False
                data["false_negative"] = False
                data["repeat_count"] = len(repeat_results)
                data["error_count"] = 0
                continue

            # Fehler zählen
            error_count = sum(
                1 for r in valid_results
                if (r == PRIME and not true_prime) or (r == COMPOSITE and true_prime)
            )

            # Fehlerstatistiken speichern
            data["error_rate"] = error_count / len(valid_results)
            data["is_error"] = (error_count > 0)
            data["false_positive"] = (not true_prime and any(r == PRIME for r in valid_results))
            data["false_negative"] = (true_prime and any(r == COMPOSITE for r in valid_results))
            data["repeat_count"] = len(repeat_results)
            data["error_count"] = error_count

    print("Fehleranalyse abgeschlossen")
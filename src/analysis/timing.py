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
        repeat_results = []
        true_prime = isprime(n)

        for _ in range(runs_per_n):
            start = time.perf_counter()
            result = fn(n)
            end = time.perf_counter()
            runtimes.append(end - start)
            repeat_results.append(result)

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

        # Letztes Ergebnis speichern (nicht als Fehlerbewertung!)
        entry["result"] = repeat_results[-1]

        # Neu: Wiederholungen und Ergebnisse speichern für Fehleranalyse
        entry["repeat_count"] = runs_per_n
        entry["repeat_results"] = repeat_results

        # Letztes Ergebnis speichern (nicht als Fehlerbewertung!)
        entry["result"] = repeat_results[-1]

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

        for n, data in numbers.items():
            result = data.get("result")
            if result is None:
                continue

            true_prime = isprime(n)
            data["true_prime"] = true_prime

            # Einzeltest-Fehler (letztes Ergebnis)
            is_error = (result != true_prime)
            data["is_error"] = is_error
            data["false_positive"] = (not true_prime and result is True)
            data["false_negative"] = (true_prime and result is False)

            # Wiederholungen & Fehleranzahl aus gespeicherten Wiederholungen
            rc = data.get("repeat_count", 1)
            repeat_results = data.get("repeat_results", [result])
            error_count = sum(r != true_prime for r in repeat_results)

            # Fehlerquote
            error_rate = round(error_count / rc, 3) if rc > 0 else 0.0

            data["error_count"] = error_count
            data["error_rate"] = error_rate

            if rc < error_count:
                print(f"⚠️ Inkonsistenz bei n = {n} ({testname}): Fehleranzahl > Wiederholungen.")

            test_runs += rc
            test_errors += error_count
            test_n += 1

        total_n += test_n
        total_tests += test_runs
        total_errors += test_errors

        rate = round(test_errors / test_runs, 4) if test_runs else 0.0
        print(f"- {testname}: Fehlerrate {rate:.2%} bei {test_n} Zahlen (insg. {test_runs} Tests, {test_errors} Fehler)")

    if total_tests > 0:
        error_percent = (total_errors / total_tests) * 100
    else:
        error_percent = 0.0

    print(f"\nGesamt: {total_errors} Fehler bei {total_tests} Wiederholungen über {total_n} Zahlen ({error_percent:.2f}%)")
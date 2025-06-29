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
        error_count = 0
        true_prime = isprime(n)

        for _ in range(runs_per_n):
            start = time.perf_counter()
            result = fn(n)
            end = time.perf_counter()
            runtimes.append(end - start)

            if result != true_prime: error_count += 1

        avg_t = mean(runtimes)
        best_t = min(runtimes)
        worst_t = max(runtimes)
        std_t = stdev(runtimes) if len(runtimes) > 1 else 0.0

        entry = test_data[test_name][n]
        entry["avg_time"] = avg_t
        entry["best_time"] = best_t
        entry["worst_time"] = worst_t
        entry["std_dev"] = std_t
        entry["true_prime"] = true_prime

        # Fehlerdaten ergänzen
        entry["repeat_count"] = runs_per_n
        entry["error_count"] = error_count
        entry["error_rate"] = round(error_count / runs_per_n, 3)

        # Letzter Einzeltest
        entry["is_error"] = (entry["result"] != true_prime)
        entry["false_positive"] = (not true_prime and entry["result"] is True)
        entry["false_negative"] = (true_prime and entry["result"] is False)

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

            # Einzelergebnis: letzter Lauf
            is_error = (result != true_prime)
            data["is_error"] = is_error
            data["false_positive"] = (not true_prime and result is True)
            data["false_negative"] = (true_prime and result is False)

            # Wiederholungen
            rc = data.get("repeat_count", 0)
            ec = data.get("error_count", 0)

            # Sicherheitshalber validieren
            if rc < ec:
                print(f"⚠️ Inkonsistenz bei n = {n} ({testname}): Fehleranzahl > Wiederholungen.")

            if rc == 0:  # kein Repeat-Modus verwendet
                rc = 1
                ec = 1 if is_error else 0
                data["repeat_count"] = rc
                data["error_count"] = ec
                data["error_rate"] = round(ec / rc, 3)

            test_runs += rc
            test_errors += ec
            test_n += 1

        total_n += test_n
        total_tests += test_runs
        total_errors += test_errors

        rate = round(test_errors / test_runs, 4) if test_runs else 0.0
        print(f"- {testname}: Fehlerrate {rate:.2%} bei {test_n} Zahlen (insg. {test_runs} Tests, {test_errors} Fehler)")

    print(f"\nGesamt: {total_errors} Fehler bei {total_tests} Wiederholungen über {total_n} Zahlen ({(total_errors / total_tests) * 100:.2f}%)")
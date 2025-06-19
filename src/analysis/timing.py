import time
from typing import Callable, List, Dict
from src.analysis import dataset
import statistics, timeit
from statistics import mean, stdev
from src.primality.tests import test_data


def measure_runtime(fn: Callable[[int], bool], inputs: List[int], test_name: str, label: str = "") -> List[Dict]:
    results = []
    for n in inputs:
        start = time.perf_counter()
        fn(n)
        end = time.perf_counter()
        runtime = end - start

        # ✅ Zeit direkt im test_data speichern
        test_data[test_name][n]["time"] = runtime

        results.append({
            "n": n,
            "avg_time": runtime,
            "std_dev": 0.0,
            "best_time": runtime,
            "worst_time": runtime,
            "label": label
        })

    return results

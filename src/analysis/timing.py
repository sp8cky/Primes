import time
from typing import Callable, List, Dict
from src.analysis import dataset
import statistics, timeit
from statistics import mean, stdev

def measure_runtime(fn: Callable[[int], bool], inputs: List[int], label: str = "") -> Dict[str, List[Dict]]:
    results = []
    for n in inputs:
        start = time.perf_counter()
        fn(n)
        end = time.perf_counter()
        runtime = end - start

        results.append({
            "n": n,
            "avg_time": runtime, # just one run, so avg_time is the same as runtime
            "std_dev": 0.0, # no variance in a single run
            "best_time": runtime,
            "worst_time": runtime,
            "label": label
        })

    return results

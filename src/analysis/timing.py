import time
from typing import Callable, List, Dict
from src.analysis import dataset
import statistics, timeit
from statistics import mean, stdev


def measure_runtime(func: Callable, numbers: List[int], label: str, repeat: int = 5, precompute: bool = False) -> List[Dict]:
    results = []
    for n in numbers:
        times = []
        # Pre-Compute einmalig
        if precompute:
            result = func(n)  # Füllt criteria_data
            times = [timeit.timeit(lambda: func(n), number=1) for _ in range(repeat)]
        else:
            for _ in range(repeat):
                start = time.perf_counter()
                result = func(n)
                times.append(time.perf_counter() - start)
        
        results.append({
            "n": n,
            "result": result,
            "best_time": min(times),
            "avg_time": mean(times),
            "std_dev": stdev(times) if len(times) > 1 else 0,
            "worst_time": max(times),
            "label": label
        })
    return results

"""
# Function to measure the runtime of a given function with various inputs
def measure_runtime(fn: Callable[[int], bool], inputs: List[int], label: str = "", repeat: int = 3) -> List[Dict]:
    results = []
    for n in inputs:
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn(n)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        best_time = min(times)
        worst_time = max(times)

        results.append({
            "n": n,
            "avg_time": avg_time,
            "std_dev": std_dev,
            "best_time": best_time,
            "worst_time": worst_time,
            "label": label
        })

    return results"""
import time
from typing import Callable, List, Dict
from src.analysis import dataset
import statistics

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

    return results
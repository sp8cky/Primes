import time
from typing import Callable, List, Dict
from src.analysis import dataset


def measure_runtime(fn: Callable[[int], bool], inputs: List[int], label: str = "") -> List[Dict]:
    results = []
    for n in inputs:
        start = time.perf_counter()
        fn(n)
        end = time.perf_counter()
        duration = end - start
        results.append({"n": n, "time": duration, "label": label})
    return results
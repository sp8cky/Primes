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

def measure_runtime_stats(fn: Callable[[int], bool], inputs: List[int], rounds: int = 3) -> Dict[str, List]:
    stats = {"n": inputs, "worst": [], "best": [], "avg": []}
    for n in inputs:
        times = []
        for _ in range(rounds):
            start = time.perf_counter()
            fn(n)
            times.append(time.perf_counter() - start)
        stats["worst"].append(max(times))
        stats["best"].append(min(times))
        stats["avg"].append(sum(times)/rounds)
    return stats
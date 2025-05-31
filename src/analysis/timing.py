import time
from typing import Callable, List, Dict
from statistics import mean, stdev
from src.analysis import dataset


def measure_runtime(fn: Callable[[int], bool], inputs: List[int], label: str = "", repeat: int = 3, verbose: bool = False) -> List[Dict]:
    results = []
    for n in inputs:
        durations = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn(n)
            end = time.perf_counter()
            durations.append(end - start)

        avg = mean(durations)
        std = stdev(durations) if repeat > 1 else 0.0

        if verbose:
            print(f"{label}: n={n} → {avg:.6f}s ± {std:.6f}s")

        results.append({
            "n": n,
            "avg_time": avg,
            "std_dev": std,
            "label": label
        })

    return results

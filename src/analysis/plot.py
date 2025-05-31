import matplotlib.pyplot as plt
import numpy as np

def plot_runtime(n_lists, time_lists, std_lists=None, labels=None, colors=None, figsize=(8, 8)):
    if labels is None:
        labels = [None] * len(n_lists)
    if colors is None:
        colors = [None] * len(n_lists)

    plt.figure(figsize=figsize)

    for n, t, s, label, color in zip(n_lists, time_lists, std_lists or [None] * len(n_lists), labels, colors):
        n_scaled = [ni // 1000 for ni in n]  # Skaliere n auf 1000er
        t_ms = [ti * 1000 for ti in t]       # Sek → ms
        if s is not None:
            s_ms = [si * 1000 for si in s]
            plt.errorbar(n_scaled, t_ms, yerr=s_ms, label=label, fmt='o-', capsize=3, color=color)
        else:
            plt.plot(n_scaled, t_ms, marker="o", label=label, color=color)

    plt.xlabel("n (in 1000)")
    plt.ylabel("Laufzeit (ms)")
    plt.yscale("log")
    plt.title("Laufzeitverhalten der Kriterien")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

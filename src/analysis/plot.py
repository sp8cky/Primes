import matplotlib.pyplot as plt
import numpy as np

# Plotting function for runtime analysis of algorithms with error bars and best/worst case shading
def plot_runtime(n_lists, time_lists, std_lists=None, best_lists=None, worst_lists=None,
                 labels=None, colors=None, figsize=(10, 6), use_log=True):
    if labels is None: labels = [None] * len(n_lists)
    if colors is None: colors = [None] * len(n_lists)

    plt.figure(figsize=figsize)

    for n, t, std, best, worst, label, color in zip(
        n_lists, time_lists,
        std_lists or [None]*len(n_lists),
        best_lists or [None]*len(n_lists),
        worst_lists or [None]*len(n_lists),
        labels, colors
    ):
        t_ms = [ti * 1000 for ti in t]
        std_ms = [s * 1000 for s in std] if std else None
        best_ms = [b * 1000 for b in best] if best else None
        worst_ms = [w * 1000 for w in worst] if worst else None

        plt.plot(n, t_ms, marker="o", label=label, color=color)

        if std:
            plt.errorbar(n, t_ms, yerr=std_ms, fmt='none', capsize=3, color=color, alpha=0.6)

        if best and worst:
            plt.fill_between(n, best_ms, worst_ms, alpha=0.1, color=color)

    plt.xlabel("Getestete Zahl n")
    plt.ylabel("Laufzeit (ms)")
    if use_log:
        plt.yscale("log")
    plt.title("Laufzeitverhalten")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

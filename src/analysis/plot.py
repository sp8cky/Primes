import matplotlib.pyplot as plt
import numpy as np

def plot_runtime(n_lists, time_lists, labels=None):
    if labels is None:
        labels = [None] * len(n_lists)
    for n, t, label in zip(n_lists, time_lists, labels):
        n_scaled = [ni // 1000 for ni in n]  # Skaliere n auf 1000er
        t_ms = [ti * 1000 for ti in t]  # calulate in milliseconds
        plt.plot(n_scaled, t_ms, marker="o", label=label)
    plt.xlabel("n (in 1000)")
    plt.ylabel("Laufzeit (ms)")
    plt.yscale("log")
    plt.title("Laufzeitverhalten der Kriterien")
    plt.legend()
    plt.grid(True)
    plt.show()


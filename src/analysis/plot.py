import matplotlib.pyplot as plt
import pandas as pd
from src.analysis import dataset

def plot_runtime(n_lists, time_lists, labels=None):
    if labels is None:
        labels = [None] * len(n_lists)
    for n, t, label in zip(n_lists, time_lists, labels):
        plt.plot(n, t, marker="o", label=label)
    plt.xlabel("n")
    plt.ylabel("Laufzeit (s)")
    plt.title("Laufzeitverhalten der Kriterien")
    plt.legend()
    plt.grid(True)
    plt.show()
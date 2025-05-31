import matplotlib.pyplot as plt

def plot_runtime(n_lists, time_lists, labels=None):
    if labels is None:
        labels = [None] * len(n_lists)
    for n, t, label in zip(n_lists, time_lists, labels):
        t_ms = [ti * 1000 for ti in t] # calulate in milliseconds
        plt.plot(n, t_ms, marker="o", label=label)
    plt.xlabel("n")
    plt.ylabel("Laufzeit (ms)")
    plt.ysxcale("log")
    plt.title("Laufzeitverhalten der Kriterien")
    plt.legend()
    plt.grid(True)
    plt.show()
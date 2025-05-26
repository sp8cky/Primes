import matplotlib.pyplot as plt
import seaborn as sns

def plot_runtime(data_dict, title="Laufzeitvergleich", xlabel="n", ylabel="Zeit (s)"):
    """
    data_dict: dict mit {Testname: [(n, zeit), (n, zeit), ...]}
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for label, data in data_dict.items():
        x = [n for n, t in data if t is not None]
        y = [t for n, t in data if t is not None]
        plt.plot(x, y, label=label, marker='o')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()
import os
import matplotlib.pyplot as plt
from src.analysis.dataset import *
from src.primality.test_config import *

def plot_runtime(n_lists, time_lists, std_lists=None, best_lists=None, worst_lists=None, labels=None, colors=None, figsize=(18, 9), use_log=True, total_numbers=None, runs_per_n=None):
    if labels is None: labels = [None] * len(n_lists)
    if colors is None: colors = [None] * len(n_lists)

    entries = []
    for i in range(len(n_lists)):
        label = labels[i]
        base_label = extract_base_label(label)
        group = TEST_GROUPS.get(base_label)
        if group is None:
            print(f"⚠️  Test '{base_label}' ist keiner bekannten Gruppe zugeordnet und wird übersprungen.")
            continue

        entries.append((base_label, group, label, n_lists[i], time_lists[i],
                        std_lists[i] if std_lists else None,
                        best_lists[i] if best_lists else None,
                        worst_lists[i] if worst_lists else None,
                        colors[i] if colors else None))

    if not entries:
        print("⚠️ Keine gültigen Daten zum Plotten.")
        return

    # Sortieren nach TEST_ORDER statt alphabetisch
    entries.sort(key=lambda x: TEST_ORDER.index(x[0]))

    # Farben und Linienstile pro Gruppe
    unique_groups = list(sorted(set(entry[1] for entry in entries)))
    group_colors = plt.cm.tab10.colors  # 10 Farben aus Matplotlib-Standardpalette
    line_styles = ['-', '--', '-.', ':']

    # Mapping Gruppe -> Farbe & Linienstil (zyklisch)
    group_style_map = {}
    for i, group in enumerate(unique_groups):
        color = group_colors[i % len(group_colors)]
        linestyle = line_styles[i % len(line_styles)]
        group_style_map[group] = (color, linestyle)

    plt.figure(figsize=figsize)

    for base_label, group, label, n, t, std, best, worst, user_color in entries:
        t_ms = [ti * 1000 for ti in t]
        std_ms = [s * 1000 for s in std] if std else None
        best_ms = [b * 1000 for b in best] if best else None
        worst_ms = [w * 1000 for w in worst] if worst else None

        color, linestyle = group_style_map.get(group, ('black', '-'))
        if user_color is not None:
            color = user_color

        plt.plot(n, t_ms, marker="o", label=f"{group} – {label}", color=color, linestyle=linestyle)

        if std:
            plt.errorbar(n, t_ms, yerr=std_ms, fmt='none', capsize=3, color=color, alpha=0.6)
        if best and worst:
            plt.fill_between(n, best_ms, worst_ms, alpha=0.1, color=color)

    plt.xlabel("Getestete Zahl n")
    plt.ylabel("Laufzeit (ms)")

    if use_log:
        plt.xscale("log")
        plt.yscale("log")

    title = "Laufzeitverhalten"
    if total_numbers is not None and runs_per_n is not None:
        title += f"\nAnzahl getesteter Zahlen: {total_numbers}, Wiederholungen: {runs_per_n}"
    plt.title(title)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename = get_timestamped_filename("test-plot", "png")
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    plt.savefig(path)
    plt.show()
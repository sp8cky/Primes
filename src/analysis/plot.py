import os
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from src.analysis.dataset import *
from src.primality.test_config import *

def plot_runtime(
    n_lists, time_lists, std_lists=None, best_lists=None, worst_lists=None,
    labels=None, colors=None, figsize=(18, 9), use_log=True,
    total_numbers=None, runs_per_n=None, group_ranges=None
):
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
    entries.sort(key=lambda x: TEST_ORDER.index(x[0]) if x[0] in TEST_ORDER else 999)

    # Farben und Linienstile pro Gruppe
    unique_groups = list(dict.fromkeys(entry[1] for entry in entries))  # Reihenfolge erhalten
    group_colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']
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

        plt.plot(n, t_ms, marker="o", label=label, color=color, linestyle=linestyle)

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

    # Gruppierung der Legenden-Einträge
    grouped_entries = defaultdict(list)
    test_index = {name: i for i, name in enumerate(TEST_ORDER)}
    group_order = []

    for base_label, group, label, *_ in entries:
        grouped_entries[group].append((test_index.get(base_label, 999), label, base_label))
        if group not in group_order:
            group_order.append(group)

    for group in grouped_entries:
        grouped_entries[group].sort(key=lambda x: x[0])

    # Legendenobjekte vorbereiten
    legend_elements = []

    for group in group_order:
        # Bereich anzeigen, auch bei Ramzy/Rao etc.
        range_str = ""
        if group_ranges and group in group_ranges:
            r = group_ranges[group]
            range_str = f" (n={r.get('n','?')}, start={r.get('start','?')}, end={r.get('end','?')})"
        else:
            range_str = " (n=?, start=?, end=?)"

        # Gruppenüberschrift (fett, kein Marker)
        legend_elements.append(Line2D(
            [0], [0], linestyle='none', label=f"{group}{range_str}",
            color='black', marker='', linewidth=0
        ))

        for _, label, base_label in grouped_entries[group]:
            color, linestyle = group_style_map.get(group, ('black', '-'))
            handle = Line2D([0], [0], color=color, linestyle=linestyle, marker='o', label=f"  {label}")
            legend_elements.append(handle)

    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename = get_timestamped_filename("test-plot", "png")
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    plt.savefig(path)
    plt.show()



def plot_runtime_grouped(
    n_lists, time_lists, std_lists=None, best_lists=None, worst_lists=None,
    labels=None, colors=None, figsize=(20, 12), use_log=True,
    total_numbers=None, runs_per_n=None, group_ranges=None
):
    if labels is None: labels = [None] * len(n_lists)
    if colors is None: colors = [None] * len(n_lists)

    # Sammle Einträge mit Gruppenzuordnung
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

    # Gruppiere nach Testgruppe
    grouped_entries = defaultdict(list)
    for entry in entries:
        grouped_entries[entry[1]].append(entry)

    groups = list(grouped_entries.keys())
    num_groups = len(groups)
    cols = 2
    rows = math.ceil(num_groups / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=False)
    axes = axes.flatten()

    group_colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']
    group_style_map = {}

    for i, group in enumerate(groups):
        color = group_colors[i % len(group_colors)]
        linestyle = line_styles[i % len(line_styles)]
        group_style_map[group] = (color, linestyle)

    for idx, group in enumerate(groups):
        ax = axes[idx]
        group_entries = grouped_entries[group]

        for base_label, _, label, n, t, std, best, worst, user_color in group_entries:
            t_ms = [ti * 1000 for ti in t]
            std_ms = [s * 1000 for s in std] if std else None
            best_ms = [b * 1000 for b in best] if best else None
            worst_ms = [w * 1000 for w in worst] if worst else None

            color, linestyle = group_style_map.get(group, ('black', '-'))
            if user_color is not None:
                color = user_color

            line, = ax.plot(n, t_ms, marker="o", label=label, color=color, linestyle=linestyle)
            if std:
                ax.errorbar(n, t_ms, yerr=std_ms, fmt='none', capsize=3, color=color, alpha=0.6)
            if best and worst:
                ax.fill_between(n, best_ms, worst_ms, alpha=0.1, color=color)

            # Beschriftung am letzten Punkt
            if n and t_ms:
                ax.text(n[-1], t_ms[-1], f" {label}", fontsize=8, color=color,
                       verticalalignment='bottom', horizontalalignment='left')

        # Gruppentitel mit Bereichsinformationen
        if group_ranges and group in group_ranges:
            r = group_ranges[group]
            title = f"{group} (n={r['n']}, start={r['start']}, end={r['end']})"
        else:
            title = f"{group} (n=?, start=?, end=?)"
        ax.set_title(title, fontsize=11)

        # Logarithmische Skalen
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='both')

        # X-Achsen Bereich
        n_values = []
        for _, _, _, n, *_ in group_entries:
            n_values.extend(n)
        if n_values:
            n_min = max(1, min(n_values) * 0.9)
            n_max = max(n_values) * 1.1
            ax.set_xlim(n_min, n_max)

        # Y-Achsen Bereich mit Puffer
        t_values = []
        for _, _, _, _, t, *_ in group_entries:
            t_values.extend([ti * 1000 for ti in t])
        if t_values:
            t_min = max(0.1, min(t_values) * 0.5)  # Mehr Platz nach unten
            t_max = max(t_values) * 2  # Mehr Platz nach oben
            ax.set_ylim(t_min, t_max)

        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.set_ylabel("Laufzeit (ms)")
        ax.set_xlabel("Getestete Zahl n")

    # Leere Subplots entfernen
    for j in range(len(groups), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Laufzeitvergleich pro Testgruppe", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    filename = get_timestamped_filename("test-subplots", "png")
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    plt.savefig(path)
    plt.show()
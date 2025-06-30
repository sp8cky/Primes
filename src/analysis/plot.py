import os
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from src.analysis.dataset import *
from src.primality.test_config import *

def plot_runtime(
    n_lists, time_lists, std_lists=None, best_lists=None, worst_lists=None,
    labels=None, colors=None, figsize=(18, 9), use_log=True,
    total_numbers=None, runs_per_n=None, group_ranges=None,
    seed=None, timestamp=None, variant=None, start=None, end=None
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

    entries.sort(key=lambda x: TEST_ORDER.index(x[0]) if x[0] in TEST_ORDER else 999)

    # Farben und Linienstile
    unique_groups = list(dict.fromkeys(entry[1] for entry in entries))
    group_colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']
    group_style_map = {
        group: (group_colors[i % len(group_colors)], line_styles[i % len(line_styles)])
        for i, group in enumerate(unique_groups)
    }

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
        plt.xscale("linear")
        plt.yscale("log")
        all_x = [x for entry in entries for x in entry[3]]
        xmin, xmax = min(all_x), max(all_x)
        step = 1000
        ticks = list(range(int(xmin//step)*step, int(xmax + step), step))
        plt.gca().set_xticks(ticks)
        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    else:
        plt.xscale("linear")
        plt.yscale("linear")

    # Titel + Untertitel je nach Variante
    title = "Laufzeitanalyse"
    if variant == 1:
        all_x = [x for entry in entries for x in entry[3]]
        subtitle = f"Variante 1, n = {total_numbers}, Wiederholungen = {runs_per_n}, start = {start}, end = {end}, Seed = {seed}"
    elif variant == 2:
        subtitle = f"Variante 2, Seed = {seed}"
    else:
        subtitle = f"Seed = {seed}"

    plt.title(f"{title}\n{subtitle}")

    # Legende gruppieren
    grouped_entries = defaultdict(list)
    test_index = {name: i for i, name in enumerate(TEST_ORDER)}
    group_order = []

    for base_label, group, label, *_ in entries:
        grouped_entries[group].append((test_index.get(base_label, 999), label, base_label))
        if group not in group_order:
            group_order.append(group)

    for group in grouped_entries:
        grouped_entries[group].sort(key=lambda x: x[0])

    legend_elements = []

    for group in group_order:
        if variant == 2 and group_ranges and group in group_ranges:
            r = group_ranges[group]
            range_str = f" (n={r.get('n','?')}, start={r.get('start','?')}, end={r.get('end','?')})"
        elif variant == 1:
            range_str = ""
        else:
            range_str = " (n=?, start=?, end=?)"

        legend_elements.append(Line2D(
            [0], [0], linestyle='none', label=f"{group}{range_str}",
            color='black', marker=None, linewidth=0
        ))

        for idx, label, base_label in grouped_entries[group]:
            user_color = None
            linestyle = '-'
            for e in entries:
                if e[0] == base_label and e[2] == label:
                    user_color = e[8]
                    _, linestyle = group_style_map.get(group, ('black', '-'))
                    break

            color = user_color if user_color is not None else group_style_map.get(group, ('black', '-'))[0]

            handle = Line2D([0], [0], color=color, linestyle=linestyle, marker='o', label=f"  {label}")
            legend_elements.append(handle)

    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if timestamp is None:
        filename = f"test-plot-seed{seed}-v{variant}.png"
    else:
        filename = f"{timestamp}-test-plot-seed{seed}-v{variant}.png"
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    plt.savefig(path)
    plt.show()



def plot_runtime_and_errorrate_by_group(
    datasets: dict,
    test_data: dict,
    group_ranges: dict = None,
    figsize=(12, 7),
    show_errors: bool = True,
    timestamp: str = None,
    seed: int = None,
    variant: int = None,
    start: int = None,
    end: int = None,
):
    os.makedirs(DATA_DIR, exist_ok=True)

    grouped_data = defaultdict(list)
    for test_name, data in datasets.items():
        base_label = extract_base_label(test_name)
        group = TEST_GROUPS.get(base_label, "Unbekannte Gruppe")
        if not data:
            continue

        n_values = [entry["n"] for entry in data]
        avg_times = [entry["avg_time"] * 1000 for entry in data]
        best_times = [entry["best_time"] * 1000 for entry in data]
        worst_times = [entry["worst_time"] * 1000 for entry in data]
        std_devs = [entry["std_dev"] * 1000 for entry in data]

        grouped_data[group].append(
            (test_name, avg_times, n_values, std_devs, best_times, worst_times)
        )

    for group, tests in grouped_data.items():
        plt.figure(figsize=figsize)
        all_n_values = []

        for test_name, avg_times, n_values, std_devs, best_times, worst_times in tests:
            # Linie plotten und Farbe abgreifen
            line_handle, = plt.plot(n_values, avg_times, marker='o', label=test_name)
            color = line_handle.get_color()

            # Fehlerbalken mit gleicher Farbe zeichnen
            plt.errorbar(n_values, avg_times, yerr=std_devs, fmt='none', capsize=3, alpha=0.6, color=color)
            plt.fill_between(n_values, best_times, worst_times, alpha=0.1, color=color)

            all_n_values.extend(n_values)

        plt.title(f"Laufzeitverhalten der Gruppe: {group}")
        plt.xlabel("Getestete Zahl n")
        plt.ylabel("Durchschnittliche Laufzeit (ms)")
        plt.yscale("log")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        # Dynamische X-Achse
        if variant == 2 and group_ranges and group in group_ranges:
            x_min = group_ranges[group].get("start", min(all_n_values))
            x_max = group_ranges[group].get("end", max(all_n_values))
        else:
            x_min = start if start is not None else min(all_n_values)
            x_max = end if end is not None else max(all_n_values)

        plt.xscale("linear")
        plt.xlim(x_min, x_max)
        step = max((x_max - x_min) // 10, 1)
        plt.xticks(range(x_min, x_max + 1, step))
        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

        # Bereichsinformation für die Legende
        range_str = ""
        if group_ranges and group in group_ranges:
            r = group_ranges[group]
            range_str = f"(n={r.get('n','?')}, start={r.get('start','?')}, end={r.get('end','?')})"
        plt.legend(title=f"{group} {range_str}", fontsize=9)

        # Fehlerbalken
        if show_errors:
            ax2 = plt.gca().twinx()
            error_rates = []
            test_labels = []
            for test_name, *_ in tests:
                test_entries = test_data.get(test_name, {})
                total_repeats = sum(entry.get("repeat_count", 0) for entry in test_entries.values())
                total_errors = sum(entry.get("error_count", 0) for entry in test_entries.values())
                rate = total_errors / total_repeats if total_repeats > 0 else 0
                error_rates.append(rate)
                test_labels.append(test_name)
            ax2.bar(test_labels, error_rates, color='red', alpha=0.4, label="Fehlerrate")
            ax2.set_ylabel("Fehlerrate")
            ax2.set_ylim(0, 1)
            ax2.legend(loc="upper right", fontsize=9)
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Speichern
        safe_group = group.replace(" ", "_").replace("/", "_")
        fname = f"{timestamp}-test-plot-group_{safe_group}-seed{seed}-v{variant}.png"
        path = os.path.join(DATA_DIR, fname)
        plt.savefig(path)
        plt.close()
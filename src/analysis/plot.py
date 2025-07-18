import os, math, statistics
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter, MaxNLocator
from src.analysis.dataset import *
from src.primality.test_config import *



   
# graph: avg runtime über alle wiederholungen in ms
# balken oben/unten: fehlerbalken, die standardabweichung zeigt
# schattierung: bereich zwischen best/worst case




def log_base_10_label(x, _):
    if x == 0:
        return "0"
    exp = int(np.log10(x))
    base = round(x / (10 ** exp))
    if base == 1:
        return f"$10^{{{exp}}}$"
    else:
        return f"${base} \\times 10^{{{exp}}}$"

def set_adaptive_xaxis(ax, start, end, force_log=None):
    start = float(start)
    end = float(end)
    use_log = force_log if force_log is not None else (np.log10(end) - np.log10(max(start, 1))) >= 3

    if use_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(log_base_10_label))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10), numticks=100))

        ticks = [start]
        mid = np.sqrt(start * end)  # log-mittig
        left_mid = np.sqrt(start * mid)
        right_mid = np.sqrt(mid * end)
        ticks += [left_mid, mid, right_mid, end]
        ticks = sorted(set(ticks))
        ax.set_xticks(ticks)
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))

        mid = (start + end) / 2
        left_mid = (start + mid) / 2
        right_mid = (mid + end) / 2
        ticks = [start, left_mid, mid, right_mid, end]
        ax.set_xticks(ticks)

 

def fixed_step_range(x_min, x_max, steps=10):
    step = choose_step_range(x_min, x_max)  # ← nutze deine eigene Logik!
    x_min_rounded = 0                      # wir wollen bei 0 starten
    x_max_rounded = round_up(x_max, step)  # nächsthöherer runder Wert
    return x_min_rounded, x_max_rounded, step


def choose_step_range(x_min, x_max, target_steps=10):
    range_ = x_max - x_min
    if range_ <= 0:
        return 1  # fallback bei ungültigem Bereich

    rough_step = range_ // target_steps
    magnitude = 10 ** int(math.floor(math.log10(rough_step)))
    multiples = [1, 2, 5, 10]

    for m in multiples:
        step = m * magnitude
        if range_ // step <= target_steps:
            return step

    return 10 * magnitude  # fallback


def round_down(x, base):
    return (x // base) * base

def round_up(x, base):
    return ((x + base - 1) // base) * base

def scientific_format(x, pos):
    if x == 0:
        return "0"
    exponent = int(math.log10(x))
    coefficient = x / (10**exponent)
    if coefficient.is_integer():
        coefficient = int(coefficient)
    return fr"${coefficient}\times10^{{{exponent}}}$"

def format_scientific_str(x):
    if x == 0:
        return "0"
    exponent = int(math.floor(math.log10(x)))
    coefficient = x / (10**exponent)
    # Falls Koeffizient ganzzahlig ist, als int formatieren
    if coefficient.is_integer():
        coefficient = int(coefficient)
    return fr"${coefficient}\times10^{{{exponent}}}$"



def plot_runtime(
    n_lists, time_lists, std_lists=None, best_lists=None, worst_lists=None,
    labels=None, colors=None, figsize=(24, 14), use_log=True,
    total_numbers=None, runs_per_n=None, group_ranges=None,
    seed=None, timestamp=None, variant=None, start=None, end=None, custom_xticks=None
):
    if labels is None: labels = [None] * len(n_lists)
    if colors is None: colors = [None] * len(n_lists)

    entries = []
    for i in range(len(n_lists)):
        label = labels[i]
        base_label = extract_base_label(label)
        testname = None
        for tn, cfg in TEST_CONFIG.items():
            if cfg.get("label") == label or tn == base_label:
                testname = tn
                break

        if testname is None:
            print(f"⚠️ Test '{label}' nicht in test_config gefunden, übersprungen.")
            continue

        group = TEST_CONFIG[testname].get("testgroup", "Unbekannte Gruppe")

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

    plt.xlabel("Testzahl n (logarithmisch)", fontsize=16)
    plt.ylabel("Laufzeit [ms]", fontsize=16)

    ax = plt.gca()
    all_x = [x for entry in entries for x in entry[3]]
    xmin = min(all_x) if all_x else 1
    xmax = max(all_x) if all_x else 10

    ax.set_xscale("log")
    ax.set_yscale("log")
    if custom_xticks:
        ax.set_xticks(custom_xticks)
        ax.set_xlim(min(custom_xticks), max(custom_xticks))
        if 0 in custom_xticks:
            # Definiere benutzerdefinierte Tick-Labels, um "0" anzuzeigen
            ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else log_base_10_label(x, _)))
    else:
        x_min, x_max, _ = fixed_step_range(0, xmax)
        ax.set_xlim(x_min, x_max)

    ax.get_xaxis().set_major_formatter(FuncFormatter(log_base_10_label))

    # Titel
    title = "Laufzeitanalyse"
    if variant == 1:
        subtitle = fr"Gesamtauswertung über {total_numbers}, zufällig gewählte Zahlen im Bereich [{format_scientific_str(start)}, {format_scientific_str(end)}], jeweils mit {runs_per_n} Wiederholungen (Seed = {seed})"
    elif variant == 2:
        subtitle = fr"Gruppenauswertung mit angepassten Werten für n, start und end pro Gruppe, jeweils mit {runs_per_n} Wiederholungen (Seed = {seed})"
    else:
        subtitle = f"Seed = {seed}"

    plt.title(f"{title}\n{subtitle}")

    # Legende gruppiert
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
        range_str = ""
        if variant == 2 and group_ranges and group in group_ranges:
            gr = group_ranges[group]
            n = gr.get('n', '?')
            start = gr.get('start', 0)
            end = gr.get('end', 0)

            start_fmt = format_scientific_str(start)
            end_fmt = format_scientific_str(end)
            range_str = f" (n={n}, start={start_fmt}, end={end_fmt})"

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

    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    filename = f"{timestamp}-test-plot-seed{seed}-v{variant}.png" if timestamp else f"test-plot-seed{seed}-v{variant}.png"
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    plt.savefig(path)
    plt.close()




def plot_runtime_and_errorrate_by_group(
    datasets: dict,
    test_data: dict,
    group_ranges: dict = None,
    figsize=(12, 7),
    show_errors: bool = True,
    timestamp: str = None,
    seed: int = None,
    variant: int = None,
    runs_per_n=None, 
):
    os.makedirs(DATA_DIR, exist_ok=True)

    grouped_data = defaultdict(list)
    for test_name, data in datasets.items():
        if test_name not in TEST_CONFIG:
            print(f"⚠️ Test '{test_name}' nicht in TEST_CONFIG gefunden, übersprungen.")
            continue

        group = TEST_CONFIG[test_name].get("plotgroup", TEST_CONFIG[test_name].get("testgroup", "Unbekannte Gruppe"))
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
        fig, ax1 = plt.subplots(figsize=figsize)
        all_n_values = []

        color_map = {}

        for test_name, avg_times, n_values, std_devs, best_times, worst_times in tests:
            line_handle, = ax1.plot(n_values, avg_times, marker='o', label=test_name)
            color = line_handle.get_color()
            color_map[test_name] = color

            avg_runtime = statistics.mean(avg_times) if avg_times else 0
            ax1.plot(0, avg_runtime, 'x', markersize=10, color=color,
                    markeredgewidth=2, transform=ax1.get_yaxis_transform())

            ax1.errorbar(n_values, avg_times, yerr=std_devs, fmt='none', capsize=3, alpha=0.6, color=color)
            ax1.fill_between(n_values, best_times, worst_times, alpha=0.1, color=color)
            all_n_values.extend(n_values)

        if group_ranges and group in group_ranges:
            gr = group_ranges[group]
            n = gr.get('n', '?')
            start = gr.get('start', 0)
            end = gr.get('end', max(all_n_values) if all_n_values else 1)
        else:
            n = '?'
            start = 0
            end = max(all_n_values) if all_n_values else 1

        subtitle = fr"Gruppenauswertung mit {n} Zahlen, zufällig gewählt im Bereich [{format_scientific_str(start)}, {format_scientific_str(end)}], jeweils mit {runs_per_n} Wiederholungen (Seed = {seed})"
        title = f"Laufzeitverhalten der Gruppe: {group}"
        ax1.set_title(f"{title}\n{subtitle}")
        ax1.set_xlabel("Testzahl n (linear)", fontsize=16)
        ax1.set_ylabel("Laufzeit [ms] (logarithmisch)" if show_errors else "Laufzeit [ms] (linear)", fontsize=16)
        ax1.set_yscale("log")
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        x_min_raw = 0
        x_max_raw = max(all_n_values)
        x_min, x_max, _ = fixed_step_range(x_min_raw, x_max_raw)

        set_adaptive_xaxis(ax1, start, end)
        ax1.set_xlim(x_min, x_max)

        if show_errors:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Fehlerrate", fontsize=16)
            ax2.set_ylim(0, 1)
            ax2.grid(False)

            for test_name, avg_times, n_values, *_ in tests:
                test_entries = test_data.get(test_name, {})
                if not test_entries:
                    continue

                error_rates_per_n = []
                for n in n_values:
                    entry = test_entries.get(n, {})
                    rate = entry.get("error_rate", None)
                    if rate is not None:
                        error_rates_per_n.append((n, rate))

                if not error_rates_per_n:
                    continue

                error_rates_per_n.sort()
                n_sorted = [n for n, _ in error_rates_per_n]
                rates_sorted = [rate for _, rate in error_rates_per_n]

                color = color_map.get(test_name, "gray")
                ax2.plot(n_sorted, rates_sorted, linestyle="--", marker="x", color=color, label=f"{test_name} Fehlerrate")

                avg_error = statistics.mean(rates_sorted) if rates_sorted else 0
                ax2.plot(1, avg_error, 'x', markersize=10, color=color,
                         markeredgewidth=2, transform=ax2.get_yaxis_transform())

            ax2.legend(loc="upper right", fontsize=16)

        range_str = ""
        if group_ranges and group in group_ranges:
            r = group_ranges[group]
            #start_fmt = format_scientific_str(r.get('start', 0))
            #end_fmt = format_scientific_str(r.get('end', 0))
            #range_str = f" (n={r.get('n','?')}, start={start_fmt}, end={end_fmt})"
            start = r.get("start", 0)
            end = r.get("end", 0)

            # Anpassung der x-Achse mit der neuen Funktion
            set_adaptive_xaxis(ax1, start, end)

        ax1.legend(title=f"{group}{range_str}", title_fontsize=16, fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14) if show_errors else None

        fig.tight_layout()

        safe_group = group.replace(" ", "_").replace("/", "_")
        fname = f"{timestamp}-test-plot-group_{safe_group}-seed{seed}-v{variant}.png"
        path = os.path.join(DATA_DIR, fname)
        plt.savefig(path)
        plt.close()

import os, math
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from src.analysis.dataset import *
from src.primality.test_config import *


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

def human_format(x, pos):
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.0f} Mrd"
    elif x >= 1_000_000:
        return f"{x/1_000_000:.0f} Mio"
    elif x >= 1_000:
        return f"{x/1_000:.0f} Tsd"
    else:
        return str(int(x))
    
def log_base_10_label(x, pos):
    if x <= 0:
        return "0"
    exponent = int(math.log10(x))
    if 10 ** exponent == x:
        return rf"$10^{{{exponent}}}$"
    else:
        return f"{int(x)}"  # fallback, z. B. für unglatte Werte

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

    plt.xlabel("Getestete Zahl n")
    plt.ylabel("Laufzeit (ms)")

    if use_log:
        plt.xscale("linear")
        plt.yscale("log")
        all_x = [x for entry in entries for x in entry[3]]
        xmin, xmax = 0, max(all_x)
        x_min, x_max, step = fixed_step_range(xmin, xmax)
        ticks = list(range(x_min, x_max + 1, step))

        ax = plt.gca()
        ax.set_xticks(ticks)
        ax.set_xlim(x_min, x_max)
        #ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    else:
        plt.xscale("linear")
        plt.yscale("linear")

    # Titel
    title = "Laufzeitanalyse"
    if variant == 1:
        subtitle = f"Variante 1, n = {total_numbers}, Wiederholungen = {runs_per_n}, start = {start}, end = {end}, Seed = {seed}"
    elif variant == 2:
        subtitle = f"Variante 2, Seed = {seed}"
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
            r = group_ranges[group]
            range_str = f" (n={r.get('n','?')}, start={r.get('start','?')}, end={r.get('end','?')})"

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

            ax1.errorbar(n_values, avg_times, yerr=std_devs, fmt='none', capsize=3, alpha=0.6, color=color)
            ax1.fill_between(n_values, best_times, worst_times, alpha=0.1, color=color)
            all_n_values.extend(n_values)

        ax1.set_title(f"Laufzeitverhalten der Gruppe: {group}")
        ax1.set_xlabel("Getestete Zahl n")
        ax1.set_ylabel("Durchschnittliche Laufzeit (ms)")
        ax1.set_yscale("log")
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        x_min_raw = 0
        x_max_raw = max(all_n_values)
        x_min, x_max, step = fixed_step_range(x_min_raw, x_max_raw)

        ax1.set_xscale("linear")
        ax1.set_xlim(x_min, x_max)
        ax1.set_xticks(list(range(x_min, x_max + 1, step)))
        #ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        ax1.xaxis.set_major_formatter(FuncFormatter(human_format))

        if show_errors:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Fehlerrate")
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

            ax2.legend(loc="upper right", fontsize=8)

        range_str = ""
        if group_ranges and group in group_ranges:
            r = group_ranges[group]
            range_str = f"(n={r.get('n','?')}, start={r.get('start','?')}, end={r.get('end','?')})"
        ax1.legend(title=f"{group} {range_str}", fontsize=9)

        fig.tight_layout()

        safe_group = group.replace(" ", "_").replace("/", "_")
        fname = f"{timestamp}-test-plot-group_{safe_group}-seed{seed}-v{variant}.png"
        path = os.path.join(DATA_DIR, fname)
        plt.savefig(path)
        plt.close()
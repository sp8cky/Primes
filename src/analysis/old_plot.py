import os, math, statistics
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter, MaxNLocator, NullLocator
from src.analysis.dataset import *
from src.primality.test_config import *

"""def plot_runtime_and_errorrate_by_group(
    datasets: dict,
    test_data: dict,
    group_ranges: dict = None,
    figsize=(12, 7),
    show_errors: bool = True,
    timestamp: str = None,
    seed: int = None,
    variant: int = None,
    runs_per_n=None,
    prob_test_repeats=None,
):
    # === Initialisierung ===
    os.makedirs(DATA_DIR, exist_ok=True)
    config = get_test_config(prob_test_repeats=prob_test_repeats, global_seed=seed)

    # === Gruppierung der Datensätze nach Plotgruppen ===
    grouped_data = defaultdict(list)
    for test_name, data in datasets.items():
        if test_name not in TEST_CONFIG:
            print(f"⚠️ Test '{test_name}' nicht in TEST_CONFIG gefunden, übersprungen.")
            continue
        if not data: continue

        group = TEST_CONFIG[test_name].get("plotgroup", TEST_CONFIG[test_name].get("testgroup", "Unbekannte Gruppe"))
        n_values = [entry["n"] for entry in data]
        avg_times = [entry["avg_time"] * 1000 for entry in data]
        best_times = [entry["best_time"] * 1000 for entry in data]
        worst_times = [entry["worst_time"] * 1000 for entry in data]
        std_devs = [entry["std_dev"] * 1000 for entry in data]

        grouped_data[group].append((test_name, avg_times, n_values, std_devs, best_times, worst_times))

    # === Iteration über jede Gruppe ===
    for group, tests in grouped_data.items():
        fig, ax1 = plt.subplots(figsize=figsize)
        all_n_values = []

        # Mapping Testnamen zu Labels (inkl. Repeats aus config)
        testname_to_label = {
            test_name: config[test_name]["label"]
            for test_name, *_ in tests
            if test_name in config
        }

        color_map = {}

        # === Laufzeitlinien und Fehlerbereiche plotten ===
        for test_name, avg_times, n_values, std_devs, best_times, worst_times in tests:
            label = testname_to_label.get(test_name, test_name)
            line_handle, = ax1.plot(n_values, avg_times, marker='o', label=label)
            color = line_handle.get_color()
            color_map[test_name] = color

            avg_runtime = statistics.mean(avg_times) if avg_times else 0
            ax1.plot(0, avg_runtime, 'x', markersize=14, color=color, markeredgewidth=3,
                     transform=ax1.get_yaxis_transform(), clip_on=False)

            ax1.errorbar(n_values, avg_times, yerr=std_devs, fmt='none', capsize=3, alpha=0.6, color=color)
            ax1.fill_between(n_values, best_times, worst_times, alpha=0.1, color=color)
            all_n_values.extend(n_values)

        # === Bereichs- und Achsenkonfiguration ===
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

        # === X-Achse: ggf. benutzerdefinierte Ticks verwenden ===
        custom_xticks = group_ranges[group].get("xticks") if group_ranges and group in group_ranges else None

        if custom_xticks:
            ax1.set_xscale("log")
            ax1.set_xticks(custom_xticks)
            ax1.set_xlim(min(custom_xticks), max(custom_xticks))
            ax1.xaxis.set_minor_locator(NullLocator())
            if 0 in custom_xticks:
                ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else log_base_10_label(x, _)))
            else:
                ax1.xaxis.set_major_formatter(FuncFormatter(log_base_10_label))
        else:
            set_adaptive_xaxis(ax1, start, end)
            ax1.set_xlim(x_min, x_max)

        # === Fehlerraten plotten (rechte Y-Achse) ===
        if show_errors:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Fehlerrate", fontsize=16)
            ax2.set_ylim(0, 1)
            ax2.grid(False)

            for test_name, *_ in tests:
                test_entries = test_data.get(test_name, {})
                if not test_entries:
                    continue
                n_values = [entry["n"] for entry in datasets[test_name]]
                error_rates_per_n = [(n, test_entries.get(n, {}).get("error_rate")) for n in n_values]
                error_rates_per_n = [(n, rate) for n, rate in error_rates_per_n if rate is not None]

                if not error_rates_per_n:
                    continue

                error_rates_per_n.sort()
                n_sorted = [n for n, _ in error_rates_per_n]
                rates_sorted = [rate for _, rate in error_rates_per_n]

                color = color_map.get(test_name, "gray")
                label_error = testname_to_label.get(test_name, test_name)
                ax2.plot(n_sorted, rates_sorted, linestyle="--", marker="x", color=color, label=f"{label_error} Fehlerrate")

                avg_error = statistics.mean(rates_sorted) if rates_sorted else 0
                ax2.plot(1, avg_error, 'x', markersize=14, color=color, markeredgewidth=3,
                         transform=ax2.get_yaxis_transform(), clip_on=False)

        # === Gemeinsame Legende erstellen ===
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        labels2_clean = [label.replace(" Fehlerrate", "") for label in labels2]
        time_dict = dict(zip(labels1, handles1))
        error_dict = dict(zip(labels2_clean, handles2))

        handles_laufzeit, labels_laufzeit = [], []
        handles_fehler, labels_fehler = [], []

        for test_name, avg_times, *_ in tests:
            color = color_map.get(test_name, "gray")
            label = testname_to_label.get(test_name, test_name)

            time_handle = time_dict.get(label, Line2D([], [], linestyle='-', color=color))
            error_handle = error_dict.get(label, Line2D([], [], linestyle='--', color=color))

            avg_time = statistics.mean(avg_times) if avg_times else 0
            test_entries = test_data.get(test_name, {})
            error_rates = [entry.get("error_rate") for entry in test_entries.values() if entry.get("error_rate") is not None]
            avg_error = statistics.mean(error_rates) if error_rates else 0

            avg_time_str = f"{avg_time:.3f} ms"
            avg_error_str = f"{avg_error:.4g}"

            labels_laufzeit.append(f"{label} Laufzeit [avg: {avg_time_str}]")
            labels_fehler.append(f"{label} Fehlerrate [avg: {avg_error_str}]")

            handles_laufzeit.append(time_handle)
            handles_fehler.append(error_handle)

        fig.legend(
            handles_laufzeit + handles_fehler,
            labels_laufzeit + labels_fehler,
            loc="upper right",
            bbox_to_anchor=(0.9, 0.95),
            ncol=2,
            fontsize=11,
            title="Legende",
            title_fontsize=12,
            columnspacing=2.8,
            handletextpad=1.2,
            borderaxespad=0.8,
            frameon=True
        )

        # === Achsenformatierung und Speichern ===
        ax1.tick_params(axis='both', which='major', labelsize=14)
        if show_errors:
            ax2.tick_params(axis='both', which='major', labelsize=14)

        fig.tight_layout()
        safe_group = group.replace(" ", "_").replace("/", "_")
        fname = f"{timestamp}-group-{safe_group}-seed{seed}-v{variant}.png"
        path = os.path.join(DATA_DIR, fname)
        plt.savefig(path)
        plt.close()



        # === Theoretische Laufzeit ===
        #plot_theory_runtimes(group=group, tests=tests,testname_to_label=testname_to_label,color_map=color_map,TEST_CONFIG=TEST_CONFIG,group_ranges=group_ranges,timestamp=timestamp,seed=seed,variant=variant,DATA_DIR=DATA_DIR,figsize=figsize)
"""

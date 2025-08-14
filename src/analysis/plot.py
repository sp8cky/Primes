import os, math, statistics
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter, MaxNLocator, NullLocator
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
        return f"${base}\\times 10^{{{exp}}}$"

def format_scientific_str(x):
    if x == 0:
        return "0"
    exponent = int(math.floor(math.log10(x)))
    coefficient = x / (10**exponent)
    # Falls Koeffizient ganzzahlig ist, als int formatieren
    if coefficient.is_integer():
        coefficient = int(coefficient)
    return fr"${coefficient}\times10^{{{exponent}}}$"


# plottet alle Tests in einem Plot
def plot_runtime(
    n_lists, time_lists, std_lists=None, best_lists=None, worst_lists=None,
    labels=None, colors=None, figsize=(24, 14), use_log=True,
    total_numbers=None, runs_per_n=None, group_ranges=None,
    seed=None, timestamp=None, variant=None, start=None, end=None, custom_xticks=None, number_type=None, filename=None
):
    # === Fallbacks für Labels/Farben ===
    if labels is None: labels = [None] * len(n_lists)
    if colors is None: colors = [None] * len(n_lists)

    # === Einträge vorbereiten und nach TEST_ORDER sortieren ===
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

    # === Farben und Linienstile je Gruppe definieren ===
    unique_groups = list(dict.fromkeys(entry[1] for entry in entries))
    group_colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']
    group_style_map = {
        group: (group_colors[i % len(group_colors)], line_styles[i % len(line_styles)])
        for i, group in enumerate(unique_groups)
    }

    # === Daten plotten ===
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
            plt.errorbar(n, t_ms, yerr=std_ms, fmt='none', capsize=2, color=color, alpha=0.3)
        if best and worst:
            plt.fill_between(n, best_ms, worst_ms, alpha=0.09, color=color)

    # === Achsenbeschriftungen ===
    plt.xlabel("Testzahl n (log.)", fontsize=20)
    plt.ylabel("Laufzeit [ms] (log.)", fontsize=20)
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    # === Nur benutzerdefinierte Xticks verwenden ===
    if custom_xticks:
        ax.set_xticks(custom_xticks)
        ax.set_xlim(min(custom_xticks), max(custom_xticks))
        if 0 in custom_xticks:
            ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else log_base_10_label(x, _)))
        else:
            ax.get_xaxis().set_major_formatter(FuncFormatter(log_base_10_label))
    else:
        if start is not None and end is not None:
            ax.set_xticks([start, end])
            ax.set_xlim(start, end)
        ax.get_xaxis().set_major_formatter(FuncFormatter(log_base_10_label))

    # === Plot-Titel ===
    title = "Laufzeitanalyse"
    if variant == 1:
        subtitle = (
            f"Gesamtauswertung über {total_numbers}, Zahlentyp {number_type}, zufällig gewählte Zahlen im Bereich [{format_scientific_str(start)}, {format_scientific_str(end)}],\n"
            f"jeweils mit {runs_per_n} Wiederholungen (Seed = {seed})")
    elif variant == 2:
        subtitle = (f"Gruppenauswertung mit angepassten Werten für n, Zahlentyp {number_type}, start und \n"
                    f"end pro Gruppe, jeweils mit {runs_per_n} Wiederholungen (Seed = {seed})")
    plt.suptitle(title, fontsize=24)
    plt.title(subtitle, fontsize=20)

    # === Legende gruppiert ===
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

            handle = Line2D([0], [0], color=color, linestyle=linestyle, marker='o', markersize=3, label=f"  {label}")
            legend_elements.append(handle)

    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # === Plot speichern ===
    fname = f"{timestamp}-plot-seed{seed}-v{variant}.png" if timestamp else f"plot-seed{seed}-v{variant}.png"
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    plt.savefig(path)
    plt.close()





############################################################################

def plot_grouped_all(datasets, test_data, group_ranges, timestamp, seed, variant, runs_per_n, prob_test_repeats, figsize=(20, 14), number_type=None, filename1=None, filename2=None):

    os.makedirs(DATA_DIR, exist_ok=True)
    config = get_test_config(prob_test_repeats=prob_test_repeats, global_seed=seed)
    grouped_data = defaultdict(list)

    for test_name, data in datasets.items():
        if test_name not in TEST_CONFIG or not data:
            continue
        plotgroup = TEST_CONFIG[test_name].get("plotgroup", TEST_CONFIG[test_name].get("testgroup", "Unbekannte Gruppe"))
        testgroup = TEST_CONFIG[test_name].get("testgroup", "Unbekannte Gruppe")  # wichtig!
        
        n_values = [entry["n"] for entry in data]
        avg_times = [entry["avg_time"] * 1000 for entry in data]
        best_times = [entry["best_time"] * 1000 for entry in data]
        worst_times = [entry["worst_time"] * 1000 for entry in data]
        std_devs = [entry["std_dev"] * 1000 for entry in data]
        
        grouped_data[plotgroup].append(
            (test_name, avg_times, n_values, std_devs, best_times, worst_times, testgroup)
        )

    # Erzeuge je Gruppe zwei separate Plots mit plot1() und plot2()
    for group, tests in grouped_data.items():
        testname_to_label = {
            test_name: config[test_name]["label"]
            for test_name, *_ in tests
            if test_name in config
        }
        color_map = {}
        safe_group = group.replace(" ", "_").replace("/", "_")

        # --- Y-Limits berechnen ---
        all_lower_bounds = []
        all_upper_bounds = []

        for _, avg_times, _, std_devs, best_times, worst_times, _ in tests:
            lower = [max(bt - std, 1e-5) for bt, std in zip(best_times, std_devs)]
            upper = [wt + std for wt, std in zip(worst_times, std_devs)]
            all_lower_bounds.extend(lower)
            all_upper_bounds.extend(upper)

        ymin = min(all_lower_bounds)
        ymax = max(all_upper_bounds)

        ylim = (ymin * 0.8, ymax * 1.2)  # oder nach Wunsch z. B. (ymin * 0.5, ymax * 2)

        
        plot_graph(
            group=group,
            tests=tests,
            config=config,
            color_map=color_map,
            group_ranges=group_ranges,
            timestamp=timestamp,
            seed=seed,
            variant=variant,
            runs_per_n=runs_per_n,
            test_data=test_data,
            datasets=datasets,
            ylim=ylim,
            figsize=figsize,
            number_type=number_type,
            filename=filename1,
        )

        plot_stats(
            group=group,
            tests=tests,
            config=config,
            color_map=color_map,
            group_ranges=group_ranges,
            timestamp=timestamp,
            seed=seed,
            variant=variant,
            runs_per_n=runs_per_n,
            test_data=test_data,
            datasets=datasets,
            ylim=ylim,
            figsize=figsize,
            number_type=number_type,
            filename=filename2,
        )

def plot_graph(group, tests, config, color_map, group_ranges, timestamp, seed, variant, runs_per_n, test_data, datasets, ylim, figsize, number_type, filename):
    fig, ax1 = plt.subplots(figsize=figsize)
    all_n_values = []

    # === Mapping Testnamen zu Labels (inkl. Repeats aus Config) ===
    testname_to_label = {
        test_name: config[test_name]["label"]
        for test_name, *_ in tests
        if test_name in config
    }

    # === Bereichsinfo aus testgroup holen ===
    first_testgroup = None
    for test_name, *_ in tests:
        if test_name in config:
            first_testgroup = config[test_name]["testgroup"]
            break

    if group_ranges and first_testgroup in group_ranges:
        gr = group_ranges[first_testgroup]
        n = gr.get('n', '?')
        start = gr.get('start', 0)
        end = gr.get('end', 1)
        custom_xticks = gr.get("xticks")
    else:
        n = '?'
        start = 0
        end = 1
        custom_xticks = None

    # === Laufzeitlinien und Fehlerbereiche plotten ===
    for test_name, avg_times, n_values, std_devs, best_times, worst_times, _ in tests:
        label = testname_to_label.get(test_name, test_name)
        line_handle, = ax1.plot(n_values, avg_times, marker='o', markersize=3, label=label)
        color = line_handle.get_color()
        color_map[test_name] = color

        avg_runtime = statistics.mean(avg_times) if avg_times else 0
        ax1.plot(0, avg_runtime, 'x', markersize=16, color=color, markeredgewidth=3, transform=ax1.get_yaxis_transform(), clip_on=False)
        all_n_values.extend(n_values)

    title = f"Laufzeitverhalten der Gruppe: {group}"
    subtitle = (
        f"Gruppenauswertung mit {n} Zahlen vom Typ {number_type}, zufällig gewählt im Bereich [{format_scientific_str(start)}, {format_scientific_str(end)}],\n "
        f"{runs_per_n} Wiederholungen (Seed = {seed})"
    )
    subtitle2 = (
        f"Gruppenauswertung mit {n} Zahlen vom Typ {number_type}, log. gleichverteilt gewählt im Bereich [{format_scientific_str(start)}, {format_scientific_str(end)}],\n "
        f"{runs_per_n} Wiederholungen (Seed = {seed})"
    )

    ax1.figure.suptitle(title, fontsize=24)
    ax1.set_title(subtitle2, fontsize=20)

    ax1.set_xlabel("Testzahl n (log.)", fontsize=20)
    ax1.set_ylabel("Laufzeit [ms] (log.)", fontsize=20)
    ax1.set_yscale("log")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # === Benutzerdefinierte X-Ticks oder adaptive Skalierung ===
    if custom_xticks:
        ax1.set_xscale("log")
        ax1.set_xticks(custom_xticks)
        ax1.set_xlim(min(custom_xticks), max(custom_xticks))
        ax1.xaxis.set_minor_locator(NullLocator())
        ax1.xaxis.set_major_formatter(FuncFormatter(log_base_10_label))
    else:
        if start is not None and end is not None:
            ax1.set_xticks([start, end])
            ax1.set_xlim(start, end)
        ax1.get_xaxis().set_major_formatter(FuncFormatter(log_base_10_label))

    # Y-Achse Skalierung
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(log_base_10_label))
    ax1.yaxis.set_minor_locator(NullLocator())

    # automatisch min/max berechnen
    if ylim:
        ax1.set_ylim(ylim)
    else:
        all_y_values = [t for _, avg, *_ in tests for t in avg if t > 0]
        ymin = min(all_y_values)
        ymax = max(all_y_values)
        
        # Fest definierte Grenzen
        hard_min = 0.00001 # 10^-5 ms
        hard_max = 10  # 10^1 ms
        
        if ymin >= hard_min and ymax <= hard_max:
            ax1.set_ylim(hard_min, hard_max)
        else:
            final_min = min(max(ymin * 0.95, 1e-6), hard_min)
            final_max = max(ymax * 1.5, hard_max)
            ax1.set_ylim(final_min, final_max)


    # === Fehlerraten auf rechter Y-Achse ===
    ax2 = ax1.twinx()
    ax2.set_ylabel("Fehlerrate", fontsize=20)
    ax2.set_ylim(0, 1)
    ax2.grid(False)

    for test_name, *_ in tests:
        test_entries = test_data.get(test_name, {})
        if not test_entries:
            continue

        error_rates = [
            entry.get("error_rate") for entry in test_entries.values()
            if entry.get("error_rate") is not None
        ]
        if not error_rates:
            continue

        avg_error = statistics.mean(error_rates)
        color = color_map.get(test_name, "gray")
        ax2.plot(1, avg_error, 'x', markersize=16, color=color, markeredgewidth=3, transform=ax2.get_yaxis_transform(), clip_on=False)

    # Legende mit Mittelwerten
    handles_laufzeit, labels_laufzeit = [], []
    handles_fehler, labels_fehler = [], []

    for test_name, avg_times, *_ in tests:
        color = color_map.get(test_name, "gray")
        label = testname_to_label.get(test_name, test_name)

        avg_time = statistics.mean(avg_times) if avg_times else 0
        test_entries = test_data.get(test_name, {})
        error_rates = [entry.get("error_rate") for entry in test_entries.values() if entry.get("error_rate") is not None]
        avg_error = statistics.mean(error_rates) if error_rates else 0

        avg_time_str = f"{avg_time:.3f} ms"
        avg_error_str = f"{avg_error:.4g}"

        handles_laufzeit.append(Line2D([], [], linestyle='-', marker='o', markersize=3, color=color))
        labels_laufzeit.append(f"{label} Laufzeit [avg: {avg_time_str}]")

        handles_fehler.append(Line2D([], [], linestyle='--', marker='x', color=color))
        labels_fehler.append(f"{label} Fehlerrate [avg: {avg_error_str}]")

    fig.legend(
        handles_laufzeit + handles_fehler,
        labels_laufzeit + labels_fehler,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.9),
        ncol=2,
        fontsize=16,
        title="Legende",
        title_fontsize=20,
        columnspacing=2.8,
        handletextpad=1.2,
        borderaxespad=0.8,
        framealpha=0.3,
        frameon=True
    )

    # Finalisieren
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()

    safe_group = group.replace(" ", "_").replace("/", "_")
    fname = f"{timestamp}-group-{safe_group}-graph-s{seed}-v{variant}.png"
    filename = f"d10-group-{safe_group}-graph-s{seed}-v{variant}.png"
    path = os.path.join(DATA_DIR, filename)
    plt.savefig(path)
    plt.close()


# plottet die Laufzeitmittelwerte, Fehlerbalken und Fehlerraten für eine Gruppe von Tests
def plot_stats(group, tests, config, color_map, group_ranges, timestamp, seed, variant, runs_per_n, test_data, datasets, ylim, figsize, number_type, filename):
    fig, ax1 = plt.subplots(figsize=figsize)
    all_n_values = []

    # === Mapping Testnamen zu Labels (inkl. Repeats aus Config) ===
    testname_to_label = {
        test_name: config[test_name]["label"]
        for test_name, *_ in tests
        if test_name in config
    }

    # === Bereichsinfo aus testgroup holen ===
    first_testgroup = None
    for test_name, *_ in tests:
        if test_name in config:
            first_testgroup = config[test_name]["testgroup"]
            break

    if group_ranges and first_testgroup in group_ranges:
        gr = group_ranges[first_testgroup]
        n = gr.get('n', '?')
        start = gr.get('start', 0)
        end = gr.get('end', 1)
        custom_xticks = gr.get("xticks")
    else:
        n = '?'
        start = 0
        end = 1
        custom_xticks = None

    # === Laufzeitlinien und Fehlerbereiche plotten ===
    for test_name, avg_times, n_values, std_devs, best_times, worst_times, _ in tests:
        label = testname_to_label.get(test_name, test_name)
        line_handle, = ax1.plot(n_values, avg_times, marker='o', markersize=3, label=label)
        color = line_handle.get_color()
        color_map[test_name] = color

        avg_runtime = statistics.mean(avg_times) if avg_times else 0
        ax1.plot(0, avg_runtime, 'x', markersize=16, color=color, markeredgewidth=3, transform=ax1.get_yaxis_transform(), clip_on=False)

        ax1.errorbar(n_values, avg_times, yerr=std_devs, fmt='none', capsize=2, alpha=0.3, color=color)
        ax1.fill_between(n_values, best_times, worst_times, alpha=0.09, color=color)
        all_n_values.extend(n_values)

    # === Achsenbeschriftung und Titel ===
    title = f"Laufzeitverhalten der Gruppe: {group}"
    subtitle = (
        f"Gruppenauswertung mit {n} Zahlen vom Typ {number_type}, zufällig gewählt im Bereich [{format_scientific_str(start)}, {format_scientific_str(end)}],\n "
        f"{runs_per_n} Wiederholungen (Seed = {seed})"
    )
    subtitle2 = (
        f"Gruppenauswertung mit {n} Zahlen vom Typ {number_type}, log. gleichverteilt gewählt im Bereich [{format_scientific_str(start)}, {format_scientific_str(end)}],\n"
        f"{runs_per_n} Wiederholungen (Seed = {seed})"
    )

    ax1.figure.suptitle(title, fontsize=24)
    ax1.set_title(subtitle2, fontsize=20)

    ax1.set_xlabel("Testzahl n (log.)", fontsize=20)
    ax1.set_ylabel("Laufzeit [ms] (log.)", fontsize=20)
    ax1.set_yscale("log")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # === Benutzerdefinierte X-Ticks oder adaptive Skalierung ===
    if custom_xticks:
        ax1.set_xscale("log")
        ax1.set_xticks(custom_xticks)
        ax1.set_xlim(min(custom_xticks), max(custom_xticks))
        ax1.xaxis.set_minor_locator(NullLocator())
        ax1.xaxis.set_major_formatter(FuncFormatter(log_base_10_label))
    else:
        if start is not None and end is not None:
            ax1.set_xticks([start, end])
            ax1.set_xlim(start, end)
        ax1.get_xaxis().set_major_formatter(FuncFormatter(log_base_10_label))

    # Y-Achse Skalierung
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(log_base_10_label))
    ax1.yaxis.set_minor_locator(NullLocator())

   # automatisch min/max berechnen
    if ylim:
        ax1.set_ylim(ylim)
    else:
        all_y_values = [t for _, avg, *_ in tests for t in avg if t > 0]
        ymin = min(all_y_values)
        ymax = max(all_y_values)
        
        # Fest definierte Grenzen
        hard_min = 0.00001 # 10^-5 ms
        hard_max = 10  # 10^1 ms
        
        if ymin >= hard_min and ymax <= hard_max:
            ax1.set_ylim(hard_min, hard_max)
        else:
            final_min = min(max(ymin * 0.95, 1e-6), hard_min)
            final_max = max(ymax * 1.5, hard_max)
            ax1.set_ylim(final_min, final_max)


    # === Fehlerraten auf rechter Y-Achse ===
    ax2 = ax1.twinx()
    ax2.set_ylabel("Fehlerrate", fontsize=20)
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
        ax2.plot(1, avg_error, 'x', markersize=16, color=color, markeredgewidth=3, transform=ax2.get_yaxis_transform(), clip_on=False)

    # === Legende ===
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
        bbox_to_anchor=(0.9, 0.9),
        ncol=2,
        fontsize=16,
        title="Legende",
        title_fontsize=20,
        columnspacing=2.8,
        handletextpad=1.2,
        borderaxespad=0.8,
        framealpha=0.3,
        frameon=True
    )

    # === Speichern ===
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    fig.tight_layout()
    safe_group = group.replace(" ", "_").replace("/", "_")
    fname = f"{timestamp}-group-{safe_group}-stats-s{seed}-v{variant}.png"
    filename = f"d10-group-{safe_group}-stats-s{seed}-v{variant}.png"
    path = os.path.join(DATA_DIR, filename)
    plt.savefig(path)
    plt.close()






# plottet theoretische Laufzeiten für die Tests in einer Gruppe
def plot_theory_runtimes(
    group,
    tests,
    testname_to_label,
    color_map,
    TEST_CONFIG,
    group_ranges,
    timestamp,
    seed,
    variant,
    DATA_DIR,
    figsize=(12, 7)
):
    fig, ax = plt.subplots(figsize=figsize)
    all_n_values = []
    all_y_values = []

    for test_name, avg_times, n_values, std_devs, best_times, worst_times in tests:
        label = testname_to_label.get(test_name, test_name)
        color = color_map.get(test_name, None)

        # Empirische Werte (Punkte + Linien)
        if avg_times and n_values:
            if test_name == "fermat":
                ax.plot(n_values, avg_times, marker='o', markersize=3, linestyle='-', color='red', linewidth=3, label=f"{label} (empirisch)")
            else:
                ax.plot(n_values, avg_times, marker='o', markersize=3, linestyle='-', color=color, alpha=0.3, label=f"{label} (empirisch)")
            all_n_values.extend(n_values)
            all_y_values.extend(avg_times)

        # Theoretische Werte berechnen
        fn_theory = TEST_CONFIG[test_name].get("runtime_theoretical_fn")
        if fn_theory:
            try:
                theory_vals = []
                for x in n_values:
                    m = int(math.log2(x)) + 1
                    val = fn_theory(m)
                    try:
                        val = float(val.evalf())
                    except Exception:
                        val = float(val)
                    val = val * 1000  # ms
                    theory_vals.append(val)
                ax.plot(n_values, theory_vals, linestyle='--', linewidth=2, color=color, label=f"{label} (theoretisch)")
                all_y_values.extend(theory_vals)
            except Exception as e:
                print(f"⚠️Fehler bei Theoriefunktion {test_name}: {e}")

    # Achsenbereich bestimmen
    min_y = max(min(all_y_values) * 0.5, 1e-4)
    max_y = max(all_y_values) * 2
    print(f"Min Y: {min_y}, Max Y: {max(all_y_values)}")
    ax.set_yscale("log")
    ax.set_ylim(min_y, max_y)

    # x-Achsenbereich (analog zu vorher)
    if group_ranges and group in group_ranges:
        gr = group_ranges[group]
        start = gr.get('start', 0)
        end = gr.get('end', max(all_n_values) if all_n_values else 1)
        custom_xticks = gr.get("xticks")
        n = gr.get('n', '?')
    else:
        start = 0
        end = max(all_n_values) if all_n_values else 1
        custom_xticks = None
        n = "?"

    subtitle = f"Gruppenauswertung mit {n} Zahlen, zufällig gewählt im Bereich [{format_scientific_str(start)}, {format_scientific_str(end)}], jeweils mit Wiederholungen (Seed = {seed})"
    ax.set_xlim(start, end)

    if custom_xticks:
        ax.set_xscale("log")
        ax.set_xticks(custom_xticks)
        ax.set_xlim(min(custom_xticks), max(custom_xticks))
        ax.xaxis.set_minor_locator(NullLocator())
        if 0 in custom_xticks:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else log_base_10_label(x, _)))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(log_base_10_label))
    else:
        if start is not None and end is not None:
            ax.set_xticks([start, end])
            ax.set_xlim(start, end)
        ax.get_xaxis().set_major_formatter(FuncFormatter(log_base_10_label))

    # Achsenbeschriftungen und Titel
    ax.set_title(f"Laufzeitverhalten Gruppe: {group}\n{subtitle}")
    ax.set_xlabel("Testzahl n (log.)", fontsize=20)
    ax.set_ylabel("Laufzeit [ms] (log.)", fontsize=20)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='upper left', ncol=1)
    ax.tick_params(axis='both', which='major', labelsize=20)

    fig.tight_layout()
    safe_group = group.replace(" ", "_").replace("/", "_")
    fname = f"{timestamp}-group-{safe_group}-theory-seed{seed}-v{variant}.png"
    path = os.path.join(DATA_DIR, fname)
    #plt.savefig(path) TODO: Uncomment this line to save the plot
    plt.close()
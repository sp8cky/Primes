from src.primality.tests import *
from src.primality.test_protocoll import *
from src.primality.generate_primes import *
from src.analysis.timing import *
from src.analysis.plot import *
from src.analysis.dataset import *
from src.primality.test_config import *
from src.analysis.analysis_wrapper import run_primetest_analysis
import time
from typing import List, Dict
from datetime import datetime




def automate_probabilistic_runs(
    iterations: int,
    base_repeats: int = 1
):
    for i in range(iterations):
        seed = random.randint(1, 1_000_000)
        prob_test_repeats = [base_repeats] * 5
        error_free = False
        run_count = 1

        while not error_free:
            repeat_str = "k" + "-".join(map(str, prob_test_repeats))

            csv_filename = f"d{run_count}-{repeat_str}-test-data-seed{seed}-v2.csv"
            plot_graph_filename = f"d{run_count}-{repeat_str}-group-Probabilistische_Tests-graph-s{seed}-v2.png"
            plot_stats_filename = f"d{run_count}-{repeat_str}-group-Probabilistische_Tests-stats-s{seed}-v2.png"

            print(f"\n[Run {run_count}] Starte Analyse mit Seed={seed}, Repeats={prob_test_repeats}...")

            # run_primetest_analysis gibt hier z.B. datasets, numbers_per_test, test_config zurück
            datasets, numbers_per_test, test_config = run_primetest_analysis(
                prob_test_repeats=prob_test_repeats,
                seed=seed,
                csv_filename=csv_filename,
                plot_filename=plot_graph_filename,
                stats_filename=plot_stats_filename
            )

            analyze_errors(test_data)

            # CSV Export
            export_test_data_to_csv(
                test_data=test_data,
                filename=csv_filename,
                test_config=test_config,
                numbers_per_test=numbers_per_test,
                metadata={
                    "seed": seed,
                    "variant": 2,
                    "prob_test_repeats": prob_test_repeats,
                    "group_ranges": group_ranges
                }
            )

            # Plots generieren (timestamp als datetime.now() oder Laufnummer)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plot_grouped_all(
                seed=seed,
                prob_test_repeats=prob_test_repeats,
                filename1=plot_graph_filename,
                filename2=plot_stats_filename
            )

            # Prüfen, ob alle Tests fehlerfrei sind
            error_free = True
            for test in tests:
                for n, entry in test_data[test].items():
                    if entry.get("error_count", 0) > 0:
                        error_free = False
                        break
                if not error_free:
                    break

            if not error_free:
                prob_test_repeats = [k + 1 for k in prob_test_repeats]
                run_count += 1
            else:
                print(f"\n✅ Alle Tests fehlerfrei bei Seed {seed} nach {run_count} Wiederholungen.")




automate_probabilistic_runs(3, base_repeats=1)
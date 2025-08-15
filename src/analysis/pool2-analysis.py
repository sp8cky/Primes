import pandas as pd
import numpy as np
import glob
import os, csv
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression



csv.field_size_limit(10**7)  # Erlaubt Felder bis ca. 10 MB
# Gruppenzuordnung
CATEGORY_MAP = {
    "Probabilistische Tests": ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen"],
    "Lucas-Tests": ["Initial Lucas", "Lucas", "Optimized Lucas"],
    "Langsame Tests": ["AKS10", "Wilson"],
    "Spezielle Tests": ["Pepin", "Lucas-Lehmer"],
    "Zusammengesetzte": ["Proth", "Proth Variant", "Pocklington", "Optimized Pocklington", "Optimized Pocklington Variant", "Generalized Pocklington", "Rao", "Ramzy"]
}
def classify_test(test_name: str) -> str:
    for group, patterns in CATEGORY_MAP.items():
        for p in patterns:
            if p in test_name:
                return group
    return "Spezielle Tests"  # Fallback

# ---------------------------
# Parsen einer Datei
# ---------------------------
def read_pool2_csv(file_path):
    """
    Liest eine Pool-2 CSV ein und gibt einen DataFrame im Pool-3-kompatiblen Detail-Format zurück.
    """
    detail_rows = []
    headers = None
    in_detail = False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Beginn der Detail-Tabelle erkennen
            if line.startswith("Gruppe,Test,Zahl"):
                headers = [h.strip() for h in line.split(",")]
                in_detail = True
                continue

            if not in_detail:
                continue  # alles vor der Detail-Tabelle ignorieren

            # Detailzeilen parsen
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(headers):
                parts += [""] * (len(headers) - len(parts))  # auffüllen

            row = dict(zip(headers, parts))

            # Typkonvertierung
            try:
                row["Zahl"] = pd.to_numeric(row.get("Zahl", ""), errors="coerce")
            except:
                row["Zahl"] = None

            for col in ["false_positive", "false_negative", "error_rate", "best_time", "avg_time", "worst_time", "std_dev"]:
                val = row.get(col, "")
                if isinstance(val, str):
                    val = val.replace(" ms", "").strip()
                try:
                    row[col] = float(val)
                except:
                    row[col] = None

            for col in ["true_prime", "is_error", "false_positive", "false_negative"]:
                val = row.get(col, "")
                if isinstance(val, str):
                    row[col] = 1 if val.strip().lower() == "true" else 0
                else:
                    row[col] = 0

            detail_rows.append(row)

    df = pd.DataFrame(detail_rows)

    # Kategorie anhand des Testnamens zuordnen
    if not df.empty:
        df["category"] = df["Test"].apply(classify_test)

    return df

def load_all_pool2(folder_path):
    """
    Lädt alle Pool-2 CSVs aus einem Ordner in einen kombinierten DataFrame.
    """
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = read_pool2_csv(file_path)
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    return combined_df


def analyse_overall(detail_df):
    """
    Aggregiert alle Detaildaten je Test und Kategorie (Durchschnittswerte).
    """
    if detail_df.empty:
        return pd.DataFrame()

    cols = [
        "avg_false_positive","avg_false_negative","avg_error_rate",
        "avg_best_time","avg_avg_time","avg_worst_time","avg_std_dev"
    ]
    for c in cols:
        if c in detail_df.columns:
            detail_df[c] = pd.to_numeric(detail_df[c], errors="coerce")

    overall = (
        detail_df
        .groupby(["Test","category"], as_index=False)
        .agg(
            avg_false_positive=("false_positive","mean"),
            avg_false_negative=("false_negative","mean"),
            avg_error_rate=("error_rate","mean"),
            avg_best_time=("best_time","mean"),
            avg_avg_time=("avg_time","mean"),
            avg_worst_time=("worst_time","mean"),
            avg_std_dev=("std_dev","std")
        )
        .sort_values(["category","avg_avg_time","avg_error_rate"], ascending=[True, True, True])
    )
    return overall


def runtime_stats(detail_df):
    """
    Laufzeit-Statistik je Test aus Detaildaten.
    """
    if detail_df.empty:
        return pd.DataFrame()

    return (
        detail_df
        .groupby(["Test","category"], as_index=False)
        .agg(
            best_time_min=("best_time","min"),
            time_avg=("avg_time","mean"),
            worst_time_max=("worst_time","max"),
            time_std=("avg_time","std"),
        )
    )

def fit_runtime_complexities_pool2(detail_df):
    """
    Führt Laufzeit-Fits für Pool-2-Daten durch.
    Gibt DataFrame mit theoretischer und praktischer Laufzeit pro Test zurück.
    """
    fit_results = []

    # Standardmodelle für praktische Laufzeit (erweitert)
    standard_models = {
        "O(1)": lambda n: np.ones_like(n),
        "O(log n)": lambda n: np.log(n),
        "O(log^2 n)": lambda n: np.log(n)**2,
        "O(log^3 n)": lambda n: np.log(n)**3,
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: n * np.log(n),
        "O(n log^2 n)": lambda n: n * np.log(n)**2,
        "O(n^2)": lambda n: n**2,
        "O(n^2 log n)": lambda n: n**2 * np.log(n),
        "O(n^3)": lambda n: n**3,
        "O(n^3 log n)": lambda n: n**3 * np.log(n),
        "O(2^n)": lambda n: 2**n,
        "O(n!)": lambda n: np.array([np.math.factorial(int(x)) for x in n]),
        "O(sqrt(n))": lambda n: np.sqrt(n),
        "O(n^(1/3))": lambda n: n**(1/3),
        "O(n^(1/4))": lambda n: n**(1/4)
    }

    # Theoretische Komplexität für bekannte Tests
    complexity_funcs = {
        "Fermat": lambda n: np.log(n)**3,
        "Miller-Selfridge-Rabin": lambda n: np.log(n)**4,
        "Solovay-Strassen": lambda n: np.log(n)**3,
        "Initial Lucas": lambda n: np.log(n)**3,
        "Lucas": lambda n: np.sqrt(n) * np.log(n)**3,
        "Optimized Lucas": lambda n: np.log(n)**5, # k * np.log(n)**3,# TODO
        "Wilson": lambda n: n * np.log(n)**2,
        "AKS10": lambda n: np.log(n)**18,
        "Proth": lambda n: np.log(n)**3,
        "Proth Variant": lambda n: np.log(n)**3,
        "Pocklington": lambda n: np.log(n)**3,
        "Optimized Pocklington": lambda n: np.log(n)**5, # TODO
        "Optimized Pocklington Variant": lambda n: np.log(n)**5, # k * np.log(n)**3,# TODO
        "Generalized Pocklington": lambda n: np.log(n)**5, # k * np.log(n)**3,# TODO
        "Rao": lambda n: np.log(n)**5, # TODO# TODO
        "Ramzy": lambda n: np.log(n)**5, # TODO
        "Pepin": lambda n: np.log(n)**5, # TODO
        "Lucas-Lehmer": lambda n: np.log(n)**5 # TODO
    }
    complexity_notation = {
        "Fermat": "O(log^3 n)",
        "Miller-Selfridge-Rabin": "O(log^4 n)",
        "Solovay-Strassen": "O(log^3 n)",
        "Initial Lucas": "O(log^3 n)",
        "Lucas": "O(sqrt(n) log^3 n)",
        "Optimized Lucas": "O(k log^3 n)",
        "Wilson": "O(n log^2 n)",
        "AKS10": "O(log^18 n)",
        "Proth": "O(log^3 n)",
        "Proth Variant": "O(log^3 n)",
        "Pocklington": "O(log^3 n)",
        "Optimized Pocklington": "O(log^5 n)",
        "Optimized Pocklington Variant": "O(k log^3 n)",
        "Generalized Pocklington": "O(k log^3 n)",
        "Rao": "O(log^5 n)",
        "Ramzy": "O(log^5 n)",
        "Pepin": "O(log^5 n)",
        "Lucas-Lehmer": "O(log^5 n)"
    }
    # Für jeden Test
    for test_name in detail_df["Test"].str.replace(r"\(k\s*=\s*\d+\)", "", regex=True).str.strip().unique():
        df_test = detail_df[detail_df["Test"].str.contains(test_name)]

        # Aggregation: Mittelwert der avg_time pro Zahl
        df_agg = df_test.groupby("Zahl", as_index=False)["avg_time"].mean()
        n_vals = df_agg["Zahl"].to_numpy()
        times = df_agg["avg_time"].to_numpy()

        # --- Fit theoretische Laufzeit ---
        th_func = complexity_funcs.get(test_name)
        th_label = complexity_notation.get(test_name, "unbekannt")
        if th_func is not None:
            x_th = th_func(n_vals).reshape(-1, 1)
            linreg = LinearRegression().fit(x_th, times)
            y_th_pred = linreg.predict(x_th)
            a_th = linreg.coef_[0]
            b_th = linreg.intercept_
            r2_th = r2_score(times, y_th_pred)
        else:
            a_th = b_th = r2_th = np.nan
            th_label = "unbekannt"

        # --- Fit praktische Laufzeit ---
        best_model_name = None
        best_r2 = -np.inf
        best_a = best_b = None
        for model_name, model_func in standard_models.items():
            try:
                x_model = model_func(n_vals).reshape(-1, 1)
                if np.std(x_model) < 1e-8:  # zu kleine Variation überspringen
                    continue
                linreg = LinearRegression().fit(x_model, times)
                y_pred = linreg.predict(x_model)
                r2 = r2_score(times, y_pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
                    best_a = linreg.coef_[0]
                    best_b = linreg.intercept_
            except Exception:
                continue

        fit_results.append({
            "Test": test_name,
            "th_laufzeit": th_label,
            "a_th": a_th,
            "b_th": b_th,
            "r2_th": r2_th,
            "prak_laufzeit": best_model_name,
            "a_prak": best_a,
            "b_prak": best_b,
            "r2_prak": best_r2
        })

    return pd.DataFrame(fit_results)


def error_stats(detail_df):
    """
    Fehleranalyse je Test aus Detaildaten.
    """
    if detail_df.empty:
        return pd.DataFrame()

    return (
        detail_df
        .groupby(["Test","category"], as_index=False)
        .agg(
            false_pos_sum=("false_positive","sum"),
            false_neg_sum=("false_negative","sum"),
            errors_sum=("is_error","sum"),
            error_rate_avg=("error_rate","mean"),
        )
    )


def group_summaries(overall_df):
    """
    Schnellster & Genauester Test pro Kategorie.
    """
    if overall_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    fastest = overall_df.loc[overall_df.groupby("category")["avg_avg_time"].idxmin()].reset_index(drop=True)
    most_accurate = overall_df.loc[overall_df.groupby("category")["avg_error_rate"].idxmin()].reset_index(drop=True)

    return fastest, most_accurate


def export_df(df, filename):
    if not isinstance(df, pd.DataFrame):
        print(f"Error: {filename} ist kein DataFrame!")
        return
    # Ordner erstellen, falls nicht existent
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Exportiert: {filename}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    folder = "C:\\Users\\julia\\Downloads\\testpool2"

    # Nur noch ein einheitlicher Detail-DF
    detail_df_all = load_all_pool2(folder)

    # Gesamtauswertung
    overall_df = analyse_overall(detail_df_all)
    print("\n ---Gesamtauswertung (Detaildaten)", overall_df)

    # Laufzeit
    rt_stats = runtime_stats(detail_df_all)
    rt_stats_sort = rt_stats.sort_values(["category","time_avg"], ascending=[True, True])
    print("\n ---Laufzeit-Statistik", rt_stats_sort)

    # Fit
    fit_results = fit_runtime_complexities_pool2(detail_df_all)
    fit_results["category"] = fit_results["Test"].apply(classify_test)
    fit_results_sort = fit_results.sort_values(["category", "r2_th"], ascending=[True, False])
    print("\n ---Fit-Statistik", fit_results_sort)

    # Fehler
    err_stats = error_stats(detail_df_all)
    err_stats_sort = err_stats.sort_values(["category","error_rate_avg"], ascending=[True, True])
    print("\n ---Fehleranalyse", err_stats_sort)

    # Gruppen
    fastest, most_accurate = group_summaries(overall_df)
    most_accurate_sort = most_accurate.sort_values("category")
    fastest_sort = fastest.sort_values("category")
    print("\n ---Schnellster Test je Kategorie", fastest_sort)
    print("\n ---Genauester Test je Kategorie", most_accurate_sort)

    # Export
    export_df(overall_df, "p2-result/1pool2_overall.csv")
    export_df(rt_stats, "p2-result/2pool2_runtime_stats.csv")
    export_df(err_stats, "p2-result/3pool2_error_stats.csv")
    export_df(fastest, "p2-result/4pool2_fastest_by_group.csv")
    export_df(most_accurate, "p2-result/5pool2_most_accurate_by_group.csv")
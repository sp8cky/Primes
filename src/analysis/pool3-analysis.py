
import os, csv, re, warnings
import numpy as np
import pandas as pd
from math import log
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# ----------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------

def model_linear(k, a, b):
    return a * k + b

def model_quad(k, a, b, c):
    return a * k**2 + b * k + c

def model_log(k, a, b):
    return a * np.log(k) + b

def model_power(k, a, p, b):
    return a * k**p + b

def model_exp(k, a, b):
    return a * np.exp(b * k)

def fit_models(k_values, times):
    results = {}

    # linear
    popt_lin, _ = curve_fit(model_linear, k_values, times)
    pred_lin = model_linear(k_values, *popt_lin)
    results["linear"] = (popt_lin, r2_score(times, pred_lin))

    # quadratisch
    popt_quad, _ = curve_fit(model_quad, k_values, times)
    pred_quad = model_quad(k_values, *popt_quad)
    results["quadratisch"] = (popt_quad, r2_score(times, pred_quad))

    # logarithmisch (k>0 vorausgesetzt)
    if np.all(k_values > 0):
        popt_log, _ = curve_fit(model_log, k_values, times)
        pred_log = model_log(k_values, *popt_log)
        results["logarithmisch"] = (popt_log, r2_score(times, pred_log))

    # potenzfunktion (p>0)
    def model_power_fixed_p(k, a, b):
        return a * k**best_p + b

    # Suche besten p über Grid-Search
    best_r2 = -np.inf
    best_p = None
    best_params = None
    for p in np.linspace(0.1, 3, 30):
        try:
            popt_pow, _ = curve_fit(lambda k,a,b: a*k**p + b, k_values, times)
            pred_pow = popt_pow[0] * k_values**p + popt_pow[1]
            r2 = r2_score(times, pred_pow)
            if r2 > best_r2:
                best_r2 = r2
                best_p = p
                best_params = popt_pow
        except:
            continue
    if best_p is not None:
        results["potenzfunktion"] = (best_params, best_r2, best_p)

    # exponentiell
    try:
        popt_exp, _ = curve_fit(model_exp, k_values, times, maxfev=10000)
        pred_exp = model_exp(k_values, *popt_exp)
        results["exponentiell"] = (popt_exp, r2_score(times, pred_exp))
    except:
        pass

    return results


def read_csv_folder(folder_path):
    results = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".csv"):
            continue
        print(f"Reading file: {fname}")
        avg_entries = []
        data_entries = []
        parsing_data = False
        with open(os.path.join(folder_path, fname), newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                if row[0] == "test_avg":
                    avg_entries.append(row)
                    continue
                if row[0] == "Gruppe":
                    parsing_data = True
                    continue
                if parsing_data and row[0] == "Probabilistisch":
                    data_entries.append(row)
        results.append({
            "file": fname,
            "avg": avg_entries,
            "data": data_entries
        })
    return results


# ----------------------------------------------------
# Analysefunktionen
# ----------------------------------------------------

def parse_avg_data(avg_entries):
    records = []
    for row in avg_entries:
        if not row or row[0] != "test_avg":
            continue
        test_name = row[1].strip()
        record = {"test": test_name}
        for field in row[2:]:
            if "=" in field:
                key, val = field.split("=", 1)
                val = val.replace(" ms", "").strip()
                try:
                    record[key] = float(val)
                except ValueError:
                    record[key] = val
        records.append(record)
    return pd.DataFrame(records)

def parse_detail_data(detail_entries):
    headers = ["Gruppe","Test","Zahl","Ergebnis","true_prime","is_error",
               "false_positive","false_negative","error_rate","best_time","avg_time",
               "worst_time","std_dev","a_values","other_fields","reason"]
    df = pd.DataFrame(detail_entries, columns=headers)

    # Umwandeln von Zahlenspalten
    numeric_cols = ["Zahl","error_rate","best_time","avg_time","worst_time","std_dev"]
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: float(str(x).replace(" ms","")) if x not in [None,""," "] else np.nan)

    # Bool-Spalten in 0/1 konvertieren
    bool_cols = ["false_positive","false_negative","is_error","true_prime"]
    for col in bool_cols:
        df[col] = df[col].astype(str).str.strip().map(lambda v: 1 if v.lower() == "true" else 0)

    return df

# Beispiel-Ausgabe für alle Modelle sortiert nach R2 pro Test
def analyse_k_influence(results_df):
# Basis-Testnamen ohne (k = ...)
    results_df["base_test"] = results_df["test"].str.replace(r"\(k\s*=\s*\d+\)", "", regex=True).str.strip()

    def models(k, a, b=None, c=None):
        return {
            "linear": lambda k, a, b: a * k + b,
            "quadratisch": lambda k, a, b, c: a * k**2 + b * k + c,
            "logarithmisch": lambda k, a, b: a * np.log(k) + b if np.all(k > 0) else np.full_like(k, np.nan),
            "wurzel": lambda k, a, b: a * np.sqrt(k) + b,
        }

    results = []

    for base_test_name in results_df["base_test"].unique():
        df_k = results_df[(results_df["base_test"] == base_test_name) & results_df["k"].notna()]
        if len(df_k) <= 1:
            print(f"Nicht genug Daten für k-Analyse bei {base_test_name}")
            continue

        k_vals = df_k["k"].to_numpy()
        times = df_k["avg_time"].to_numpy()
        model_funcs = models(k_vals, None, None, None)

        for model_name, func in model_funcs.items():
            try:
                if model_name == "quadratisch":
                    popt, _ = curve_fit(func, k_vals, times, maxfev=10000)
                else:
                    popt, _ = curve_fit(func, k_vals, times)
                predicted = func(k_vals, *popt)
                r2 = r2_score(times, predicted)
                # Parameter korrekt benennen
                param_names = ["a", "b", "c"][:len(popt)]
                row = {"Test": base_test_name, "model": model_name, "r2": r2}
                for name, val in zip(param_names, popt):
                    row[f"param_{name}"] = val
                results.append(row)
            except Exception as e:
                print(f"Fit Fehler bei {base_test_name} Modell {model_name}: {e}")
                continue

    return pd.DataFrame(results)



def runtime_analysis(df, complexity_func):
    n_vals = df["Zahl"].astype(float)
    times = df["avg_time"]

    stats = {
        "time_min": times.min(),
        "time_max": times.max(),
        "time_avg": times.mean(),
        "time_std": times.std()
    }

    predicted = complexity_func(n_vals)
    def model(n, a): return a * complexity_func(n)
    popt, _ = curve_fit(model, n_vals, times, maxfev=10000)
    residuals = times - model(n_vals, *popt)
    r2 = 1 - (np.sum(residuals**2) / np.sum((times - np.mean(times))**2))

    stats["fitted_a"] = popt[0]
    stats["r2"] = r2
    return stats

def fit_runtime_complexities(detail_df, complexity_funcs, complexity_notation):
    fit_results = []

    # Standardmodelle für praktische Laufzeit
    standard_models = {
        "O(1)": lambda n: np.ones_like(n),
        "O(log n)": lambda n: np.log(n),
        "O(log^2 n)": lambda n: np.log(n)**2,
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: n * np.log(n),
        "O(n^2)": lambda n: n**2,
    }

    for test_name in detail_df["Test"].str.replace(r"\(k\s*=\s*\d+\)", "", regex=True).str.strip().unique():
        df_test = detail_df[detail_df["Test"].str.contains(test_name)]

        # Aggregation über k: Mittelwert der Laufzeiten pro n
        df_agg = df_test.groupby("Zahl", as_index=False)["avg_time"].mean()
        n_vals = df_agg["Zahl"].to_numpy()
        times = df_agg["avg_time"].to_numpy()

        # --- Fit theoretische Laufzeit ---
        th_func = complexity_funcs.get(test_name)
        th_label = complexity_notation.get(test_name, "unbekannt")
        if th_func is not None:
            x_th = th_func(n_vals).reshape(-1,1)
            linreg = LinearRegression().fit(x_th, times)
            y_th_pred = linreg.predict(x_th)
            a_th = linreg.coef_[0]
            b_th = linreg.intercept_
            r2_th = r2_score(times, y_th_pred)
        else:
            th_label = "unbekannt"
            a_th = b_th = r2_th = np.nan

        # --- Fit praktische Laufzeit ---
        best_model_name = None
        best_r2 = -np.inf
        best_a = best_b = None
        for model_name, model_func in standard_models.items():
            try:
                x_model = model_func(n_vals).reshape(-1,1)
                if np.std(x_model) < 1e-8:
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

def error_analysis(detail_df):
    df = detail_df.copy()

    # k extrahieren
    df["k"] = df["Test"].apply(
        lambda x: int(re.search(r"k\s*=\s*(\d+)", x).group(1)) if re.search(r"k\s*=\s*(\d+)", x) else 1
    )

    # Basis-Testnamen ohne k
    df["base_test"] = df["Test"].str.replace(r"\(k\s*=\s*\d+\)", "", regex=True).str.strip()

    # 1. Gruppieren nach Test und k (abhängig von k)
    agg_k = df.groupby(["base_test", "k"]).agg(
        false_pos_sum=("false_positive", "sum"),
        false_neg_sum=("false_negative", "sum"),
        avg_error_rate=("is_error", "mean")  # falls "is_error" 1/0 für Fehler ist
    ).reset_index()

    # 2. Gesamtauswertung unabhängig von k (nur 1 Zeile pro Test)
    agg_total = df.groupby("base_test").agg(
        false_pos_sum=("false_positive", "sum"),
        false_neg_sum=("false_negative", "sum"),
        avg_error_rate=("is_error", "mean")
    ).reset_index()
    agg_total["k"] = "all"  # Kennzeichnung, dass alle k zusammengefasst sind

    # Beide DataFrames zurückgeben
    return agg_k, agg_total



# ----------------------------------------------------
# Main-Auswertung
# ----------------------------------------------------
def analyse_folder(folder_path, complexity_funcs):
    data_all = read_csv_folder(folder_path)
    all_avg = []
    all_detail = []

    for run in data_all:
        df_avg = parse_avg_data(run["avg"])
        df_detail = parse_detail_data(run["data"])
        all_avg.append(df_avg)
        all_detail.append(df_detail)

    avg_df = pd.concat(all_avg, ignore_index=True)
    detail_df = pd.concat(all_detail, ignore_index=True)

    # Fehleranalyse: gibt zwei DataFrames zurück
    err_k_df, err_total_df = error_analysis(detail_df)

    results = []
    for test in detail_df["Test"].unique():
        df_test = detail_df[detail_df["Test"] == test]

        # k extrahieren
        match = re.search(r"k\s*=\s*(\d+)", test)
        k_val = int(match.group(1)) if match else 1

        # passende theoretische Laufzeit finden
        if any(name in test for name in complexity_funcs):
            base_name = next(name for name in complexity_funcs if name in test)
            runtime_res = runtime_analysis(df_test, complexity_funcs[base_name])
        else:
            base_name = None
            runtime_res = {}

        # Fehlerwerte aus k-abhängigem DataFrame extrahieren
        base_test_name = test.replace(f"(k = {k_val})", "").strip()
        err_row = err_k_df[(err_k_df["base_test"] == base_test_name) & (err_k_df["k"] == k_val)]
        if len(err_row) == 0:
            false_pos_sum = 0
            false_neg_sum = 0
            avg_error_rate = 0
        else:
            false_pos_sum = err_row["false_pos_sum"].values[0]
            false_neg_sum = err_row["false_neg_sum"].values[0]
            avg_error_rate = err_row["avg_error_rate"].values[0]

        results.append({
            "test": test,
            "k": k_val,
            "theoretical": base_name,
            **runtime_res,
            "best_time": df_test["best_time"].min(),
            "avg_time": df_test["avg_time"].mean(),
            "worst_time": df_test["worst_time"].max(),
            "std_dev": df_test["avg_time"].std(),
            "false_pos_sum": false_pos_sum,
            "false_neg_sum": false_neg_sum,
            "avg_error_rate": avg_error_rate
        })

    results_df = pd.DataFrame(results)

    # 1. Gesamtauswertung unabhängig von k
    total_stats = results_df.groupby("test").agg(
        avg_false_positive=("false_pos_sum", "mean"),
        avg_false_negative=("false_neg_sum", "mean"),
        avg_error_rate=("avg_error_rate", "mean"),
        avg_best_time=("best_time", "mean"),
        avg_avg_time=("avg_time", "mean"),
        avg_worst_time=("worst_time", "mean"),
        avg_std_dev=("std_dev", "mean")
    ).reset_index()

    # Sortieren nach avg_avg_time
    total_stats_sorted = total_stats.sort_values(by="avg_avg_time", ascending=False)

    print("\n--- Gesamtauswertung (sortiert nach avg_avg_time) ---")
    print(total_stats_sorted.to_string(index=False))

    # 2. Laufzeit-Statistik sortiert nach avg_time
    print("\n--- Laufzeitanalyse - Statistik (sortiert nach avg_time) ---")
    runtime_sorted = results_df.sort_values(by="avg_time", ascending=False)
    print(runtime_sorted[["test", "k", "best_time", "avg_time", "worst_time", "std_dev"]]
          .to_string(index=False, float_format="%.10f"))

    # 3. Laufzeit-K-Analyse (lineare Abhängigkeit testen)
    print("\n--- Laufzeitanalyse - Einfluss von k ---")
    k_analysis_df = analyse_k_influence(results_df)
    k_analysis_df_sorted = k_analysis_df.sort_values(by="r2", ascending=False)
    print(k_analysis_df_sorted.to_string(index=False, float_format="%.10f"))

    # 4. Laufzeit-Fit-Analyse unabhängig von k
    print("\n--- Laufzeitanalyse - Fit-Güte ---")
    fit_df = fit_runtime_complexities(detail_df, complexity_funcs, complexity_notation)
    print(fit_df)

    # 5. Fehleranalyse
    print("\n--- Fehleranalyse - Abhängig von k ---")
    err_k_sorted = err_k_df.sort_values(by="avg_error_rate", ascending=False)
    print(err_k_sorted.to_string(index=False, float_format="%.10f"))

    print("\n--- Fehleranalyse - Gesamt (unabhängig von k) ---")
    err_total_sorted = err_total_df.sort_values(by="avg_error_rate", ascending=False)
    print(err_total_sorted.to_string(index=False, float_format="%.10f"))

    # CSV speichern in gewünschter Reihenfolge
    csv_df = pd.concat([
        total_stats.assign(section="Gesamtauswertung"),
        results_df.assign(section="Laufzeit-Statistik"),
        k_analysis_df.assign(section="Laufzeit-K-Analyse"),
        fit_df.assign(section="Laufzeit-Fit-Analyse"),
        err_k_df.assign(section="Fehleranalyse_k"),
        err_total_df.assign(section="Fehleranalyse_gesamt")
    ], ignore_index=True, sort=False)

    csv_df.to_csv("p3-result/pool3_analysis.csv", index=False)

    return results_df, err_k_df, err_total_df



if __name__ == "__main__":
    standard_models = {
        "O(1)": lambda n: np.ones_like(n),
        "O(log n)": lambda n: np.log(n),
        "O(log^2 n)": lambda n: np.log(n)**2,
        "O(log^3 n)": lambda n: np.log(n)**3,
        "O(log^4 n)": lambda n: np.log(n)**4,
        "O(log^5 n)": lambda n: np.log(n)**5,
        "O(log^6 n)": lambda n: np.log(n)**6,
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: n * np.log(n),
        "O(n log^2 n)": lambda n: n * np.log(n)**2,
        "O(n^2)": lambda n: n**2,
        "O(n^2 log n)": lambda n: n**2 * np.log(n),
        "O(n^3)": lambda n: n**3,
        "O(n^3 log n)": lambda n: n**3 * np.log(n),
        "O(n^4)": lambda n: n**4,
        "O(2^n)": lambda n: 2**n,
        "O(n!)": lambda n: np.array([np.math.factorial(int(x)) for x in n]),
        "O(sqrt(n))": lambda n: np.sqrt(n),
        "O(n^(1/3))": lambda n: n**(1/3)
    }
    
    complexity_funcs = {
        "Fermat": lambda n: (np.log(n)**3),
        "Miller-Selfridge-Rabin": lambda n: (np.log(n)**4),
        "Solovay-Strassen": lambda n: (np.log(n)**3),
    }
    complexity_notation = {
        "Fermat": "O(log^3 n)",
        "Miller-Selfridge-Rabin": "O(log^4 n)",
        "Solovay-Strassen": "O(log^3 n)",
    }
    
    folder = "C:\\Users\\julia\\Downloads\\testpool3"
    analyse_folder(folder, complexity_funcs)
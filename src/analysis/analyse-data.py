# -*- coding: utf-8 -*-
import os, csv, re, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =============================================================================
# Globale Einstellungen
# =============================================================================
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------
# Kategorien (für Pool1 & Pool2)
# ----------------------------------------------------
CATEGORY_MAP = {
    "Probabilistische Tests": ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen"],
    "Lucas-Tests": ["Initial Lucas", "Lucas", "Optimized Lucas",],
    "Langsame Tests": ["Wilson", "AKS10"],
    "Spezielle Tests": ["Pepin", "Lucas-Lehmer"],
    "Zusammengesetzte": ["Proth", "Proth Variant", "Pocklington", "Optimized Pocklington", "Optimized Pocklington Variant", "Generalized Pocklington", "Rao", "Ramzy"],
}
def classify_test(test_name: str) -> str:
    for group, patterns in CATEGORY_MAP.items():
        for p in patterns:
            if p in test_name:
                return group
    return "Spezielle Tests"  # Fallback

# =============================================================================
# Laden/Parsen
# =============================================================================
def read_pool1_csv(file_path: str) -> pd.DataFrame:
    """
    Liest eine Pool-1 CSV ein und gibt einen DataFrame im Detail-Format zurück.
    Erwartete Detail-Header: 'Gruppe,Test,Zahl,...'
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

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(headers):
                parts += [""] * (len(headers) - len(parts))  # auffüllen

            row = dict(zip(headers, parts))

            # Typkonvertierung
            try:
                row["Zahl"] = pd.to_numeric(row.get("Zahl", ""), errors="coerce")
            except Exception:
                row["Zahl"] = None

            for col in [
                "false_positive","false_negative","error_rate",
                "best_time","avg_time","worst_time","std_dev"
            ]:
                val = row.get(col, "")
                if isinstance(val, str):
                    val = val.replace(" ms", "").strip()
                try:
                    row[col] = float(val)
                except Exception:
                    row[col] = None

            for col in ["true_prime", "is_error", "false_positive", "false_negative"]:
                val = row.get(col, "")
                if isinstance(val, str):
                    row[col] = 1 if val.strip().lower() == "true" else 0
                else:
                    row[col] = 0

            detail_rows.append(row)

    df = pd.DataFrame(detail_rows)
    if not df.empty:
        df["category"] = df["Test"].apply(classify_test)
    return df


def load_all_pool1(folder_path: str) -> pd.DataFrame:
    """
    Lädt alle Pool-1 CSVs aus einem Ordner in einen kombinierten DataFrame.
    """
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = read_pool1_csv(file_path)
            if not df.empty:
                all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    return combined_df


def read_pool2_folder(folder_path: str):
    """
    Liest alle Pool-3 CSVs und extrahiert 'test_avg' & Detailzeilen ('Probabilistisch').
    """
    results = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".csv"):
            continue
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
        results.append({"file": fname, "avg": avg_entries, "data": data_entries})
    return results


def parse_pool2_avg(avg_entries) -> pd.DataFrame:
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


def parse_pool2_detail(detail_entries) -> pd.DataFrame:
    headers = [
        "Gruppe","Test","Zahl","Ergebnis","true_prime","is_error",
        "false_positive","false_negative","error_rate","best_time","avg_time",
        "worst_time","std_dev","a_values","other_fields","reason"
    ]
    df = pd.DataFrame(detail_entries, columns=headers)

    # Zahlenfelder
    numeric_cols = ["Zahl","error_rate","best_time","avg_time","worst_time","std_dev"]
    for col in numeric_cols:
        df[col] = df[col].apply(
            lambda x: float(str(x).replace(" ms","")) if x not in [None,""," "] else np.nan
        )

    # Bools zu 0/1
    bool_cols = ["false_positive","false_negative","is_error","true_prime"]
    for col in bool_cols:
        df[col] = df[col].astype(str).str.strip().map(lambda v: 1 if v.lower() == "true" else 0)
    return df


def load_all_pool2(folder_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lädt alle Pool-2 CSVs → (avg_df, detail_df).
    """
    raw = read_pool2_folder(folder_path)
    all_avg, all_detail = [], []
    for r in raw:
        df_avg = parse_pool2_avg(r["avg"])
        df_det = parse_pool2_detail(r["data"])
        if not df_avg.empty:
            all_avg.append(df_avg)
        if not df_det.empty:
            all_detail.append(df_det)
    avg_df = pd.concat(all_avg, ignore_index=True) if all_avg else pd.DataFrame()
    detail_df = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    if not detail_df.empty:
        detail_df["category"] = detail_df["Test"].apply(classify_test)
    return avg_df, detail_df

# =============================================================================
# Gemeinsame Analysen
# =============================================================================
def analyse_overall(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert alle Detaildaten je Test und Kategorie (Durchschnittswerte).
    """
    if detail_df.empty:
        return pd.DataFrame()

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


def runtime_stats(detail_df: pd.DataFrame) -> pd.DataFrame:
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


def error_stats(detail_df: pd.DataFrame) -> pd.DataFrame:
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

def group_summaries(overall_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Schnellster & Genauester Test pro Kategorie (nur Pool1 gefordert).
    """
    if overall_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    fastest_list = []
    most_accurate_list = []

    for cat_name, test_list in CATEGORY_MAP.items():
        # Filter auf Tests dieser Kategorie
        cat_df = overall_df[overall_df["Test"].apply(lambda x: any(t in x for t in test_list))]
        if cat_df.empty:
            continue

        # Schnellster Test
        fastest = cat_df.loc[cat_df["avg_avg_time"].idxmin()].copy()
        fastest["category"] = cat_name  # Kategorie aus CATEGORY_MAP erzwingen
        fastest_list.append(fastest)

        # Genauester Test
        most_accurate = cat_df.loc[cat_df["avg_error_rate"].idxmin()].copy()
        most_accurate["category"] = cat_name
        most_accurate_list.append(most_accurate)

    fastest_df = pd.DataFrame(fastest_list).reset_index(drop=True)
    most_accurate_df = pd.DataFrame(most_accurate_list).reset_index(drop=True)

    return fastest_df, most_accurate_df


# -------------------------
# Runtime Complexity (Unified)
# -------------------------
def fit_runtime_complexities_unified(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Führt Laufzeit-Fits (theoretisch & praktisch) für einen Detail-DataFrame durch.
    Erwartet Spalten: ['Test','Zahl','avg_time'] (+ optional 'category').
    """
    if detail_df.empty:
        return pd.DataFrame()

    fit_results = []

    # Praktische Standardmodelle (erweitert) – robust gegen Overflows
    def safe_factorial_array(n):
        out = []
        for x in n:
            try:
                if x < 20:
                    out.append(float(np.math.factorial(int(x))))
                else:
                    out.append(np.inf)
            except Exception:
                out.append(np.inf)
        return np.array(out, dtype=float)

    standard_models = {
        "O(1)":             lambda n: np.ones_like(n, dtype=float),
        "O(log n)":         lambda n: np.log(n),
        "O(log^2 n)":       lambda n: np.log(n)**2,
        "O(log^3 n)":       lambda n: np.log(n)**3,
        "O(log^4 n)":       lambda n: np.log(n)**4,
        "O(log^5 n)":       lambda n: np.log(n)**5,
        "O(log^6 n)":       lambda n: np.log(n)**6,
        "O(log^7 n)":       lambda n: np.log(n)**7,
        "O(log^8 n)":       lambda n: np.log(n)**8,
        "O(log^9 n)":       lambda n: np.log(n)**9,
        "O(log^10 n)":       lambda n: np.log(n)**10,
        "O(n)":             lambda n: n.astype(float),
        "O(n log n)":       lambda n: n * np.log(n),
        "O(n log^2 n)":     lambda n: n * (np.log(n)**2),
        "O(n log^3 n)":     lambda n: n * (np.log(n)**3),
        "O(n log^4 n)":     lambda n: n * (np.log(n)**4),
        "O(n log^5 n)":     lambda n: n * (np.log(n)**5),
        "O(n^2)":           lambda n: n**2,
        "O(n^2 log n)":     lambda n: (n**2) * np.log(n),
        "O(n^2 log^2 n)":     lambda n: (n**2) * np.log(n)**2,
        "O(n^2 log^3 n)":     lambda n: (n**2) * np.log(n)**3,
        "O(n^3)":           lambda n: n**3,
        "O(n^3 log n)":     lambda n: (n**3) * np.log(n),
        "O(n^3 log^2 n)":     lambda n: (n**3) * np.log(n)**2,
        "O(n^3 log^3 n)":     lambda n: (n**3) * np.log(n)**3,
        "O(2^n)":           lambda n: np.power(2.0, n, where=np.isfinite(n), dtype=float),
        "O(n!)":            lambda n: safe_factorial_array(n),
        "O(sqrt(n))":       lambda n: np.sqrt(n),
        "O(n^(1/3))":       lambda n: np.cbrt(n),
        "O(n^(1/4))":       lambda n: np.power(n, 0.25),
    }

    # Theoretische Komplexitäten
    complexity_funcs = {
        # Probabilistische
        "Fermat":                        lambda n: np.log(n)**3,
        "Miller-Selfridge-Rabin":        lambda n: np.log(n)**4,
        "Solovay-Strassen":              lambda n: np.log(n)**3,
        # Lucas-Familie
        "Initial Lucas":                 lambda n: n**2 * np.log(n)**3,
        "Lucas":                         lambda n: n * np.log2(n) * np.log(n)**3,
        "Optimized Lucas":               lambda n: n * np.log(n)**3,
        # Langsame
        "Wilson":                        lambda n: n * np.log(n)**2,
        "AKS10":                         lambda n: np.log(n)**18,
        # Zusammengesetzte
        "Proth":                         lambda n: np.log(n)**3,
        "Proth Variant":                 lambda n: np.log(n)**3,
        "Pocklington":                   lambda n: np.log(n)**3,
        "Optimized Pocklington":         lambda n: np.log2(n) * np.log(n)**3, 
        "Optimized Pocklington Variant": lambda n: np.log2(n) * np.log(n)**3, 
        "Generalized Pocklington":       lambda n: np.log2(n) * np.log(n)**3, 
        "Rao":                           lambda n: (np.log(n)**2) * n,
        "Ramzy":                         lambda n: np.log(np.log(n)) * (np.log(n)**3),
        # Spezielle
        "Pepin":                         lambda n: 2**n * np.log(2),
        "Lucas-Lehmer":                  lambda n: np.log(n)**2 * np.log2(n)
    }
    complexity_notation = {
        # Probabilistische
        "Fermat": "O((log n)^3)",
        "Miller-Selfridge-Rabin": "O((log n)^4)",
        "Solovay-Strassen": "O((log n)^3)",
        # Lucas-Familie
        "Initial Lucas": "O(n^2 (log n)^3)",
        "Lucas": "O(n log n * (log n)^3)",
        "Optimized Lucas": "O(n (log n)^3)",
        # Langsame
        "Wilson": "O(n (log n)^2)",
        "AKS10": "O((log n)^{18})",
        # Zusammengesetzte
        "Proth": "O((log n)^3)",
        "Proth Variant": "O((log n)^3)",
        "Pocklington": "O((log n)^3)",
        "Optimized Pocklington": "O(log n * (log n)^3)",
        "Optimized Pocklington Variant": "O(log n * (log n)^3)",
        "Generalized Pocklington": "O(log n * (log n)^3)",
        "Rao": "O(n (log n)^2)",
        "Ramzy": "O(log(log n) * (log n)^3)",
        # Spezielle
        "Pepin": "O(2^n * log(2))",
        "Lucas-Lehmer": "O((log n)^2 * log n)"
    }

    # Basis-Testnamen ohne (k = ...)
    base_names = (
        detail_df["Test"].str.replace(r"\(k\s*=\s*\d+\)", "", regex=True).str.strip().unique()
    )
    for test_name in base_names:
        df_test = detail_df[detail_df["Test"].str.contains(re.escape(test_name))]
        if df_test.empty:
            continue

        # Aggregation: Mittelwert der avg_time pro Zahl
        df_agg = df_test.groupby("Zahl", as_index=False)["avg_time"].mean().dropna()
        if df_agg.empty:
            continue

        n_vals = df_agg["Zahl"].to_numpy(dtype=float)
        times = df_agg["avg_time"].to_numpy(dtype=float)
        finite_mask = np.isfinite(n_vals) & np.isfinite(times)
        n_vals, times = n_vals[finite_mask], times[finite_mask]
        if len(n_vals) < 2:
            continue

        # --- Fit theoretische Laufzeit ---
        th_func = complexity_funcs.get(test_name)
        th_label = complexity_notation.get(test_name, "unbekannt")
        if th_func is not None:
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    # Spezieller Fall für Pepin
                    if test_name == "Pepin":
                        # log-Fit, um Overflow zu vermeiden
                        x_th = n_vals.reshape(-1, 1)   # exponent in 2^n
                        y_th = np.log(times)           # log(y) statt y
                        linreg = LinearRegression().fit(x_th, y_th)
                        y_th_pred = linreg.predict(x_th)
                        a_th = float(linreg.coef_[0])
                        b_th = float(linreg.intercept_)
                        r2_th = float(r2_score(y_th, y_th_pred))
                    else:
                        x_th = th_func(n_vals).astype(float).reshape(-1, 1)
                        if np.all(np.isfinite(x_th)) and np.std(x_th) >= 1e-12:
                            linreg = LinearRegression().fit(x_th, times)
                            y_th_pred = linreg.predict(x_th)
                            a_th = float(linreg.coef_[0])
                            b_th = float(linreg.intercept_)
                            r2_th = float(r2_score(times, y_th_pred))
                        else:
                            a_th = b_th = r2_th = np.nan
            except Exception:
                a_th = b_th = r2_th = np.nan
        else:
            a_th = b_th = r2_th = np.nan
            th_label = "unbekannt"

        # --- Fit praktische Laufzeit (Best-of) ---
        best_model_name = None
        best_r2 = -np.inf
        best_a = best_b = np.nan
        for model_name, model_func in standard_models.items():
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    x_model = np.asarray(model_func(n_vals), dtype=float).reshape(-1, 1)
                if not np.all(np.isfinite(x_model)):
                    continue
                if np.std(x_model) < 1e-12:
                    continue
                linreg = LinearRegression().fit(x_model, times)
                y_pred = linreg.predict(x_model)
                r2 = r2_score(times, y_pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
                    best_a = float(linreg.coef_[0])
                    best_b = float(linreg.intercept_)
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
            "r2_prak": float(best_r2) if np.isfinite(best_r2) else np.nan
        })

    res = pd.DataFrame(fit_results)
    if not res.empty:
        res["category"] = res["Test"].apply(classify_test)
    return res

# =============================================================================
# POOL 2 – K-Analyse & Fehleranalyse (k-spezifisch)
# =============================================================================
def build_pool2_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baut ein kompaktes Ergebnis-DF je Test(k) für POOL2 (für k-Analyse).
    """
    if detail_df.empty:
        return pd.DataFrame(columns=["test","k","avg_time","best_time","worst_time","std_dev"])

    def extract_k(name: str) -> int:
        m = re.search(r"\(k\s*=\s*(\d+)\)", str(name))
        return int(m.group(1)) if m else 1

    tmp = detail_df.copy()
    tmp["k"] = tmp["Test"].apply(extract_k)

    results = (
        tmp.groupby("Test", as_index=False)
           .agg(
               k=("k","first"),
               avg_time=("avg_time","mean"),
               best_time=("best_time","min"),
               worst_time=("worst_time","max"),
               std_dev=("avg_time","std")
           )
           .rename(columns={"Test":"test"})
    )
    return results


def analyse_k_influence_pool2(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Beispiel-Ausgabe für mehrere Modelle sortiert nach R2 pro Basis-Test (POOL3).
    Erwartet Spalten: ['test','k','avg_time'].
    """
    if results_df.empty:
        return pd.DataFrame(columns=["Test","model","r2","param_a","param_b","param_c"])

    # Basis-Testnamen ohne (k = ...)
    results_df = results_df.copy()
    results_df["base_test"] = results_df["test"].str.replace(r"\(k\s*=\s*\d+\)", "", regex=True).str.strip()

    def models():
        return {
            "linear":        lambda k, a, b: a * k + b,
            "quadratisch":   lambda k, a, b, c: a * k**2 + b * k + c,
            "logarithmisch": lambda k, a, b: a * np.log(k) + b,
            "wurzel":        lambda k, a, b: a * np.sqrt(k) + b,
        }

    out = []
    for base_test_name in results_df["base_test"].unique():
        df_k = results_df[(results_df["base_test"] == base_test_name) & results_df["k"].notna()]
        if len(df_k) <= 1:
            continue

        k_vals = df_k["k"].to_numpy(dtype=float)
        times = df_k["avg_time"].to_numpy(dtype=float)
        model_funcs = models()

        for model_name, func in model_funcs.items():
            try:
                # Anzahl Parameter je Modell
                if model_name == "quadratisch":
                    p0 = [1.0, 1.0, 1.0]
                else:
                    p0 = None
                popt, _ = _curve_fit_dispatch(func, k_vals, times, p0=p0)
                predicted = func(k_vals, *popt)
                r2 = r2_score(times, predicted)
                row = {"Test": base_test_name, "model": model_name, "r2": r2}
                # Parameter korrekt benennen
                param_names = ["a","b","c"][:len(popt)]
                for name, val in zip(param_names, popt):
                    row[f"param_{name}"] = val
                out.append(row)
            except Exception:
                continue
    return pd.DataFrame(out)


def _curve_fit_dispatch(func, x, y, p0=None):
    """
    Hilfsfunktion: robustes curve_fit mit maxfev.
    """
    from scipy.optimize import curve_fit
    if p0 is not None:
        return curve_fit(func, x, y, p0=p0, maxfev=10000)
    return curve_fit(func, x, y, maxfev=10000)


def analyse_overall_independent_k(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert alle Detaildaten je Test und Kategorie, unabhängig von k.
    """
    if detail_df.empty:
        return pd.DataFrame()

    # Entferne "(k = …)" aus Testnamen
    df = detail_df.copy()
    df["base_test"] = df["Test"].str.replace(r"\s*\(k\s*=\s*\d+\)", "", regex=True)

    overall_indep = (
        df
        .groupby(["base_test","category"], as_index=False)
        .agg(
            avg_false_positive=("false_positive","mean"),
            avg_false_negative=("false_negative","mean"),
            avg_error_rate=("error_rate","mean"),
            avg_best_time=("best_time","mean"),
            avg_avg_time=("avg_time","mean"),
            avg_worst_time=("worst_time","mean"),
            avg_std_dev=("std_dev","mean")
        )
        .sort_values(["category","avg_avg_time","avg_error_rate"], ascending=[True, True, True])
    )
    return overall_indep


def runtime_stats_independent_k(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Laufzeit-Statistik je Test, unabhängig von k.
    """
    if detail_df.empty:
        return pd.DataFrame()

    # Entferne "(k = …)" aus Testnamen
    df = detail_df.copy()
    df["base_test"] = df["Test"].str.replace(r"\s*\(k\s*=\s*\d+\)", "", regex=True)

    rt_indep = (
        df
        .groupby(["base_test","category"], as_index=False)
        .agg(
            best_time_min=("best_time","min"),
            time_avg=("avg_time","mean"),
            worst_time_max=("worst_time","max"),
            time_std=("avg_time","std"),
        )
        .sort_values(["category","time_avg"], ascending=[True, True])
    )
    return rt_indep

def error_stats_independent_k(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fehleranalyse je Test, unabhängig von k.
    Aggregiert alle Detaildaten pro Basis-Test (ohne k) für Pool2.
    """
    if detail_df.empty:
        return pd.DataFrame()

    df = detail_df.copy()
    # k aus Testnamen entfernen, Basis-Test extrahieren
    df["base_test"] = df["Test"].str.replace(r"\(k\s*=\s*\d+\)", "", regex=True).str.strip()

    # Fehleranalyse pro Basis-Test
    err_total = (
        df.groupby("base_test", as_index=False)
          .agg(
              false_pos_sum=("false_positive","sum"),
              false_neg_sum=("false_negative","sum"),
              avg_error_rate=("is_error","mean")
          )
          .sort_values("avg_error_rate", ascending=True)
    )
    return err_total


# =============================================================================
# Export
# =============================================================================
def export_df(df: pd.DataFrame, filename: str):
    if not isinstance(df, pd.DataFrame):
        print(f"Error: {filename} ist kein DataFrame!")
        return
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    df.to_csv(filename, index=False)
    #print(f"Exportiert: {filename}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # >>>>> Pfade anpassen <<<<<
    folder1 = "C:\\Users\\julia\\Downloads\\T1"
    folder2 = "C:\\Users\\julia\\Downloads\\T2"

# ------------------------------------------------
    # Daten einlesen
    # ------------------------------------------------
    print("\n=== Schritt 1: Daten laden ===")
    data1 = load_all_pool1(folder1)
    avg2, data2 = load_all_pool2(folder2)
    print(f"Pool1 Detail: {len(data1)} Zeilen")
    print(f"Pool2 Detail: {len(data2)} Zeilen")

    # ------------------------------------------------
    # ANALYSE
    # ------------------------------------------------
    print("\n=== Schritt 3a (BEIDE): Gesamtauswertung (analyse_overall) ===")
    print("\n--- Pool1: Gesamtauswertung ---")
    overall1 = analyse_overall(data1)
    overall1_sorted = overall1.sort_values(by="avg_avg_time", ascending=True)
    print(overall1_sorted.to_string(index=False, float_format="%.10f"))
    
    print("\n=== Schritt 4 (BEIDE): Laufzeit-Statistik (sortiert) ===")
    rt1 = runtime_stats(data1).sort_values("time_avg", ascending=True)
    rt2 = runtime_stats(data2).sort_values("time_avg", ascending=True)
    print("\n--- Pool1: Laufzeit-Statistik ---")
    print(rt1.to_string(index=False, float_format="%.10f"))

    print("\n=== Schritt 5 (BEIDE): Runtime-Complexity (Fit theoretisch & praktisch) ===")
    fit1 = fit_runtime_complexities_unified(data1)
    fit2 = fit_runtime_complexities_unified(data2)
    fit1_sorted = fit1.sort_values("r2_th", ascending=True)
    fit2_sorted = fit2.sort_values("r2_th", ascending=True)
    print("\n--- Pool1: Fit-Statistik ---")
    print(fit1_sorted.to_string(index=False, float_format="%.10f"))


    print("\n=== Schritt 6 (BEIDE): Fehleranalyse ===")
    err1 = error_stats(data1).sort_values(["category","error_rate_avg"], ascending=[True, True])
    err2 = error_stats(data2).sort_values(["category","error_rate_avg"], ascending=[True, True])
    print("\n--- Pool1: Fehleranalyse ---")
    print(err1.to_string(index=False, float_format="%.10f"))

    print("\n=== Schritt 7 (POOL1): Gruppen-Zusammenfassung ===")
    fastest, accurate = group_summaries(overall1)
    print("\n--- Pool1: Schnellster Test je Kategorie ---")
    print(fastest.sort_values("category").to_string(index=False, float_format="%.10f"))
    print("\n--- Pool1: Genauester Test je Kategorie ---")
    print(accurate.sort_values("category").to_string(index=False, float_format="%.10f"))


    print("\n=== Schritt 2 (POOL2): Analyse von k (analyse_k_influence) ===")
    pool2_results = build_pool2_results(data2)
    k_analysis_df = analyse_k_influence_pool2(pool2_results)
    k_analysis_sorted = k_analysis_df.sort_values(["model","r2"], ascending=[False, False])
    print(k_analysis_sorted.to_string(index=False, float_format="%.10f"))

    print("\n=== Schritt 3a (BEIDE): Gesamtauswertung (analyse_overall) ===")
    overall2 = analyse_overall(data2)
    overall2_sorted = overall2.sort_values(by="avg_avg_time", ascending=True)
    print("\n--- Pool2: Gesamtauswertung ---")
    print(overall2_sorted.to_string(index=False, float_format="%.10f"))
    print("\n--- Pool2: Gesamtauswertung (unabhängig von k) ---")
    overall2_indep = analyse_overall_independent_k(data2)
    overall2_indep_sorted = overall2_indep.sort_values(by="avg_avg_time", ascending=True)
    print(overall2_indep_sorted.to_string(index=False, float_format="%.10f"))

    print("\n=== Schritt 4 (BEIDE): Laufzeit-Statistik (sortiert) ===")
    print("\n--- Pool2: Laufzeit-Statistik ---")
    print(rt2.to_string(index=False, float_format="%.10f"))
    print("\n--- Pool2: Laufzeit-Statistik (unabhängig von k) ---")
    rt2_indep = runtime_stats_independent_k(data2).sort_values("time_avg", ascending=True)
    print(rt2_indep.to_string(index=False, float_format="%.10f"))

    print("\n=== Schritt 5 (BEIDE): Runtime-Complexity (Fit theoretisch & praktisch) ===")
    print("\n--- Pool2: Fit-Statistik ---")
    print(fit2_sorted.to_string(index=False, float_format="%.10f"))

    print("\n=== Schritt 6 (BEIDE): Fehleranalyse ===")
    print("\n--- Pool2: Fehleranalyse ---")
    print(err2.to_string(index=False, float_format="%.10f"))
    
    print("\n--- Pool2: Fehleranalyse (gesamt, unabhängig von k) ---")
    err_total = error_stats_independent_k(data2)
    print(err_total.to_string(index=False, float_format="%.10f"))

    export_df(overall1_sorted, "results/p1_1_overall.csv")
    export_df(rt1, "results/p1_2_runtime_stats.csv")
    export_df(fit1_sorted, "results/p1_3_fit_stats.csv")
    export_df(err1, "results/p1_4_error_stats.csv")
    export_df(fastest, "results/p1_5_fastest_by_group.csv")
    export_df(accurate, "results/p1_6_most_accurate_by_group.csv")
    export_df(pool2_results, "results/p2_1_results_compact.csv")
    export_df(k_analysis_sorted, "results/p2_2_k_influence.csv")
    export_df(overall2_sorted, "results/p2_3_overall.csv")
    export_df(overall2_indep_sorted, "results/p2_4_overall_independent.csv")
    export_df(rt2, "results/p2_5_runtime_stats.csv")
    export_df(rt2_indep, "results/p2_6_runtime_stats_independent.csv")
    export_df(fit2_sorted, "results/p2_7_fit_stats.csv")
    export_df(err2, "results/p2_8_error_stats.csv")
    export_df(err_total, "results/p2_9_error_total.csv")

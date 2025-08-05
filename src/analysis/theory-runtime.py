import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# --- Erweiterte Komplexitätsfunktionen ---
def get_extended_complexity_functions() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "log(n)": lambda n: np.log(n),
        "log^1.5(n)": lambda n: np.log(n) ** 1.5,
        "log²(n)": lambda n: np.log(n) ** 2,
        "log³(n)": lambda n: np.log(n) ** 3,
        "log(n) * log(log(n))": lambda n: np.log(n) * np.log(np.log(n)),
        "sqrt(n)": lambda n: np.sqrt(n),
        "n^0.1": lambda n: n ** 0.1,
        "n^0.25": lambda n: n ** 0.25,
        "n^0.5": lambda n: n ** 0.5,
        "n": lambda n: n,
        "n log(n)": lambda n: n * np.log(n),
        "n²": lambda n: n ** 2,
        "n³": lambda n: n ** 3,
    }

# --- Least Squares Fit ---
def fit_model(x: np.ndarray, y: np.ndarray, transform: Callable[[np.ndarray], np.ndarray]) -> tuple:
    x_t = transform(x).reshape(-1, 1)
    x_t = np.hstack([x_t, np.ones_like(x_t)])
    beta, _, _, _ = np.linalg.lstsq(x_t, y, rcond=None)
    y_pred = x_t @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return beta, y_pred, r2

# --- Binärlänge als Eingabegröße ---
def binary_length(n: np.ndarray) -> np.ndarray:
    return np.floor(np.log2(n)).astype(int) + 1

# --- Dynamisches Headerlesen ---
def load_csv_with_dynamic_header(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("Gruppe,Test,Zahl"):
            return pd.read_csv(filepath, skiprows=i, encoding="utf-8")
    raise ValueError(f"Kein Header in Datei: {filepath}")

# --- CSVs einlesen ---
def load_all_csvs(data_dir):
    dfs = []
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            try:
                df = load_csv_with_dynamic_header(os.path.join(data_dir, f))
                dfs.append(df)
            except Exception as e:
                print(f"Fehler bei {f}: {e}")
    if not dfs:
        raise RuntimeError("Keine CSV-Dateien geladen.")
    return pd.concat(dfs, ignore_index=True)

# --- Extrahiere Basis-Testname ohne (k = x) ---
def extract_base_testname(testname: str) -> str:
    return re.sub(r"\s*\(k\s*=\s*\d+\)", "", testname).strip()

# --- Extrahiere k aus Testnamen ---
def extract_k(testname: str) -> int:
    m = re.search(r"\(k\s*=\s*(\d+)\)", testname)
    return int(m.group(1)) if m else 1

# --- Ausgabe pro Test und pro k ---
def analyze_complexity_per_k(df: pd.DataFrame):
    funcs = get_extended_complexity_functions()
    print("\n📈 Analyse pro Test & k:")
    for test_name in sorted(df["Test"].unique()):
        subset = df[df["Test"] == test_name]
        grouped = subset.groupby("Zahl", as_index=False)["avg_time"].mean()
        x = grouped["Zahl"].values
        y = grouped["avg_time"].values

        best_fit_n = {"r2": -np.inf}
        best_fit_bin = {"r2": -np.inf}
        best_name_n = ""
        best_name_bin = ""

        for name, func in funcs.items():
            try:
                _, _, r2_n = fit_model(x, y, func)
                if r2_n > best_fit_n["r2"]:
                    best_fit_n = {"name": name, "r2": r2_n}
                    best_name_n = name
            except:
                pass
            try:
                x_bin = binary_length(x)
                _, _, r2_b = fit_model(x_bin, y, func)
                if r2_b > best_fit_bin["r2"]:
                    best_fit_bin = {"name": name, "r2": r2_b}
                    best_name_bin = name
            except:
                pass

        print(f"🔍 {test_name}:")
        print(f"  ▶️  Beste Komplexität mit n Dezimal: {best_name_n}   (R² = {best_fit_n['r2']:.4f})")
        print(f"  ▶️  Beste Komplexität mit Bitlänge : {best_name_bin} (R² = {best_fit_bin['r2']:.4f})")

# --- Zusammenfassung über alle k (Laufzeit normalisiert durch k) ---
def analyze_complexity_overall(df: pd.DataFrame):
    funcs = get_extended_complexity_functions()
    print("\n📊 Zusammenfassung über alle k (mit k-Faktor):")

    df = df.copy()
    df["BaseTest"] = df["Test"].apply(extract_base_testname)
    df["k"] = df["Test"].apply(extract_k)
    df["time_per_k"] = df["avg_time"] / df["k"]

    for test_name in sorted(df["BaseTest"].unique()):
        subset = df[df["BaseTest"] == test_name]
        grouped = subset.groupby("Zahl", as_index=False)["time_per_k"].mean()

        x_n = grouped["Zahl"].values
        x_bin = binary_length(x_n)
        y = grouped["time_per_k"].values

        best_fit_n = {"r2": -np.inf}
        best_fit_bin = {"r2": -np.inf}
        best_name_n = ""
        best_name_bin = ""

        for name, func in funcs.items():
            try:
                _, _, r2_n = fit_model(x_n, y, func)
                if r2_n > best_fit_n["r2"]:
                    best_fit_n = {"name": name, "r2": r2_n}
                    best_name_n = name
            except:
                pass
            try:
                _, _, r2_b = fit_model(x_bin, y, func)
                if r2_b > best_fit_bin["r2"]:
                    best_fit_bin = {"name": name, "r2": r2_b}
                    best_name_bin = name
            except:
                pass

        print(f"🔹 {test_name}:")
        print(f"  ▶️  Beste Komplexität mit n Dezimal: k * {best_name_n}   (R² = {best_fit_n['r2']:.4f})")
        print(f"  ▶️  Beste Komplexität mit Bitlänge : k * {best_name_bin} (R² = {best_fit_bin['r2']:.4f})")

# --- Main ---
def main(data_dir: str):
    df = load_all_csvs(data_dir)
    df['avg_time'] = df['avg_time'].apply(lambda v: float(str(v).split()[0]) if isinstance(v, str) else v)

    analyze_complexity_per_k(df)
    analyze_complexity_overall(df)

# --- Aufruf ---
if __name__ == "__main__":
    main("C:\\Users\\julia\\OneDrive\\Dokumente\\Studium\\Semester M4\\MA\\Datensammlung\\Pool 3 - prob, v2, 1-bd1")
    #main("C:\\Users\\julia\\OneDrive\\Dokumente\\Studium\\Semester M4\\MA\\Datensammlung\\Pool 1 - alle, v1, 1-k1")

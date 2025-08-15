import pandas as pd
import numpy as np
import glob
import os, csv
csv.field_size_limit(10**7)  # Erlaubt Felder bis ca. 10 MB
# Gruppenzuordnung
CATEGORY_MAP = {
    "Zusammengesetzte": [
        "Proth", "Proth Variant", "Pocklington", "Optimized Pocklington", "Optimized Pocklington Variant", "Generalized Pocklington", "Rao", "Ramzy"
    ],
    "Spezielle Tests": [
        "Pepin", "Lucas-Lehmer"
    ],
    "Langsame Tests": [
        "AKS10", "Wilson"
    ],
    "Lucas-Tests": [
        "Initial Lucas", "Lucas", "Optimized Lucas"
    ],
    "Probabilistische Tests": [
        "Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen"
    ]
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
def parse_pool2_csv(path):
    """
    Liest eine Datei im "Pool2"-Format:
    - Meta-Zeilen (ignoriert)
    - test_avg-Zeilen -> kompaktes DF
    - Detailtabelle ("Gruppe,Test,....") -> DF
    """
    avg_rows = []
    detail_rows = []
    detail_headers = None
    in_detail = False

    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            # Leere Zeilen überspringen
            if not row or all((c is None or str(c).strip() == "") for c in row):
                continue

            first = row[0].strip()

            # test_avg-Zeilen: test_avg,<Testname>,key=val, key=val, ...
            if first == "test_avg":
                test_name = row[1].strip()
                rec = {"test": test_name}
                for field in row[2:]:
                    field = field.strip()
                    if not field or "=" not in field:
                        continue
                    k, v = field.split("=", 1)
                    v = v.strip()
                    # "0.065 ms" -> 0.065
                    if v.endswith(" ms"):
                        v = v.replace(" ms", "").strip()
                    try:
                        rec[k.strip()] = float(v)
                    except ValueError:
                        rec[k.strip()] = v
                avg_rows.append(rec)
                continue

            # Beginn der Detailtabelle: Kopfzeile "Gruppe,Test,Zahl,..."
            if not in_detail and first == "Gruppe":
                detail_headers = [c.strip() for c in row]
                in_detail = True
                continue

            # Detailzeilen
            if in_detail:
                # Manche Zeilen haben viele trailing commas -> auf Header-Länge kürzen/padden
                row = list(row[:len(detail_headers)]) + [""] * max(0, len(detail_headers) - len(row))
                detail_rows.append(dict(zip(detail_headers, row)))
                continue

            # Alles andere (Meta-/group_range) ignorieren

    avg_df = pd.DataFrame(avg_rows)

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        # Typkonvertierungen
        def to_float_ms(x):
            s = str(x).strip()
            if s.endswith(" ms"):
                s = s[:-3].strip()
            try:
                return float(s)
            except Exception:
                return np.nan

        num_cols_ms = ["best_time", "avg_time", "worst_time", "std_dev"]
        for c in num_cols_ms:
            if c in detail_df.columns:
                detail_df[c] = detail_df[c].map(to_float_ms)

        if "Zahl" in detail_df.columns:
            detail_df["Zahl"] = pd.to_numeric(detail_df["Zahl"], errors="coerce")

        # Bool/Fehler
        for bc in ["true_prime", "is_error", "false_positive", "false_negative"]:
            if bc in detail_df.columns:
                detail_df[bc] = detail_df[bc].astype(str).str.strip().str.lower().map(
                    {"true": 1, "false": 0}
                ).fillna(0).astype(int)

        if "error_rate" in detail_df.columns:
            detail_df["error_rate"] = pd.to_numeric(detail_df["error_rate"], errors="coerce")

        # Spaltennamen säubern
        if "Test" in detail_df.columns:
            detail_df["test"] = detail_df["Test"].astype(str)
        else:
            detail_df["test"] = ""

    # Kategorie (Gruppe) je Test zuweisen – sowohl für avg als auch für detail
    if not avg_df.empty:
        avg_df["category"] = avg_df["test"].map(classify_test)
    if not detail_df.empty:
        detail_df["category"] = detail_df["test"].map(classify_test)

    return avg_df, detail_df


def load_all_csv(folder):
    all_avg = []
    all_detail = []
    for file in glob.glob(os.path.join(folder, "*.csv")):
        print(f"Reading file: {os.path.basename(file)}")
        avg_df, detail_df = parse_pool2_csv(file)
        if not avg_df.empty:
            all_avg.append(avg_df)
        if not detail_df.empty:
            all_detail.append(detail_df)

    avg_df_all = pd.concat(all_avg, ignore_index=True) if all_avg else pd.DataFrame()
    detail_df_all = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    return avg_df_all, detail_df_all

# ---------------------------
# Analysen
# ---------------------------
def analyse_overall_from_avg(avg_df):
    """
    Gesamt-Auswertung aus den test_avg-Zeilen (über alle Dateien gemittelt).
    """
    if avg_df.empty:
        return pd.DataFrame()

    cols = [
        "avg_false_positive","avg_false_negative","avg_error_rate",
        "avg_best_time","avg_avg_time","avg_worst_time","avg_std_dev"
    ]
    for c in cols:
        if c in avg_df.columns:
            avg_df[c] = pd.to_numeric(avg_df[c], errors="coerce")

    overall = (avg_df
               .groupby(["test","category"], as_index=False)[cols]
               .mean()
               .sort_values(["category","avg_avg_time","avg_error_rate"], ascending=[True, True, True]))
    return overall

def runtime_stats_from_detail(detail_df):
    """
    Laufzeit-Statistik je Test aus Detaildaten (unabhängig von k; hier gibt es k nicht).
    """
    if detail_df.empty:
        return pd.DataFrame()

    grp = detail_df.groupby(["test","category"])
    out = grp.agg(
        time_min=("avg_time","min"),
        time_max=("avg_time","max"),
        time_avg=("avg_time","mean"),
        time_std=("avg_time","std"),
        best_time_min=("best_time","min"),
        worst_time_max=("worst_time","max"),
    ).reset_index()
    return out

def error_stats_from_detail(detail_df):
    """
    Fehleranalyse je Test aus Detaildaten.
    """
    if detail_df.empty:
        return pd.DataFrame()

    grp = detail_df.groupby(["test","category"])
    out = grp.agg(
        false_pos_sum=("false_positive","sum"),
        false_neg_sum=("false_negative","sum"),
        errors_sum=("is_error","sum"),
        error_rate_avg=("error_rate","mean"),
    ).reset_index()
    return out

def group_summaries(avg_overall):
    """
    Pro gewünschter Kategorie: schnellster & genauester Test.
    Nutzt die Aggregation aus den avg-Werten.
    """
    if avg_overall.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Schnellster (kleinstes avg_avg_time) pro category
    fastest_idx = avg_overall.groupby("category")["avg_avg_time"].idxmin()
    fastest = avg_overall.loc[fastest_idx].reset_index(drop=True)

    # Genauester (kleinstes avg_error_rate) pro category
    accurate_idx = avg_overall.groupby("category")["avg_error_rate"].idxmin()
    most_accurate = avg_overall.loc[accurate_idx].reset_index(drop=True)

    return fastest, most_accurate

# ---------------------------
# Ausgabe & Export
# ---------------------------
def print_section(title, df, cols=None):
    print(f"\n--- {title} ---")
    if df is None or df.empty:
        print("(keine Daten)")
        return
    if cols:
        print(df[cols].to_string(index=False))
    else:
        print(df.to_string(index=False))

def export_df(file_path, df):
    if df is None or df.empty:
        print(f"[warn] {file_path} – keine Daten, übersprungen")
        return
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"[export] wrote: {os.path.abspath(file_path)}  rows={len(df)}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Ordner anpassen:
    folder = "C:\\Users\\julia\\Downloads\\testpool2"

    avg_df_all, detail_df_all = load_all_csv(folder)

    # 1) Gesamtauswertung (aus avg)
    overall_avg = analyse_overall_from_avg(avg_df_all)
    print_section(
        "Gesamtauswertung (aus test_avg)",
        overall_avg,
        cols=[
            "category","test",
            "avg_false_positive","avg_false_negative","avg_error_rate",
            "avg_best_time","avg_avg_time","avg_worst_time","avg_std_dev"
        ],
    )

    # 2) Laufzeit-Statistik (Detaildaten)
    rt_stats = runtime_stats_from_detail(detail_df_all)
    print_section(
        "Laufzeitanalyse - Statistik (Detaildaten)",
        rt_stats,
        cols=[
            "category","test",
            "best_time_min","time_avg","worst_time_max","time_std"
        ],
    )

    # 3) Fehleranalyse (Detaildaten)
    err_stats = error_stats_from_detail(detail_df_all)
    print_section(
        "Fehleranalyse (Detaildaten)",
        err_stats,
        cols=[
            "category","test","false_pos_sum","false_neg_sum","error_rate_avg","errors_sum"
        ],
    )

    # 4) Gruppenauswertung (aus avg)
    fastest, most_accurate = group_summaries(overall_avg)
    print_section(
        "Gruppenauswertung – Schnellster Test je Kategorie (aus avg)",
        fastest,
        cols=["category","test","avg_avg_time","avg_error_rate"]
    )
    print_section(
        "Gruppenauswertung – Genauester Test je Kategorie (aus avg)",
        most_accurate,
        cols=["category","test","avg_error_rate","avg_avg_time"]
    )

    # Export nur 1 CSV
    export_df("p2-result/1pool2_overall_avg.csv", overall_avg)
    export_df("p2-result/2pool2_runtime_stats.csv", rt_stats)
    export_df("p2-result/3pool2_error_stats.csv", err_stats)
    export_df("p2-result/4pool2_fastest_by_group.csv", fastest)
    export_df("p2-result/5pool2_most_accurate_by_group.csv", most_accurate)
    export_df("p2-result/pool2_detail_raw.csv", detail_df_all)
    export_df("p2-result/pool2_avg_raw.csv", avg_df_all)
import os
import glob
import numpy as np
import pandas as pd
import re

def parse_avg_line(line: str):
    parts = line.strip().split(',')
    test_name = parts[1]
    time_match = re.search(r'avg_time=([\d\.]+) ms', line)
    error_match = re.search(r'avg_error_rate=([\d\.]+)', line)
    avg_time = float(time_match.group(1)) if time_match else None
    avg_error = float(error_match.group(1)) if error_match else None
    return test_name, avg_time, avg_error

def read_csv_custom(filepath):
    avg_data = []
    detail_lines = []
    start_details = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('test_avg'):
                avg_data.append(parse_avg_line(line))
            elif line.startswith('Gruppe,Test,Zahl'):
                start_details = True
                detail_lines.append(line)
            elif start_details:
                detail_lines.append(line)

    avg_df = pd.DataFrame(avg_data, columns=['Test', 'Avg_Time_ms', 'Avg_Error_Rate'])

    if detail_lines:
        from io import StringIO
        detail_csv = ''.join(detail_lines)
        detail_df = pd.read_csv(StringIO(detail_csv))
        time_cols = ['best_time', 'avg_time', 'worst_time', 'std_dev']
        for col in time_cols:
            if col in detail_df.columns:
                detail_df[col] = detail_df[col].astype(str).str.replace(' ms', '').replace('nan', '0').astype(float)
        return avg_df, detail_df
    else:
        return avg_df, pd.DataFrame()

def load_all_csvs(folder_path):
    all_avg = []
    all_details = []
    for file in glob.glob(os.path.join(folder_path, '*.csv')):
        avg_df, detail_df = read_csv_custom(file)
        avg_df['source_file'] = os.path.basename(file)
        all_avg.append(avg_df)
        if not detail_df.empty:
            detail_df['source_file'] = os.path.basename(file)
            all_details.append(detail_df)
    all_avg_df = pd.concat(all_avg, ignore_index=True)
    all_details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    return all_avg_df, all_details_df

def compute_avg_metrics(all_avg_df):
    summary = all_avg_df.groupby('Test').agg(
        Mean_Avg_Time_ms=('Avg_Time_ms', 'mean'),
        Std_Avg_Time_ms=('Avg_Time_ms', 'std'),
        Mean_Error_Rate=('Avg_Error_Rate', 'mean'),
        Std_Error_Rate=('Avg_Error_Rate', 'std'),
        Count=('Test', 'count')
    ).reset_index()
    return summary

def compute_error_types(detail_df):
    if detail_df.empty:
        return pd.DataFrame()
    error_summary = detail_df.groupby('Test').agg(
        Total_Tests=('is_error', 'count'),
        Errors=('is_error', 'sum'),
        False_Positives=('false_positive', 'sum'),
        False_Negatives=('false_negative', 'sum')
    ).reset_index()
    error_summary['FP_Rate'] = error_summary['False_Positives'] / error_summary['Total_Tests']
    error_summary['FN_Rate'] = error_summary['False_Negatives'] / error_summary['Total_Tests']
    error_summary['Error_Rate'] = error_summary['Errors'] / error_summary['Total_Tests']
    return error_summary

def identify_error_cases(detail_df):
    if detail_df.empty:
        return pd.DataFrame()
    errors = detail_df[detail_df['is_error'] == True][['Test', 'Zahl', 'reason', 'false_positive', 'false_negative']]
    return errors

def group_summary(detail_df):
    if detail_df.empty:
        return pd.DataFrame()
    group_sum = detail_df.groupby('Gruppe').agg(
        Mean_Avg_Time_ms=('avg_time', 'mean'),
        Mean_Error_Rate=('error_rate', 'mean')
    ).reset_index()
    return group_sum

###################

def complexity_models(n_vals, log10_n, log2_n):
    models = {}

    # Logarithmus-Modelle mit explizitem n in der Darstellung:
    for k in range(1, 21):
        models[f"log_10^{k}(n)"] = log10_n ** k
        models[f"log_2^{k}(n)"] = log2_n ** k

    # Polynomial- und gemischte Modelle mit n:
    models["n"] = n_vals
    models["n * log_10(n)"] = n_vals * log10_n
    models["n * log_2(n)"] = n_vals * log2_n
    models["n^2"] = n_vals ** 2
    models["n^3"] = n_vals ** 3
    models["sqrt(n)"] = np.sqrt(n_vals)
    models["n^0.1"] = n_vals ** 0.1
    models["n^0.5"] = n_vals ** 0.5
    models["n^1.5"] = n_vals ** 1.5
    models["log_10(n) * log_10(log_10(n))"] = log10_n * np.log10(np.where(log10_n > 0, log10_n, 1))
    models["log_2(n) * log_2(log_2(n))"] = log2_n * np.log2(np.where(log2_n > 0, log2_n, 1))

    return models


def fit_model(x, y):
    # Lineare Regression y ~ a*x + b mit Kleinste-Quadrate
    A = np.vstack([x, np.ones(len(x))]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coeffs
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return slope, intercept, r2


def analyze_test_complexity(detail_df):
    print("📊 Laufzeitanalyse - Beste Komplexitätsfits je Test")
    detail_df = detail_df.copy()
    detail_df = detail_df.dropna(subset=['avg_time', 'Zahl'])
    detail_df = detail_df[detail_df['avg_time'] > 0]

    def safe_int(x):
        try:
            return int(str(x).strip())
        except:
            return None
    detail_df['number_int'] = detail_df['Zahl'].apply(safe_int)
    detail_df = detail_df.dropna(subset=['number_int'])

    results = {}

    for test_name, group in detail_df.groupby('Test'):
        n_vals_g = group['number_int'].values.astype(float)
        y_g = group['avg_time'].values.astype(float)

        # Logarithmen berechnen, vermeiden von -inf durch minimalen Wert 1
        log10_n_g = np.log10(np.where(n_vals_g > 0, n_vals_g, 1))
        log2_n_g = np.log2(np.where(n_vals_g > 0, n_vals_g, 1))

        models = complexity_models(n_vals_g, log10_n_g, log2_n_g)

        best_r2_dec = -np.inf
        best_model_dec = None
        best_params_dec = None

        best_r2_bin = -np.inf
        best_model_bin = None
        best_params_bin = None

        # Dezimallog-Modelle fitten
        for name, x_vals in models.items():
            if name.startswith("log_10") or name in ["n", "n * log_10(n)", "n^2", "n^3", "sqrt(n)", "n^0.1", "n^0.5", "n^1.5", "log_10(n) * log_10(log_10(n))"]:
                slope, intercept, r2 = fit_model(x_vals, y_g)
                # Filter für vernünftige Modelle: z.B. positive Steigung
                if r2 > best_r2_dec and slope > 0:
                    best_r2_dec = r2
                    best_model_dec = name
                    best_params_dec = (slope, intercept)

        # Binärlog-Modelle fitten
        for name, x_vals in models.items():
            if name.startswith("log_2") or name in ["n", "n * log_2(n)", "n^2", "n^3", "sqrt(n)", "n^0.1", "n^0.5", "n^1.5", "log_2(n) * log_2(log_2(n))"]:
                slope, intercept, r2 = fit_model(x_vals, y_g)
                if r2 > best_r2_bin and slope > 0:
                    best_r2_bin = r2
                    best_model_bin = name
                    best_params_bin = (slope, intercept)

        results[test_name] = {
            "decimal": {
                "best_model": best_model_dec,
                "r2": best_r2_dec,
                "params": best_params_dec
            },
            "binary": {
                "best_model": best_model_bin,
                "r2": best_r2_bin,
                "params": best_params_bin
            }
        }
    return results


def print_complexity_results(results):
    print("📊 Laufzeitanalyse - Beste Komplexitätsfits je Test\n")
    for test, res in results.items():
        dec = res['decimal']
        bin_ = res['binary']
        print(f"🔍 {test}:")
        print(f"  ▶️ Dezimal: {dec['best_model']} mit R² = {dec['r2']:.4f}")
        print(f"  ▶️ Binär:   {bin_['best_model']} mit R² = {bin_['r2']:.4f}")
        print()


def export_complexity_results(results, filename):
    rows = []
    for test, res in results.items():
        rows.append({
            "Test": test,
            "Best_Model_Decimal": res['decimal']['best_model'],
            "R2_Decimal": res['decimal']['r2'],
            "Slope_Decimal": res['decimal']['params'][0],
            "Intercept_Decimal": res['decimal']['params'][1],
            "Best_Model_Binary": res['binary']['best_model'],
            "R2_Binary": res['binary']['r2'],
            "Slope_Binary": res['binary']['params'][0],
            "Intercept_Binary": res['binary']['params'][1],
        })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

########################



def export_summary_csv(avg_summary, error_summary, error_cases, group_sum, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        # Durchschnittliche Werte pro Test
        f.write('--- Durchschnittliche Werte pro Test über alle Dateien ---\n')
        avg_summary.to_csv(f, index=False)
        f.write('\n\n')

        # Fehlerarten pro Test
        f.write('--- Fehlerarten pro Test ---\n')
        error_summary.to_csv(f, index=False)
        f.write('\n\n')

        # Kritische Fehlerfälle
        f.write('--- Kritische Fehlerfälle ---\n')
        error_cases.to_csv(f, index=False)
        f.write('\n\n')

        # Gruppenzusammenfassung
        f.write('--- Gruppenzusammenfassung ---\n')
        group_sum.to_csv(f, index=False)
        f.write('\n')

# Beispiel wie man es in main aufruft:
if __name__ == '__main__':
    folder_path = "C:\\Users\\julia\\OneDrive\\Dokumente\\Studium\\Semester M4\\MA\\Datensammlung\\Pool 1 - alle, v1, 1-k1"

    avg_df, detail_df = load_all_csvs(folder_path)

    print("\n--- Durchschnittliche Werte pro Test über alle Dateien ---")
    avg_summary = compute_avg_metrics(avg_df)
    print(avg_summary)

    print("\n--- Fehlerarten pro Test ---")
    error_summary = compute_error_types(detail_df)
    print(error_summary)

    print("\n--- Kritische Fehlerfälle ---")
    error_cases = identify_error_cases(detail_df)
    print(error_cases)

    print("\n--- Gruppenzusammenfassung ---")
    group_sum = group_summary(detail_df)
    print(group_sum)

    export_summary_csv(avg_summary, error_summary, error_cases, group_sum, '.\data\pool1-time-error.csv')

    # Neue Laufzeitanalyse
    print("\n--- Laufzeitanalyse: Kleinste-Quadrate-Fit für Komplexitätsmodelle ---")
    complexity_results = analyze_test_complexity(detail_df)
    print_complexity_results(complexity_results)
    export_complexity_results(complexity_results, '.\data\pool1-complexity.csv')



    
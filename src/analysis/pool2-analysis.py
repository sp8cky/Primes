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
    folder_path = "C:\\Users\\julia\\OneDrive\\Dokumente\\Studium\\Semester M4\\MA\\Datensammlung\\Pool 2 - alle, v2, 1-k10"

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

    export_summary_csv(avg_summary, error_summary, error_cases, group_sum, '.\data\pool2-time-error.csv')




    
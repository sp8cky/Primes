import os
import csv
import numpy as np
from collections import defaultdict
from scipy.stats import linregress
import math

class ProbabilisticTestAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.all_data = []
        self.test_stats = defaultdict(list)
        self.k_values = set()
        
    def load_all_csv_files(self):
        """Lädt alle CSV-Dateien aus dem angegebenen Ordner"""
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.folder_path, filename)
                self.load_csv_file(filepath)
    
    def load_csv_file(self, filepath):
        """Lädt eine einzelne CSV-Datei und extrahiert relevante Daten"""
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            
        # Finde die Zeile mit den Spaltenüberschriften
        header_index = next(i for i, line in enumerate(lines) if line and line[0] == 'Gruppe')
        
        # Extrahiere die Datenzeilen
        data_lines = lines[header_index+1:]
        
        # Extrahiere Test-Avg-Informationen
        test_avg_lines = [line for line in lines if line and line[0] == 'test_avg']
        for line in test_avg_lines:
            test_name = line[1]
            k_value = int(test_name.split('k = ')[1].split(')')[0])
            self.k_values.add(k_value)
            
            stats = {
                'k': k_value,
                'false_positive': float(line[2].split('=')[1]),
                'false_negative': float(line[3].split('=')[1]),
                'error_rate': float(line[4].split('=')[1]),
                'best_time': float(line[5].split('=')[1].split(' ')[0]),
                'avg_time': float(line[6].split('=')[1].split(' ')[0]),
                'worst_time': float(line[7].split('=')[1].split(' ')[0]),
                'std_dev': float(line[8].split('=')[1].split(' ')[0])
            }
            self.test_stats[test_name.split(' (')[0]].append(stats)
        
        # Extrahiere die einzelnen Testdaten
        for line in data_lines:
            if len(line) < 12:  # Sicherstellen, dass die Zeile genug Daten enthält
                continue
                
            test_name = line[1]
            base_test_name = test_name.split(' (')[0]
            k_value = int(test_name.split('k = ')[1].split(')')[0])
            
            entry = {
                'test': base_test_name,
                'k': k_value,
                'number': int(line[2]),
                'result': line[3],
                'true_prime': line[4] == 'True',
                'is_error': line[5] == 'True',
                'false_positive': line[6] == 'True',
                'false_negative': line[7] == 'True',
                'error_rate': float(line[8]),
                'best_time': float(line[9].split(' ')[0]),
                'avg_time': float(line[10].split(' ')[0]),
                'worst_time': float(line[11].split(' ')[0]),
                'std_dev': float(line[12].split(' ')[0]) if len(line) > 12 else 0
            }
            self.all_data.append(entry)
    
    def calculate_overall_stats(self):
        """Berechnet die Gesamtstatistiken für alle Tests über alle Durchläufe"""
        tests = set(entry['test'] for entry in self.all_data)
        overall_stats = {}
        
        for test in tests:
            test_entries = [entry for entry in self.all_data if entry['test'] == test]
            test_avg_entries = [stat for stat_list in self.test_stats[test] for stat in stat_list]
            
            # Fehlerstatistiken
            total_fp = sum(entry['false_positive'] for entry in test_entries)
            total_fn = sum(entry['false_negative'] for entry in test_entries)
            total_tests = len(test_entries)
            
            # Laufzeitstatistiken
            avg_times = [entry['avg_time'] for entry in test_entries]
            best_times = [entry['best_time'] for entry in test_entries]
            worst_times = [entry['worst_time'] for entry in test_entries]
            
            overall_stats[test] = {
                'false_positive_rate': total_fp / total_tests,
                'false_negative_rate': total_fn / total_tests,
                'error_rate': (total_fp + total_fn) / total_tests,
                'avg_best_time': np.mean(best_times),
                'avg_avg_time': np.mean(avg_times),
                'avg_worst_time': np.mean(worst_times),
                'time_std_dev': np.std(avg_times),
                'min_time': np.min(best_times),
                'max_time': np.max(worst_times)
            }
        
        return overall_stats
    
    def analyze_runtime_vs_k(self):
        """Analysiert den Zusammenhang zwischen k und der Laufzeit"""
        results = {}
        
        for test, stats_list in self.test_stats.items():
            k_values = []
            avg_times = []
            
            for stat in stats_list:
                k_values.append(stat['k'])
                avg_times.append(stat['avg_time'])
            
            # Lineare Regression
            slope, intercept, r_value, p_value, std_err = linregress(k_values, avg_times)
            
            # Logarithmische Regression
            log_k = [math.log(k) for k in k_values]
            slope_log, intercept_log, r_value_log, p_value_log, std_err_log = linregress(log_k, avg_times)
            
            results[test] = {
                'linear': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'fit_type': 'linear'
                },
                'logarithmic': {
                    'slope': slope_log,
                    'intercept': intercept_log,
                    'r_squared': r_value_log**2,
                    'fit_type': 'logarithmic'
                },
                'best_fit': 'logarithmic' if r_value_log**2 > r_value**2 else 'linear'
            }
        
        return results
    
    def analyze_runtime_complexity(self, theoretical_complexity):
        """
        Vergleicht die praktischen Laufzeiten mit der theoretischen Komplexität
        theoretical_complexity: Dict mit {testname: 'O(...)'}
        """
        results = {}
        
        for test in self.test_stats.keys():
            # Extrahiere alle Zahlen und Laufzeiten für diesen Test
            test_entries = [entry for entry in self.all_data if entry['test'] == test]
            numbers = [entry['number'] for entry in test_entries]
            times = [entry['avg_time'] for entry in test_entries]
            
            # Filtere ungültige Werte (<= 0) heraus
            valid_pairs = [(n, t) for n, t in zip(numbers, times) if t > 0]
            if not valid_pairs:
                results[test] = {
                    'theoretical': theoretical_complexity.get(test, 'N/A'),
                    'practical': 'N/A (no valid times)',
                    'r_squared': 0,
                    'exponent': 0
                }
                continue
                
            valid_numbers, valid_times = zip(*valid_pairs)
            
            # Lineare Regression von log(n) vs log(time)
            try:
                log_numbers = [math.log(n) for n in valid_numbers]
                log_times = [math.log(t) for t in valid_times]
                
                slope, intercept, r_value, p_value, std_err = linregress(log_numbers, log_times)
                
                # Bestimme den praktischen Fit
                if abs(slope - 1) < 0.2:
                    practical_fit = 'O(n)'
                elif abs(slope - 2) < 0.3:
                    practical_fit = 'O(n²)'
                elif abs(slope - math.log(2)) < 0.2:
                    practical_fit = 'O(log n)'
                else:
                    practical_fit = f'O(n^{slope:.2f})'
                
                results[test] = {
                    'theoretical': theoretical_complexity.get(test, 'N/A'),
                    'practical': practical_fit,
                    'r_squared': r_value**2,
                    'exponent': slope
                }
            except Exception as e:
                results[test] = {
                    'theoretical': theoretical_complexity.get(test, 'N/A'),
                    'practical': f'N/A (error: {str(e)})',
                    'r_squared': 0,
                    'exponent': 0
                }
        
        return results
    
    def analyze_errors(self):
        """Analysiert Fehlerraten und deren Verteilung"""
        error_analysis = {}
        
        for test in self.test_stats.keys():
            test_entries = [entry for entry in self.all_data if entry['test'] == test]
            total = len(test_entries)
            
            if total == 0:
                continue
                
            false_positives = sum(1 for entry in test_entries if entry['false_positive'])
            false_negatives = sum(1 for entry in test_entries if entry['false_negative'])
            errors = sum(1 for entry in test_entries if entry['is_error'])
            
            # Fehler nach Zahlengröße
            error_numbers = [entry['number'] for entry in test_entries if entry['is_error']]
            avg_error_number = np.mean(error_numbers) if error_numbers else 0
            std_error_number = np.std(error_numbers) if error_numbers else 0
            
            error_analysis[test] = {
                'false_positive_rate': false_positives / total,
                'false_negative_rate': false_negatives / total,
                'total_error_rate': errors / total,
                'avg_error_number': avg_error_number,
                'std_error_number': std_error_number,
                'error_count': errors,
                'total_tests': total
            }
        
        return error_analysis
    
    def print_results(self, overall_stats, runtime_k_results, complexity_results, error_results):
        """Gibt die Ergebnisse in tabellarischer Form aus"""
        
        # Funktion zur Erstellung von Tabellen
        def print_table(title, headers, rows):
            print(f"\n=== {title} ===")
            col_widths = [max(len(str(h)), *[len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]
            header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            separator = "-+-".join('-' * w for w in col_widths)
            print(header_row)
            print(separator)
            for row in rows:
                print(" | ".join(str(x).ljust(w) for x, w in zip(row, col_widths)))
        
        # Gesamtstatistiken
        headers = ["Test", "FP-Rate", "FN-Rate", "Error-Rate", "Avg-Time (ms)", "Best-Time", "Worst-Time", "Std-Dev"]
        rows = []
        for test, stats in overall_stats.items():
            rows.append([
                test,
                f"{stats['false_positive_rate']:.4f}",
                f"{stats['false_negative_rate']:.4f}",
                f"{stats['error_rate']:.4f}",
                f"{stats['avg_avg_time']:.4f}",
                f"{stats['avg_best_time']:.4f}",
                f"{stats['avg_worst_time']:.4f}",
                f"{stats['time_std_dev']:.4f}"
            ])
        print_table("GESAMTSTATISTIKEN", headers, rows)
        
        # Laufzeit vs. k
        headers = ["Test", "Bester Fit", "R²", "Steigung", "Achsenabschnitt"]
        rows = []
        for test, results in runtime_k_results.items():
            best_fit = results['best_fit']
            fit_data = results[best_fit]
            rows.append([
                test,
                best_fit,
                f"{fit_data['r_squared']:.4f}",
                f"{fit_data['slope']:.4f}",
                f"{fit_data['intercept']:.4f}"
            ])
        print_table("LAUFZEIT vs. K", headers, rows)
        
        # Komplexitätsanalyse
        headers = ["Test", "Theoretisch", "Praktisch", "R²", "Exponent"]
        rows = []
        for test, results in complexity_results.items():
            rows.append([
                test,
                results['theoretical'],
                results['practical'],
                f"{results['r_squared']:.4f}",
                f"{results['exponent']:.4f}"
            ])
        print_table("LAUFZEITKOMPLEXITÄT", headers, rows)
        
        # Fehleranalyse
        headers = ["Test", "FP-Rate", "FN-Rate", "Error-Rate", "Avg-Error-Num", "Error-Count"]
        rows = []
        for test, results in error_results.items():
            rows.append([
                test,
                f"{results['false_positive_rate']:.4f}",
                f"{results['false_negative_rate']:.4f}",
                f"{results['total_error_rate']:.4f}",
                f"{results['avg_error_number']:.1f}",
                f"{results['error_count']}/{results['total_tests']}"
            ])
        print_table("FEHLERANALYSE", headers, rows)

def main():
    # Konfiguration
    csv_folder = "C:\\Users\\julia\\Downloads\\testpool3"  # Hier den Pfad zu Ihren CSV-Dateien angeben
    
    # Theoretische Komplexitäten (anpassen nach Bedarf)
    theoretical_complexity = {
        "Fermat": "O(k log^3 n)",
        "Miller-Selfridge-Rabin": "O(k log^4 n)",
        "Solovay-Strassen": "O(k log^3 n)"
    }
    
    # Analyzer erstellen und ausführen
    analyzer = ProbabilisticTestAnalyzer(csv_folder)
    analyzer.load_all_csv_files()
    
    # Analysen durchführen
    overall_stats = analyzer.calculate_overall_stats()
    runtime_k_results = analyzer.analyze_runtime_vs_k()
    complexity_results = analyzer.analyze_runtime_complexity(theoretical_complexity)
    error_results = analyzer.analyze_errors()
    
    # Ergebnisse ausgeben
    analyzer.print_results(overall_stats, runtime_k_results, complexity_results, error_results)

if __name__ == "__main__":
    main()
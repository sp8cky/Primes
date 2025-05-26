import timeit

def time_function(func, *args, rounds=5):
    """ Führt eine Zeitmessung für die übergebene Funktion mit den gegebenen Argumenten durch.
    Gibt die durchschnittliche Laufzeit in Sekunden zurück."""
    total_time = timeit.timeit(lambda: func(*args), number=rounds)
    return total_time / rounds

def measure_times(func, inputs, rounds=5):
    """ Führt die Zeitmessung über eine Liste von Eingabewerten durch.
    Gibt Liste von (n, zeit) Paaren zurück. """
    results = []
    for n in inputs:
        try:
            elapsed = time_function(func, n, rounds=rounds)
            results.append((n, elapsed))
        except Exception as e:
            results.append((n, None))  # Fehlerhafte Eingaben ignorieren
    return results
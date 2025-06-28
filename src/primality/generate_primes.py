import random
from typing import List, Dict
from sympy import isprime, primerange, primefactors
from math import log2
import src.primality.helpers as helpers
from src.primality.test_config import *
from src.analysis.dataset import extract_base_label
from typing import List

def generate_all_test_numbers(n, start, end, test_config) -> Dict[str, List[int]]:
    numbers_per_test = {}
    for test_name, cfg in test_config.items():
        number_type = cfg.get("number_type", "general")
        print(f"Generiere Zahlen für Test '{test_name}' ({number_type})...")
        try: # Versuche direkt n Zahlen zu generieren
            numbers = generate_numbers_for_test(n, start, end, number_type)
        except ValueError as e: # Wenn zu wenig möglich sind, versuche maximal verfügbare
            print(f"⚠️ Für Test '{test_name}' konnten nicht {n}, sondern nur weniger Zahlen generiert werden: {e}")
            try: # versuche maximal viele zu generieren
                max_available = 1000  # großzügiger Puffer
                candidates = generate_numbers_for_test(max_available, start, end, number_type)
                numbers = candidates[:n]  # falls doch mehr verfügbar
                print(f"→ Nur {len(numbers)} Zahlen für Test '{test_name}' verwendet.")
            except ValueError as e2:
                print(f"⛔ Test '{test_name}' wird komplett übersprungen: {e2}")
                continue

        numbers_per_test[test_name] = numbers
    return numbers_per_test

def generate_numbers_per_group(n, start, end, test_config):
    # 1. Gruppiere Tests nach Gruppe
    group_to_tests = {}
    for test_name in test_config:
        base_label = extract_base_label(test_config[test_name]["label"])
        group = TEST_GROUPS.get(base_label)
        if group is None:
            print(f"⚠️ Test {test_name} mit Label {base_label} hat keine Gruppe.")
            continue
        group_to_tests.setdefault(group, []).append(test_name)

    # 2. Für jede Gruppe Zahlen generieren (einmal pro Gruppe)
    group_numbers = {}
    for group, tests in group_to_tests.items():
        # Wir nehmen den Typ des ersten Tests für die Generierung
        first_test = tests[0]
        number_type = test_config[first_test].get("number_type", "general")
        print(f"Generiere {n} Zahlen für Gruppe '{group}' vom Typ '{number_type}'...")
        try:
            numbers = generate_numbers_for_test(n, start, end, number_type)
        except ValueError as e:
            print(f"⚠️ Für Gruppe '{group}' konnten nicht {n} Zahlen generiert werden: {e}")
            # Versuche max verfügbare
            try:
                max_available = 1000
                candidates = generate_numbers_for_test(max_available, start, end, number_type)
                numbers = candidates[:n]
                print(f"→ Nur {len(numbers)} Zahlen für Gruppe '{group}' verwendet.")
            except ValueError as e2:
                print(f"⛔ Gruppe '{group}' wird komplett übersprungen: {e2}")
                numbers = []

        group_numbers[group] = numbers

    # 3. Erstelle dict Testname → Liste von Zahlen der jeweiligen Gruppe
    numbers_per_test = {}
    for group, tests in group_to_tests.items():
        nums = group_numbers.get(group, [])
        for test in tests:
            numbers_per_test[test] = nums

    return numbers_per_test



# 1. Generische grobe Filterung
def generate_numbers(n: int, start: int, end: int, number_type: str = "general", max_attempts=10000) -> List[int]:
    numbers = []
    attempts = 0
    while len(numbers) < n and (attempts < max_attempts):
        attempts += 1
        candidate = random.randint(start, end)
        if candidate < 2 or (candidate % 2 == 0 and candidate > 2) or helpers.is_real_potency(candidate):
            continue
        numbers.append(candidate)
    if len(numbers) < n:
        raise ValueError(f"Nur {len(numbers)} Zahlen generiert, weniger als benötigt ({n})")
    return numbers

# 2. Spezielle Generatoren:
def generate_fermat_numbers(n: int, start: int, end: int) -> List[int]:
    fermat_candidates = []
    k = 0
    while True:
        f = 2 ** (2 ** k) + 1
        if f > end:
            break
        if f >= start:
            fermat_candidates.append(f)
        k += 1
    if len(fermat_candidates) < n:
        raise ValueError(f"Nicht genug Fermat-Zahlen im Bereich [{start}, {end}] (nur {len(fermat_candidates)})")
    return random.sample(fermat_candidates, n)

def generate_mersenne_numbers(n: int, start: int, end: int) -> List[int]:
    mersenne_candidates = []
    p = 2
    while True:
        m = 2 ** p - 1
        if m > end:
            break
        if m >= start:
            mersenne_candidates.append(m)
        p += 1
    if len(mersenne_candidates) < n:
        raise ValueError(f"Nicht genug Mersenne-Zahlen im Bereich [{start}, {end}] (nur {len(mersenne_candidates)})")
    return random.sample(mersenne_candidates, n)

# generate proth numbers N = k * 2**m + 1, where k is odd and 1 < k < 2**m
def generate_proth_numbers(n: int, start: int, end: int) -> List[int]:
    numbers = []
    attempts = 0
    max_attempts = n * 50
    while len(numbers) < n and attempts < max_attempts:
        attempts += 1
        m = random.randint(3, int(log2(end)))
        k = random.randint(1, 2**m - 1)
        N = k * 2**m + 1
        if start <= N <= end:
            numbers.append(N)
    if len(numbers) < n:
        raise ValueError(f"Nicht genug Proth-Zahlen im Bereich [{start}, {end}] (nur {len(numbers)})")
    return sorted(numbers[:n])

# generate pocklington numbers, which are of the form N = p * q + 1, where p is prime and q is a prime factor of N-1
def generate_pocklington_numbers(n: int, start: int, end: int) -> List[int]:
    results = []
    for candidate in range(start, end + 1):
        decomposition = helpers.find_pocklington_decomposition(candidate)
        if decomposition is not None:
            results.append(candidate)
            if len(results) >= n:
                break
    if len(results) < n:
        raise ValueError(f"Nicht genug Pocklington-Zahlen im Bereich [{start}, {end}] (nur {len(results)})")
    return results

# generate rao numbers, which are of the form R = p * 2^n + 1, where p is prime and R - 1 is divisible by a power of 2
def generate_rao_numbers(n: int, start: int, end: int) -> List[int]:
    results = []
    for candidate in range(start, end + 1):
        decomposition = helpers.find_rao_decomposition(candidate)
        if decomposition is not None:
            results.append(candidate)
            if len(results) >= n:
                break
    if len(results) < n:
        raise ValueError(f"Nicht genug Rao-Zahlen im Bereich [{start}, {end}] (nur {len(results)})")
    return results


# generate ramzy numbers, which are of the form N = K * p^n + 1 with a prime p and satisfying the Ramzy condition p^{n-1} >= K * p^j
def generate_ramzy_numbers(n: int, start: int, end: int) -> List[int]:
    results = []
    for candidate in range(start, end + 1):
        decomposition = helpers.find_ramzy_decomposition(candidate)
        if decomposition is not None:
            results.append(candidate)
            if len(results) >= n:
                break
    if len(results) < n:
        raise ValueError(f"Nicht genug Ramzy-Zahlen im Bereich [{start}, {end}] (nur {len(results)})")
    return results

# generate lucas primes with an easy factorization condition
def generate_lucas_primes(n: int, start: int, end: int) -> List[int]:
    primes = list(primerange(start, end))
    lucas_primes = []

    for p in primes:
        max_allowed = int(log2(p)**2)
        factors = primefactors(p - 1)
        if all(f <= max_allowed for f in factors):
            lucas_primes.append(p)
            if len(lucas_primes) >= n:
                break

    if len(lucas_primes) < n:
        raise ValueError(f"Nicht genug Lucas-Primzahlen im Bereich [{start}, {end}] (nur {len(lucas_primes)})")

    return lucas_primes

# generate large and small primes based on a threshold
def generate_large_primes(n: int, start: int, end: int) -> List[int]:
    primes = list(primerange(start, end))
    # Optional: Filter für "groß" z.B. obere Hälfte
    threshold = start + (end - start) // 2
    large_primes = [p for p in primes if p >= threshold]
    if len(large_primes) < n:
        raise ValueError(f"Nicht genug große Primzahlen im Bereich [{start}, {end}] (nur {len(large_primes)})")
    return random.sample(large_primes, n)

# generate small primes, which are primes below a certain threshold
def generate_small_primes(n: int, start: int, end: int) -> List[int]:
    primes = list(primerange(start, end))
    threshold = start + (end - start) // 2
    small_primes = [p for p in primes if p < threshold]
    if len(small_primes) < n:
        raise ValueError(f"Nicht genug kleine Primzahlen im Bereich [{start}, {end}] (nur {len(small_primes)})")
    return random.sample(small_primes, n)

# 3. Mapping Testtyp → Generator
def generate_numbers_for_test(n: int, start: int, end: int, number_type: str) -> List[int]:
    if number_type == "fermat":
        return generate_fermat_numbers(n, start, end)
    elif number_type == "mersenne":
        return generate_mersenne_numbers(n, start, end)
    elif number_type == "proth":
        return generate_proth_numbers(n, start, end)
    elif number_type == "pocklington":
        return generate_pocklington_numbers(n, start, end)
    elif number_type == "ramzy":
        return generate_ramzy_numbers(n, start, end)
    elif number_type == "rao":
        return generate_rao_numbers(n, start, end)
    elif number_type == "large_prime":
        return generate_large_primes(n, start, end)
    elif number_type == "small_prime":
        return generate_small_primes(n, start, end)
    elif number_type == "lucas":
        return generate_lucas_primes(n, start, end)
    else:
        return generate_numbers(n, start, end, number_type)
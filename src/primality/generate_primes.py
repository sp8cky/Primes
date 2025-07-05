import random
from typing import List, Dict
from collections import defaultdict
from sympy import isprime, primerange, primefactors
from math import log2
import src.primality.helpers as helpers
from src.primality.test_config import *
from src.analysis.dataset import extract_base_label


def generate_numbers_per_group(n, start, end, TEST_CONFIG, group_ranges=None, allow_partial_numbers=True, seed=None):
    r = random.Random(seed)
    group_to_numbers = {}
    numbers_per_test = defaultdict(list)

    # Reihenfolge wie in TEST_GROUPS definiert, nach Auftreten der Keys
    seen_groups = []
    for conf in TEST_CONFIG.values():
        g = conf.get("testgroup")
        if g and g not in seen_groups:
            seen_groups.append(g)

    print("Starte Abschnitt: Zahlengenerierung pro Test...")

    for group in seen_groups:
        # Finde Tests, die zu dieser Gruppe gehören
        relevant_tests = [
            name for name, conf in TEST_CONFIG.items()
            if conf.get("testgroup") == group
        ]
        if not relevant_tests:
            continue

        # Hole Typ aus erster Testdefinition
        example_test = TEST_CONFIG[relevant_tests[0]]
        number_type = example_test["number_type"]

        # Gruppenspezifische Parameter (oder Fallback auf global)
        group_n = group_ranges.get(group, {}).get("n", n) if group_ranges else n
        group_start = group_ranges.get(group, {}).get("start", start) if group_ranges else start
        group_end = group_ranges.get(group, {}).get("end", end) if group_ranges else end

        try:
            numbers = generate_numbers_for_test(group_n, group_start, group_end, number_type, r)
            if len(numbers) < group_n:
                raise ValueError(f"Nicht genug {number_type}-Zahlen im Bereich [{group_start}, {group_end}] (nur {len(numbers)})")
        except ValueError as e:
            if allow_partial_numbers and 'numbers' in locals() and len(numbers) > 0:
                print(f"⚠️ Gruppe '{group}' bekommt nur {len(numbers)} von {group_n} Zahlen.")
            else:
                print(f"⛔ Gruppe '{group}' wird komplett übersprungen: {e}")
                continue

        print(f"Generiere {len(numbers)} Zahlen für Gruppe '{group}' vom Typ '{number_type}'...")
        group_to_numbers[group] = numbers

        for testname in relevant_tests:
            numbers_per_test[testname] = numbers

    print("\nAbschnitt 'Zahlengenerierung pro Test' abgeschlossen")
    return numbers_per_test


# 1. Generische grobe Filterung
def generate_numbers(n: int, start: int, end: int, number_type: str = "general", max_attempts=10000, r=None) -> List[int]:
    if r is None:
        r = random.Random()
    numbers = set()
    attempts = 0
    while len(numbers) < n and (attempts < max_attempts):
        attempts += 1
        candidate = r.randint(start, end)
        if candidate < 2 or (candidate % 2 == 0 and candidate > 2) or helpers.is_real_potency(candidate):
            continue
        numbers.add(candidate)
    if len(numbers) < n:
        raise ValueError(f"Nur {len(numbers)} Zahlen generiert, weniger als benötigt ({n})")
    return sorted(numbers)

# 2. Spezielle Generatoren:
def generate_fermat_numbers(n: int, start: int, end: int, r=None) -> List[int]:
    fermat_candidates = set()
    k = 0
    while True:
        f = 2 ** (2 ** k) + 1
        if f > end:
            break
        if f >= start:
            fermat_candidates.add(f)
        k += 1
    if len(fermat_candidates) == 0:
        raise ValueError(f"Keine Fermat-Zahlen im Bereich [{start}, {end}] gefunden")
    return sorted(list(fermat_candidates))[:n]

def generate_mersenne_numbers(n: int, start: int, end: int, r=None) -> List[int]:
    if r is None:
        r = random.Random()
    mersenne_candidates = set()
    p = 2
    while True:
        m = 2 ** p - 1
        if m > end:
            break
        if m >= start:
            mersenne_candidates.add(m)
        p += 1
    if len(mersenne_candidates) < n:
        raise ValueError(f"Nicht genug Mersenne-Zahlen im Bereich [{start}, {end}] (nur {len(mersenne_candidates)})")
    return sorted(r.sample(list(mersenne_candidates), n))

def generate_proth_numbers(n: int, start: int, end: int, r=None) -> List[int]:
    if r is None:
        r = random.Random()
    numbers = set()
    attempts = 0
    max_attempts = n * 50
    while len(numbers) < n and attempts < max_attempts:
        attempts += 1
        m = r.randint(3, int(log2(end)))
        k = r.randint(1, 2**m - 1)
        N = k * 2**m + 1
        if start <= N <= end:
            numbers.add(N)
    if len(numbers) < n:
        raise ValueError(f"Nicht genug Proth-Zahlen im Bereich [{start}, {end}] (nur {len(numbers)})")
    return sorted(numbers)

def generate_pocklington_numbers(n: int, start: int, end: int, max_attempts=None, r=None) -> List[int]:
    if r is None:
        r = random.Random()
    if max_attempts is None:
        max_attempts = 100 * n
    candidates = set()
    results = set()
    attempts = 0
    while len(results) < n and attempts < max_attempts:
        candidate = r.randint(start, end)
        if candidate in candidates:
            attempts += 1
            continue
        candidates.add(candidate)
        decomposition = helpers.find_pocklington_decomposition(candidate)
        if decomposition is not None:
            results.add(candidate)
        attempts += 1
    if len(results) < n:
        raise ValueError(f"Nicht genug Pocklington-Zahlen im Bereich [{start}, {end}] (nur {len(results)})")
    return sorted(results)

def generate_rao_numbers(n: int, start: int, end: int, max_attempts=None, r=None) -> List[int]:
    if r is None:
        r = random.Random()
    if max_attempts is None:
        max_attempts = 100 * n
    candidates = set()
    results = set()
    attempts = 0
    while len(results) < n and attempts < max_attempts:
        candidate = r.randint(start, end)
        if candidate in candidates:
            attempts += 1
            continue
        candidates.add(candidate)
        decomposition = helpers.find_rao_decomposition(candidate)
        if decomposition is not None:
            results.add(candidate)
        attempts += 1
    if len(results) < n:
        raise ValueError(f"Nicht genug Rao-Zahlen im Bereich [{start}, {end}] (nur {len(results)})")
    return sorted(results)

def generate_ramzy_numbers(n: int, start: int, end: int, max_attempts=None, r=None) -> List[int]:
    if r is None:
        r = random.Random()
    if max_attempts is None:
        max_attempts = 100 * n
    candidates = set()
    results = set()
    attempts = 0
    while len(results) < n and attempts < max_attempts:
        candidate = r.randint(start, end)
        if candidate in candidates:
            attempts += 1
            continue
        candidates.add(candidate)
        decomposition = helpers.find_ramzy_decomposition(candidate)
        if decomposition is not None:
            results.add(candidate)
        attempts += 1
    if len(results) < n:
        raise ValueError(f"Nicht genug Ramzy-Zahlen im Bereich [{start}, {end}] (nur {len(results)})")
    return sorted(results)

def generate_lucas_primes(n: int, start: int, end: int, r=None) -> List[int]:
    if r is None:
        r = random.Random()
    primes = list(primerange(start, end))
    lucas_primes = []
    for p in primes:
        max_allowed = int(log2(p)**2)
        factors = primefactors(p - 1)
        if all(f <= max_allowed for f in factors):
            lucas_primes.append(p)
    if len(lucas_primes) < n:
        raise ValueError(f"Nicht genug Lucas-Primzahlen im Bereich [{start}, {end}] (nur {len(lucas_primes)})")
    return sorted(r.sample(list(set(lucas_primes)), n))


# 3. Mapping Testtyp → Generator
def generate_numbers_for_test(n: int, start: int, end: int, number_type: str, r=None) -> List[int]:
    if number_type == "fermat":
        return generate_fermat_numbers(n, start, end, r)
    elif number_type == "mersenne":
        return generate_mersenne_numbers(n, start, end, r)
    elif number_type == "proth":
        return generate_proth_numbers(n, start, end, r)
    elif number_type == "pocklington":
        return generate_pocklington_numbers(n, start, end, r=r)
    elif number_type == "ramzy":
        return generate_ramzy_numbers(n, start, end, r=r)
    elif number_type == "rao":
        return generate_rao_numbers(n, start, end, r=r)
    elif number_type == "lucas":
        return generate_lucas_primes(n, start, end, r=r)
    else:
        return generate_numbers(n, start, end, number_type, r=r)

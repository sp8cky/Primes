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
            desired_prime_ratio = 0.5  # Beispiel: Standardwert 50% Primzahlen
            # Falls du den prime_ratio-Wert aus number_type auslesen willst, kannst du das vorher tun, z.B.:
            if number_type.startswith("g") and ":" in number_type:
                _, ratio_str = number_type.split(":")
                try:
                    desired_prime_ratio = float(ratio_str)
                except ValueError:
                    desired_prime_ratio = 0.5

            numbers = generate_numbers_for_test(group_n, group_start, group_end, number_type, r, prime_ratio=desired_prime_ratio)
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

def generate_numbers(n: int, start: int, end: int, number_type: str = "general", max_attempts=10000, r=None, prime_ratio: float = 0.5) -> List[int]:
    if r is None:
        r = random.Random()
    
    primes_needed = round(n * prime_ratio)
    comps_needed = n - primes_needed

    primes = set()
    composites = set()
    attempts = 0

    while (len(primes) < primes_needed or len(composites) < comps_needed) and attempts < max_attempts:
        attempts += 1
        candidate = r.randint(start, end)
        if candidate < 2 or (candidate % 2 == 0 and candidate > 2) or helpers.is_real_potency(candidate):
            continue
        if isprime(candidate):
            if len(primes) < primes_needed:
                primes.add(candidate)
        else:
            if len(composites) < comps_needed:
                composites.add(candidate)

    total_generated = len(primes) + len(composites)
    if total_generated < n:
        print(f"⚠️ WARNUNG: Nur {total_generated} Zahlen generiert (Prim: {len(primes)}, Zusammengesetzt: {len(composites)}) von {n} geforderten Zahlen.")
        # Hier kein Error, weil allow_partial_numbers erlaubt ggf.

    if len(primes) < primes_needed:
        print(f"⚠️ WARNUNG: Nur {len(primes)} Primzahlen generiert, benötigt waren {primes_needed}")
    if len(composites) < comps_needed:
        print(f"⚠️ WARNUNG: Nur {len(composites)} zusammengesetzte Zahlen generiert, benötigt waren {comps_needed}")

    print(f"Generierte Zahlen: {len(primes)} Primzahlen, {len(composites)} zusammengesetzte Zahlen, Gesamt: {total_generated}")
    print(f"Primzahlen: {sorted(primes)}")
    print(f"Zusammengesetzt: {sorted(composites)}")

    return sorted(list(primes.union(composites)))

# 2. Spezielle Generatoren:
def generate_fermat_numbers(n: int, start: int, end: int, r=None) -> List[int]:
    if r is None:
        r = random.Random()
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

def generate_pocklington_numbers(n: int, start: int, end: int, r=None, max_attempts=None) -> List[int]:
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

def generate_rao_numbers(n: int, start: int, end: int, r=None, max_attempts=None) -> List[int]:
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

def generate_ramzy_numbers(n: int, start: int, end: int, r=None, max_attempts=None) -> List[int]:
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
def generate_numbers_for_test(n: int, start: int, end: int, number_type: str, r=None, prime_ratio: float = 0.5) -> List[int]:
    if r is None:
        r = random.Random()
    
    prime_ratio = 0.5  # default
    if number_type.startswith("g"):
        if ":" in number_type:
            _, ratio_str = number_type.split(":")
            try:
                prime_ratio = float(ratio_str)
            except ValueError:
                raise ValueError(f"Ungültiges Format für number_type: {number_type} (z. B. 'g:0.3')")
    # Lucas und Pepin immer nur Primzahlen, prime_ratio ignorieren
    if number_type in ["lucas", "pepin"]:
        return generate_lucas_primes(n, start, end, r) if number_type == "lucas" else generate_fermat_numbers(n, start, end, r)

    if number_type in ["large", "small"]:
        return generate_numbers(n, start, end, number_type="general", r=r, prime_ratio=1.0)

    n_primes = round(n * prime_ratio)
    n_comps = n - n_primes

    prime_generators = {
        "fermat": generate_fermat_numbers,
        "mersenne": generate_mersenne_numbers,
        "proth": generate_proth_numbers,
        "pocklington": generate_pocklington_numbers,
        "ramzy": generate_ramzy_numbers,
        "rao": generate_rao_numbers,
    }

    if number_type in prime_generators:
        primes = []
        composites = []
        if n_primes > 0:
            # Achtung: r als benanntes Argument übergeben!
            try:
                primes = prime_generators[number_type](n_primes, start, end, r=r)
            except ValueError:
                primes = []
        if n_comps > 0:
            try:
                composites = generate_numbers(n_comps, start, end, number_type="general", r=r, prime_ratio=0.0)
            except ValueError:
                composites = []

        numbers = primes + composites
        if len(numbers) < n:
            print(f"⚠️ WARNUNG: Nur {len(numbers)} Zahlen für {number_type} (prim={len(primes)}, comp={len(composites)}) von {n} gefordert")
        return sorted(numbers)

    return generate_numbers(n, start, end, number_type="general", r=r, prime_ratio=prime_ratio)
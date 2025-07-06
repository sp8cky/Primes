import random
from typing import List, Dict
from collections import defaultdict
from sympy import isprime, primerange, primefactors
from math import log2
import src.primality.helpers as helpers
from src.primality.test_config import *
from src.analysis.dataset import extract_base_label

# calculate the distribution of prime and composite numbers based on the num_type
def compute_number_distribution(n: int, num_type: str) -> tuple[int, int, float]:
    if num_type == "p":
        prime_ratio = 1.0
    elif num_type == "z":
        prime_ratio = 0.0
    elif num_type.startswith("g:"):
        try:
            prime_ratio = float(num_type.split(":")[1])
            if not (0.0 <= prime_ratio <= 1.0):
                raise ValueError
        except (IndexError, ValueError):
            raise ValueError(f"Ungültiges Format für num_type: '{num_type}' — erwartet 'g:x' mit x ∈ [0,1]")
    else:
        raise ValueError(f"Unbekannter num_type: '{num_type}' – erlaubt sind 'p', 'z' oder 'g:x'")

    n_primes = round(n * prime_ratio)
    n_composites = n - n_primes
    return n_primes, n_composites, prime_ratio


def is_valid_composite(candidate: int) -> bool:
    return candidate >= 2 and candidate % 2 == 1 and not helpers.is_real_potency(candidate) and not isprime(candidate)

def generate_numbers_per_group(n, start, end, TEST_CONFIG, group_ranges=None, allow_partial_numbers=True, seed=None, num_type: str = "g:x"):
    r = random.Random(seed)
    group_to_numbers = {}
    numbers_per_test = defaultdict(list)

    seen_groups = []
    for conf in TEST_CONFIG.values():
        g = conf.get("testgroup")
        if g and g not in seen_groups:
            seen_groups.append(g)

    print("Starte Abschnitt: Zahlengenerierung pro Test...")

    for group in seen_groups:
        relevant_tests = [
            name for name, conf in TEST_CONFIG.items()
            if conf.get("testgroup") == group
        ]
        if not relevant_tests:
            continue

        group_n = group_ranges.get(group, {}).get("n", n) if group_ranges else n
        group_start = group_ranges.get(group, {}).get("start", start) if group_ranges else start
        group_end = group_ranges.get(group, {}).get("end", end) if group_ranges else end

        try:
            p_count, z_count, _ = compute_number_distribution(group_n, num_type)
            primes = set()
            composites = set()
            attempts = 0
            max_attempts = 10000

            while (len(primes) < p_count or len(composites) < z_count) and attempts < max_attempts:
                attempts += 1
                candidate = r.randint(group_start, group_end)
                if isprime(candidate):
                    if len(primes) < p_count:
                        primes.add(candidate)
                elif is_valid_composite(candidate):
                    if len(composites) < z_count:
                        composites.add(candidate)

            total = len(primes) + len(composites)
            if total < group_n:
                print(f"⚠️ Gruppe '{group}' bekommt nur {total} von {group_n} Zahlen.")
            else:
                print(f"Generiere {total} Zahlen für Gruppe '{group}'...")

            numbers = sorted(primes.union(composites))
            group_to_numbers[group] = numbers
            for testname in relevant_tests:
                numbers_per_test[testname] = numbers

        except ValueError as e:
            if allow_partial_numbers:
                print(f"⚠️ Gruppe '{group}' wird nur teilweise ausgefüllt: {e}")
            else:
                print(f"⛔ Gruppe '{group}' wird übersprungen: {e}")
                continue

    print("\nAbschnitt 'Zahlengenerierung pro Test' abgeschlossen")
    return numbers_per_test

def generate_numbers(n: int, start: int, end: int, r=None, p_count=None, z_count=None, max_attempts=10000) -> List[int]:
    if r is None:
        r = random.Random()

    # Wenn keine expliziten Werte gegeben, verwende Standardverteilung (50/50)
    if p_count is None or z_count is None:
        p_count = round(n * 0.5)
        z_count = n - p_count

    primes = set()
    composites = set()
    attempts = 0

    while (len(primes) < p_count or len(composites) < z_count) and attempts < max_attempts:
        attempts += 1
        candidate = r.randint(start, end)
        if candidate < 2 or (candidate % 2 == 0 and candidate > 2) or helpers.is_real_potency(candidate):
            continue
        if isprime(candidate):
            if len(primes) < p_count:
                primes.add(candidate)
        else:
            if is_valid_composite(candidate) and len(composites) < z_count:
                composites.add(candidate)

    total_generated = len(primes) + len(composites)
    if total_generated < (p_count + z_count):
        print(f"⚠️ WARNUNG: Nur {total_generated} Zahlen generiert (Prim: {len(primes)}, Zusammengesetzt: {len(composites)}) von {p_count + z_count} geforderten Zahlen.")

    if len(primes) < p_count:
        print(f"⚠️ WARNUNG: Nur {len(primes)} Primzahlen generiert, benötigt waren {p_count}")
    if len(composites) < z_count:
        print(f"⚠️ WARNUNG: Nur {len(composites)} zusammengesetzte Zahlen generiert, benötigt waren {z_count}")

    print(f"Generierte Zahlen: {len(primes)} Primzahlen, {len(composites)} zusammengesetzte Zahlen, Gesamt: {total_generated}")
    print(f"Primzahlen: {sorted(primes)}")
    print(f"Zusammengesetzt: {sorted(composites)}")

    return sorted(primes.union(composites))

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
def generate_numbers_for_test(n: int, start: int, end: int, num_type: str = "g:x", r=None) -> List[int]:
    if r is None:
        r = random.Random()

    # Mapping: spezielle Typen → Generatorfunktion
    prime_generators = {
        "fermat": generate_fermat_numbers,
        "mersenne": generate_mersenne_numbers,
        "proth": generate_proth_numbers,
        "pocklington": generate_pocklington_numbers,
        "ramzy": generate_ramzy_numbers,
        "rao": generate_rao_numbers,
        "lucas": generate_lucas_primes,
        "pepin": generate_fermat_numbers,  # Pepin-Test nutzt Fermat-Zahlen
    }

    # Sonderfall: rein primzahlbasiert
    if num_type in ["lucas", "pepin"]:
        return prime_generators[num_type](n, start, end, r=r)

    # Sonderfall: spezielle Formtypen (Fermat, Mersenne, etc.)
    if num_type in prime_generators:
        try:
            p_count, z_count, _ = compute_number_distribution(n, "g:1.0")  # nur Primzahlen
            primes = prime_generators[num_type](p_count, start, end, r=r)
        except ValueError as e:
            print(f"⚠️ Fehler bei Primzahl-Generierung für {num_type}: {e}")
            primes = []

        composites = []
        if z_count > 0:
            try:
                composites = generate_numbers(z_count, start, end, r=r, p_count=0, z_count=z_count)
            except ValueError as e:
                print(f"⚠️ Fehler bei zusammengesetzten Zahlen für {num_type}: {e}")
                composites = []

        return sorted(primes + composites)

    # Allgemeiner Fall: p, z, g:x
    try:
        p_count, z_count, _ = compute_number_distribution(n, num_type)
    except ValueError as e:
        print(f"❌ Ungültiger num_type '{num_type}': {e}")
        return []

    return generate_numbers(n, start, end, r=r, p_count=p_count, z_count=z_count)
import random, math
from typing import List, Dict
from collections import defaultdict
from sympy import isprime, primerange, perfect_power
from math import log2
import src.primality.helpers as helpers
from src.primality.test_config import *

# Weist eine benutzerdefinierte Liste von Zahlen allen Tests einer bestimmten Gruppe zu.
def assign_custom_numbers_to_group(group_name: str, number_list: List[int], TEST_CONFIG: dict) -> Dict[str, List[int]]:

    assigned = {}
    for testname, conf in TEST_CONFIG.items():
        if conf.get("testgroup") == group_name:
            assigned[testname] = number_list
            print(f"✅ Benutzerdefinierte Zahlen an Test '{testname}' der Gruppe '{group_name}' zugewiesen.")
    if not assigned:
        print(f"⚠️ Keine Tests mit der Gruppe '{group_name}' gefunden.")
    return assigned

# Berechnet die gewünschte Verteilung von Primzahlen und Kompositzahlen aus num_type.
def compute_number_distribution(n: int, num_type: str) -> tuple[int, int, float]:
    # num_type bleibt so wie bisher (p, z, g:x)
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
    print(f"🔢 Berechne Verteilung für n={n}, num_type='{num_type}': {n_primes} Primzahlen, {n - n_primes} Zusammengesetzte Zahlen")
    n_composites = n - n_primes
    print(f"🔢 Verteilung: {n_primes} Primzahlen, {n_composites} Zusammengesetzte Zahlen (Verhältnis: {prime_ratio})")
    return n_primes, n_composites, prime_ratio


def is_valid_composite(candidate: int) -> bool:
    return candidate >= 2 and candidate % 2 == 1 and not perfect_power(candidate) and not isprime(candidate)

# Weist einer Testgruppe eine benutzerdefinierte Menge an Zahlen zu.
def generate_numbers_per_group(
    n, start, end, TEST_CONFIG, group_ranges=None, allow_partial_numbers=True, seed=None, num_type: str = "g:x"
):
    r = random.Random(seed)
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

        # Spezialfall: Wenn alle Tests in der Gruppe denselben speziellen number_type haben → gemeinsame Spezialgenerierung
        group_number_types = {TEST_CONFIG[t].get("number_type", "") for t in relevant_tests}
        unique_number_type = group_number_types.pop() if len(group_number_types) == 1 else ""

        if unique_number_type and unique_number_type in {"fermat", "mersenne", "proth", "pocklington", "rao", "ramzy"}:
            try:
                result = generate_numbers_for_test(
                    group_n, group_start, group_end,
                    num_type=num_type,
                    number_type=unique_number_type,
                    r=r,
                    testname=group
                )
                for testname in relevant_tests:
                    numbers_per_test[testname] = result
                    p_count, z_count, _ = compute_number_distribution(group_n, "g:1.0")
                    print(f"🔍 Test '{testname}' num_type='{num_type}', number_type='{unique_number_type}': {len(result)} Zahlen (p: {p_count}, z: {z_count}): {result}")
            except ValueError as e:
                print(f"⚠️ Fehler bei Gruppengenerierung für '{group}': {e}")
                if not allow_partial_numbers:
                    continue
            continue

        # Standardfall: gemeinsame Zufallszahlengenerierung (gemäß num_type)
        try:
            p_count, z_count, _ = compute_number_distribution(group_n, num_type)
            shared_numbers = generate_numbers(
                group_n, group_start, group_end,
                r=r,
                p_count=p_count,
                z_count=z_count,
                max_attempts=10000,
            )
            for testname in relevant_tests:
                numbers_per_test[testname] = shared_numbers
                print(f"🔍 Test '{testname}' num_type='{num_type}', number_type='': {len(shared_numbers)} Zufallszahlen (p: {p_count}, z: {z_count}): {shared_numbers}")
        except ValueError as e:
            print(f"⚠️ Fehler bei Zahlengenerierung für Gruppe '{group}': {e}")
            if not allow_partial_numbers:
                continue

    print("\nAbschnitt 'Zahlengenerierung pro Test' abgeschlossen")
    return numbers_per_test 

# Erzeugt eine Liste von Zufallszahlen (Prims und Komposite) im angegebenen Bereich.
def generate_numbers(n: int, start: int, end: int, r=None, p_count=None, z_count=None, max_attempts=10000, use_log_intervals: bool = False) -> List[int]:
    if r is None: r = random.Random()

    if start < 1:
        start = 1
    if end < start:
        end = start + 1

    if p_count is None or z_count is None:
        p_count = round(n * 0.5)
        z_count = n - p_count

    if use_log_intervals:
        boundaries = [1, 10, 100, 1000, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 10**12, 10**13, 10**14, 10**15, 10**16, 10**17, 10**18]
        boundaries = [b for b in boundaries if start <= b <= end]
        if len(boundaries) == 0 or boundaries[0] != start:
            boundaries.insert(0, start)
        if boundaries[-1] != end:
            boundaries.append(end)
    else:
        boundaries = [start, end]

    intervals = len(boundaries) - 1
    log_weights = [math.log10(boundaries[i+1]) - math.log10(max(1, boundaries[i])) for i in range(intervals)]
    total_log_weight = sum(log_weights)
    interval_weights = [w / total_log_weight for w in log_weights]

    interval_prime_weights = [interval_weights[i] * p_count for i in range(intervals)]
    interval_composite_weights = [interval_weights[i] * z_count for i in range(intervals)]

    per_interval_p = [max(1, int(round(w))) for w in interval_prime_weights]
    per_interval_z = [max(1, int(round(w))) for w in interval_composite_weights]

    rem_p = p_count - sum(per_interval_p)
    rem_z = z_count - sum(per_interval_z)

    for i in range(intervals):
        if rem_p <= 0 and rem_z <= 0:
            break
        if rem_p > 0:
            per_interval_p[i] += 1
            rem_p -= 1
        if rem_z > 0:
            per_interval_z[i] += 1
            rem_z -= 1

    primes, composites = set(), set()

    for i in range(intervals):
        start_i, end_i = boundaries[i], boundaries[i+1] - 1
        if end_i < start_i:
            continue

        local_primes, local_composites = set(), set()
        attempts = 0

        while (len(local_primes) < per_interval_p[i] or len(local_composites) < per_interval_z[i]) and attempts < max_attempts:
            attempts += 1
            log_start = math.log10(start_i)
            log_end = math.log10(end_i)
            candidate = int(10 ** r.uniform(log_start, log_end))

            if candidate in primes or candidate in composites or candidate < 2 or (candidate % 2 == 0 and candidate > 2) or perfect_power(candidate):
                continue

            if isprime(candidate):
                if len(local_primes) < per_interval_p[i]:
                    local_primes.add(candidate)
            elif is_valid_composite(candidate):
                if len(local_composites) < per_interval_z[i]:
                    local_composites.add(candidate)
        primes.update(local_primes)
        composites.update(local_composites)

    total_generated = len(primes) + len(composites)

    if total_generated < n:
        remaining = n - total_generated
        additional = set()
        attempts = 0
        while len(additional) < remaining and attempts < max_attempts * 2:
            attempts += 1
            log_val = r.uniform(math.log10(max(2, start)), math.log10(end))
            candidate = int(10 ** log_val)

            if candidate in primes or candidate in composites or candidate in additional:
                continue
            if candidate < 2 or (candidate % 2 == 0 and candidate > 2) or perfect_power(candidate):
                continue

            if isprime(candidate) and (len(primes) + sum(1 for x in additional if isprime(x)) < p_count):
                additional.add(candidate)
            elif is_valid_composite(candidate) and (len(composites) + sum(1 for x in additional if not isprime(x)) < z_count):
                additional.add(candidate)

        primes.update(x for x in additional if isprime(x))
        composites.update(x for x in additional if not isprime(x))

    return sorted(primes.union(composites))[:n]


# Spezielle Generatoren:
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
    mersenne_candidates = []
    max_p = (end + 1).bit_length()
    for p in primerange(2, max_p + 1):
        m = 2 ** p - 1
        if m > end:
            break
        if m >= start:
            mersenne_candidates.append(m)
    if len(mersenne_candidates) == 0:
        raise ValueError(f"Keine Mersenne-Zahlen im Bereich [{start}, {end}] gefunden")
    if len(mersenne_candidates) < n:
        print(f"⚠️ Warnung: Nur {len(mersenne_candidates)} Mersenne-Zahlen im Bereich gefunden, benötigt: {n}")
        return sorted(mersenne_candidates)
    return sorted(r.sample(mersenne_candidates, n))

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

# Mapping Testtyp → Generator
def generate_numbers_for_test(
    n: int, start: int, end: int, num_type: str = "g:x", number_type: str = "", r=None, testname: str = ""
) -> List[int]:
    if r is None:
        r = random.Random()

    prime_generators = {
        "fermat": generate_fermat_numbers,
        "mersenne": generate_mersenne_numbers,
        "proth": generate_proth_numbers,
        "pocklington": generate_pocklington_numbers,
        "ramzy": generate_ramzy_numbers,
        "rao": generate_rao_numbers,
    }

    try:
        if number_type in prime_generators:
            p_count, z_count, _ = compute_number_distribution(n, "g:1.0")
            primes = []
            composites = []
            try:
                primes = prime_generators[number_type](p_count, start, end, r=r)
            except ValueError:
                pass
            try:
                if z_count > 0:
                    composites = generate_numbers(z_count, start, end, r=r, p_count=0, z_count=z_count, max_attempts=10000)
            except ValueError:
                pass
            result = sorted(primes + composites)
            return result

        else:
            p_count, z_count, _ = compute_number_distribution(n, num_type)
            result = generate_numbers(n, start, end, r=r, p_count=p_count, z_count=z_count, max_attempts=10000)
            return result

    except ValueError as e:
        print(f"⚠️ Fehler bei Test '{testname}': {e}")
        return []
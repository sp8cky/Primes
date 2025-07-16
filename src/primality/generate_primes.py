import random
from typing import List, Dict
from collections import defaultdict
from sympy import isprime, primerange, primefactors
from math import log2
import src.primality.helpers as helpers
from src.primality.test_config import *
from src.analysis.dataset import extract_base_label

def assign_custom_numbers_to_group(
    group_name: str,
    number_list: List[int],
    TEST_CONFIG: dict
) -> Dict[str, List[int]]:
    """
    Weist eine benutzerdefinierte Liste von Zahlen allen Tests einer bestimmten Gruppe zu.

    Args:
        group_name (str): Name der Zielgruppe (z. B. "MillerTests")
        number_list (List[int]): Liste von Zahlen, die zugewiesen werden sollen
        TEST_CONFIG (dict): Konfiguration aller Tests

    Returns:
        Dict[str, List[int]]: Mapping von Testnamen zu der gemeinsam zugewiesenen Liste
    """
    assigned = {}
    for testname, conf in TEST_CONFIG.items():
        if conf.get("testgroup") == group_name:
            assigned[testname] = number_list
            print(f"✅ Benutzerdefinierte Zahlen an Test '{testname}' der Gruppe '{group_name}' zugewiesen.")
    if not assigned:
        print(f"⚠️ Keine Tests mit der Gruppe '{group_name}' gefunden.")
    return assigned

# calculate the distribution of prime and composite numbers based on the num_type
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
    n_composites = n - n_primes
    return n_primes, n_composites, prime_ratio


def is_valid_composite(candidate: int) -> bool:
    return candidate >= 2 and candidate % 2 == 1 and not helpers.is_real_potency(candidate) and not isprime(candidate)

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

        if unique_number_type and unique_number_type in {
            "fermat", "mersenne", "proth", "pocklington", "rao", "ramzy" #"lucas", 
        }:
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
                    print(f"🔍 Test '{testname}' num_type='{num_type}', number_type='{unique_number_type}': {len(result)} Zahlen (p: {p_count}, z: {z_count}): ZAHL1 {result}")
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
                z_count=z_count
            )
            for testname in relevant_tests:
                numbers_per_test[testname] = shared_numbers
                print(f"🔍 Test '{testname}' num_type='{num_type}', number_type='': {len(shared_numbers)} Zufallszahlen (p: {p_count}, z: {z_count}): ZAHL2 {shared_numbers}")
        except ValueError as e:
            print(f"⚠️ Fehler bei Zahlengenerierung für Gruppe '{group}': {e}")
            if not allow_partial_numbers:
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
    mersenne_candidates = []
    max_p = (end + 1).bit_length()  # größtes p so dass 2^p -1 <= end
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
        #"lucas": generate_lucas_primes,
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
                    composites = generate_numbers(z_count, start, end, r=r, p_count=0, z_count=z_count)
            except ValueError:
                pass
            result = sorted(primes + composites)
            #print(f"33🔍 Test '{testname}' num_type='{num_type}', number_type='{number_type}': {len(result)} Zahlen (p: {len(primes)}, z: {len(composites)}): {result}")
            return result

        else:
            p_count, z_count, _ = compute_number_distribution(n, num_type)
            result = generate_numbers(n, start, end, r=r, p_count=p_count, z_count=z_count)
            #print(f"44🔍 Test '{testname}' num_type='{num_type}', number_type='{number_type}': {len(result)} Zufallszahlen (p: {p_count}, z: {z_count}): {result}")
            return result

    except ValueError as e:
        print(f"⚠️ Fehler bei Test '{testname}': {e}")
        return []
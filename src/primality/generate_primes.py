import random
from sympy import isprime, primerange, primefactors
from math import log2
import src.primality.helpers as helpers
from typing import List

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
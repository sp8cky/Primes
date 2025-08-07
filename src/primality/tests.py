import src.primality.helpers as helpers
from src.primality.constants import *
from src.primality.test_protocoll import get_global_seed
import random, math, pytest
from math import gcd, log2, sqrt
from statistics import mean
from sympy import jacobi_symbol, gcd, log, factorint, primerange, isprime, divisors, totient, n_order, perfect_power, cyclotomic_poly, GF, ZZ, rem, symbols, Poly
from sympy.abc import X
from typing import Optional, List, Dict, Tuple, Any, Union

def fermat_test(n: int, k: int = 1, seed: Optional[int] = None) -> bool:
    if n <= 1: return INVALID
    if n == 2: return PRIME

    for i in range(k):
        a_seed = get_global_seed(seed, n, "Fermat", i)
        r = random.Random(a_seed)
        a = r.randint(2, n - 1)
        if gcd(a, n) != 1 or pow(a, n - 1, n) != 1:
            return COMPOSITE
    return PRIME

def miller_selfridge_rabin_test(n: int, k: int = 5, seed: Optional[int] = None) -> bool:
    if (n < 2) or (n % 2 == 0 and n > 2) or perfect_power(n): return INVALID
    if n in (2, 3): return PRIME
    # Zerlegung von n - 1 in 2^s * m
    m = n - 1
    s = 0
    while m % 2 == 0:
        m //= 2
        s += 1

    for i in range(k):
        a_seed = get_global_seed(seed, n, "Miller-Selfridge-Rabin", i)
        r = random.Random(a_seed)
        a = r.randint(2, n - 1)
        if gcd(a, n) != 1: return COMPOSITE
        if pow(a, n - 1, n) == 1: continue
        if any(pow(a, 2**j * m, n) == n - 1 for j in range(s)): continue
        return COMPOSITE

    return PRIME

def solovay_strassen_test(n: int, k: int = 5, seed: Optional[int] = None) -> bool:
    if n < 2 or (n % 2 == 0 and n > 2): return INVALID
    if n == 2 or n == 3: return PRIME

    for i in range(k):
        a_seed = get_global_seed(seed, n, "Solovay-Strassen", i)
        r = random.Random(a_seed)
        a = r.randint(2, n - 1)
        jacobi = helpers.jacobisymbol(a, n)
        if jacobi == 0 or pow(a, (n - 1) // 2, n) != jacobi % n:
            return COMPOSITE

    return PRIME


def initial_lucas_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: return INVALID
    if n == 2: return PRIME

    for a in range(2, n):
        if pow(a, n - 1, n) != 1: continue  # Bedingung (i) nicht erfüllt

        for m in range(1, n - 1):
            if pow(a, m, n) == 1:
                break  # Bedingung (ii) verletzt
        else:
            # Alle m getestet, kein Verstoß gegen Bedingung (ii)
            return PRIME  # EIN solches a gefunden → n ist prim

    return COMPOSITE  # Kein a erfüllt beide Bedingungen


def lucas_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: return INVALID
    if n == 2: return PRIME

    for a in range(2, n):
        if pow(a, n - 1, n) != 1:
            continue  # Bedingung (i) verletzt

        for m in divisors(n - 1)[:-1]:
            if pow(a, m, n) == 1:  # Bedingung (ii) verletzt
                break
        else:
            return PRIME  # a erfüllt beide Bedingungen

    return COMPOSITE  # Kein a gefunden

def optimized_lucas_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: return INVALID
    if n == 2: return PRIME

    factors = factorint(n - 1)
    for q in factors:
        for a in range(2, n):
            cond1 = pow(a, n - 1, n) == 1
            cond2 = pow(a, (n - 1) // q, n) != 1
            if cond1 and cond2:
                break
        else:
            return COMPOSITE

    return PRIME


def wilson_criterion(p: int, seed: Optional[int] = None) -> bool:
    if p <= 1: return INVALID
    result = math.factorial(p - 1) % p == p - 1
    if result:
        return PRIME
    return COMPOSITE


def aks04_test(n: int, seed: Optional[int] = None) -> bool:
    print("Prüfe AKS04-Test für", n)
    if n <= 1 or perfect_power(n): return INVALID

    # Schritt 1: Finde kleinstes r mit ord_r(n) > log^2(n)
    log_n = math.log2(n)
    log_sq = pow(log_n, 2)
    r = 2
    while True:
        if gcd(n, r) == 1 and helpers.order(n, r) > log_sq:
            break
        r += 1

    # Schritt 2: Prüfe ggT(n, a) für a <= r
    for a in range(2, r + 1):
        d = gcd(a, n)
        if 1 < d < n:
            return COMPOSITE  # echter Teiler => nicht prim

    # Schritt 3: n <= r => n ist prim
    if n <= r: return PRIME

    # Schritt 4: Polynomprüfung für a <= sqrt(phi(r)) * log n
    phi_r = totient(r)
    log_n = log2(n)
    max_a = int(sqrt(phi_r) * log_n) + 1
    mod_poly = Poly(X**r - 1, X, domain=GF(n))

    for a in range(1, max_a + 1):
        # Compute (X + a)^n mod (X^r - 1) mod n
        left = Poly(X + a, X, domain=GF(n)) ** n
        left = left.rem(mod_poly)  # Explicit remainder computation

        # Compute X^n + a mod (X^r - 1) mod n
        right = Poly(X**n + a, X, domain=GF(n)).rem(mod_poly)

        if left != right: return COMPOSITE

    return PRIME


def aks10_test(n: int, seed: Optional[int] = None) -> bool:
    print("Prüfe AKS10-Test für", n)
    if n <= 1 or perfect_power(n): return INVALID

    l = math.ceil(math.log2(n))
    l_sq = pow(l, 2)
    r = 2
    while True:
        if gcd(n, r) == 1 and helpers.order(n, r) > l_sq:
            break
        r += 1

    # Prüfe kleine Primteiler
    l_pow5 = pow(l, 5)
    for p in primerange(2, l_pow5 + 1):
        if n % p == 0:
            if p == n:
                return PRIME
            else:
                return COMPOSITE

    # polynomial condition check
    max_a = math.floor(math.sqrt(r) * l)
    mod_poly = Poly(X**r - 1, X, domain=GF(n))
    
    for a in range(1, max_a + 1):
        left = Poly(X + a, X, domain=GF(n)) ** n
        left = left.rem(mod_poly) 
        right = Poly(X**n + a, X, domain=GF(n)).rem(mod_poly)
        if left != right: return COMPOSITE

    return PRIME


# Prüft ob eine Fermat-Zahl n prim ist
def pepin_test(n: int, seed: Optional[int] = None) -> bool:
    if not helpers.is_fermat_number(n): return INVALID

    if pow(3, (n - 1) // 2, n) != n - 1: 
        return COMPOSITE
    
    return PRIME

# Prüft ob n=2^p-1 eine Mersenne-Primzahl (mit primem p) ist
def lucas_lehmer_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: return INVALID
    is_mersenne = helpers.is_mersenne_number(n)
    p = (n + 1).bit_length() - 1
    if not is_mersenne or not isprime(p): return INVALID
    if p == 2: return PRIME
    # Test
    S = 4
    sequence = [S]
    for _ in range(p - 2):
        S = (pow(S, 2, n) - 2) % n
        sequence.append(S)
    if S != 0:
        return COMPOSITE
    return PRIME

def proth_test(n: int, seed: Optional[int] = None) -> bool: ##5.6
    if n <= 1: return INVALID
    
    # Check if n is of the form K*2^m + 1 with K < 2^m
    m, temp = 0, n - 1
    while temp % 2 == 0:
        temp //= 2
        m += 1
    K = temp
    if K >= 2**m:
        return NOT_APPLICABLE
    
    # Test
    for a in range(2, n):
        if pow(a, (n - 1) // 2, n) == n - 1:
            return PRIME
    return COMPOSITE


def proth_test_variant(n: int, seed: Optional[int] = None) -> bool: ##5.9
    if n <= 1: return INVALID
    if n % 2 == 0: return COMPOSITE
    
    decomposition = helpers.find_proth_decomposition(n)
    if decomposition is None: return NOT_APPLICABLE
    
    K, e = decomposition

    # Prüfe Bedingung 2^e > K (2^n > K im Theorem)
    if 2 ** e <= K: return NOT_APPLICABLE

    for a in range(2, n):
        if pow(a, n - 1, n) != 1:
            return COMPOSITE
        if pow(a, (n - 1) // 2, n) == n - 1:
            return PRIME

    return COMPOSITE


def pocklington_test(n: int, seed: Optional[int] = None) -> bool: ##5.7
    if n <= 1: return INVALID

    # Factorize n-1 as q^m * R
    factors = factorint(n - 1)
    if not factors: return NOT_APPLICABLE
    
    q, m = next(iter(factors.items()))
    R = (n - 1) // (q ** m)
    if (n - 1) % q != 0 or R % q == 0:
        return NOT_APPLICABLE

    # Test
    for a in range(2, n):
        if pow(a, n - 1, n) == 1 and gcd(pow(a, (n - 1) // q, n) - 1, n) == 1:
            return PRIME
    return COMPOSITE

def optimized_pocklington_test(n: int, seed: Optional[int] = None) -> bool: ##5.8
    if n <= 1: return INVALID

    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n - 1)
    F = math.prod(factors.keys())
    R = (n - 1) // F

    if gcd(F, R) != 1: return NOT_APPLICABLE

    # test for each prime factor q of F
    for q in factors:
        found = False

        for a in range(2, n):
            if pow(a, n - 1, n) == 1 and gcd(pow(a, (n - 1) // q, n) - 1, n) == 1:
                found = True
                break
        if not found: return COMPOSITE

    return PRIME


def optimized_pocklington_test_variant(n: int, B: Optional[int] = None, seed: Optional[int] = None) -> bool: ##5.10
    if n <= 1: return INVALID

    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n - 1)
    F = math.prod(pow(p, e) for p, e in factors.items())
    R = (n - 1) // F

    if B is None: B = int(math.isqrt(n) // F) + 1
    if F * B <= math.isqrt(n): return NOT_APPLICABLE

    for p in primerange(2, B): 
        if R % p == 0: return COMPOSITE

    for q in factors:
        found = False
        for a in range(2, n):
            if pow(a, n - 1, n) == 1 and gcd(pow(a, (n - 1) // q, n) - 1, n) == 1:
                found = True
                break
        if not found: return COMPOSITE

    # b-Test
    b = 2
    found_b = False
    while b < n:
        if pow(b, n-1, n) == 1 and gcd(pow(b, F, n) - 1, n) == 1:
            found_b = True
            break
        b += 1
    if not found_b: return COMPOSITE

    return PRIME

def generalized_pocklington_test(n: int, seed: Optional[int] = None) -> bool: #6.12
    if n <= 1: return INVALID

    decomposition = helpers.find_pocklington_decomposition(n)
    if decomposition is None: return NOT_APPLICABLE

    K, p, e = decomposition
    
    for a in range(2, n):
        if pow(a, n - 1, n) == 1 and gcd(pow(a, (n - 1) // p, n) - 1, n) == 1:
            return PRIME
        
    return COMPOSITE

def grau_test(n: int, seed: Optional[int] = None) -> bool: ##6.13
    print(f"Prüfe Grau-Test für {n}...")
    if n <= 1: return INVALID

    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition: return NOT_APPLICABLE

    K, p, exp = decomposition
    a = helpers.find_quadratic_non_residue(p)
    if a is None: return NOT_APPLICABLE

    exponent = (n - 1) // p
    base = pow(a, exponent, n)
    phi_p = cyclotomic_poly(p, base) % n
    if phi_p != 0: return COMPOSITE
    return PRIME


def grau_probability_test(n: int, seed: Optional[int] = None) -> bool: #6.14
    print(f"Prüfe Grau-Probability-Test für {n}...")
    if n <= 1: return INVALID
    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition: return NOT_APPLICABLE
    
    K, p, n_exp = decomposition
    a = helpers.find_quadratic_non_residue(p)
    if a is None: return NOT_APPLICABLE

    log_p_K = math.log(K, p) if K != 0 else float("-inf")

    for j in range(n_exp - 1, -1, -1):
        phi_value = pow(a, K * pow(p, n_exp - j - 1), n)
        phi_p = cyclotomic_poly(p, phi_value) % n
        if (phi_p == 0) and (2 * (n_exp - j) > log_p_K + n_exp):
            return PRIME
    return COMPOSITE

def rao_test(n: int, seed: Optional[int] = None) -> bool: ## 6.6
    if n <= 1: return INVALID
    if n == 2: return PRIME
    
    # Spezielle Zerlegung für Rao-Test (n = p2^n + 1)
    decomposition = helpers.find_rao_decomposition(n)
    if not decomposition: return NOT_APPLICABLE
    p, n_exp = decomposition 

    exponent = (n - 1) // 2
    if pow(3, exponent, n) != (n - 1): return COMPOSITE
    if (pow(3, pow(2, n_exp - 1), n) + 1) % n != 0: return PRIME

    return COMPOSITE


def ramzy_test(n: int, seed: Optional[int] = None) -> bool: #6.15
    if n <= 1: return INVALID
    if n == 2: return PRIME
    decomposition = helpers.find_ramzy_decomposition(n)
    if not decomposition: return NOT_APPLICABLE

    K, p, n_exp = decomposition  # n = K*p^n + 1
    
    for j in range(0, n_exp): # Finde passendes j gemäß Bedingung p^{n-1} ≥ Kp^j
        if pow(p, n_exp - 1) >= K * pow(p, j):
            for a in range(2, n):
                # Bedingung (i): a^{Kp^{n-j-1}} ≡ L ≠ 1 mod n
                exponent = K * pow(p, n_exp - j - 1)
                L = pow(a, exponent, n)
                if L == 1: continue
                if pow(L, p**(j+1), n) == 1: # Bedingung (ii): L^{p^{j+1}} ≡ 1 mod n
                    return PRIME
    return COMPOSITE
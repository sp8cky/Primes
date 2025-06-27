import math
from math import gcd
from sympy import is_quad_residue, cyclotomic_poly, isprime, primefactors, perfect_power, n_order, primerange
# Helper functions for primality tests and number theory
def divides(a: int, b: int) -> bool:
    if a == 0: raise ValueError("Division by 0.")
    return b % a == 0

# Check if n is a real potency, i.e., n = a^b with a, b > 1
def is_real_potency(n: int) -> bool:
    result = perfect_power(n)
    return isinstance(result, tuple) and result[1] > 1

# calculate the order of n modulo r
def order(n: int, r: int) -> int:
    try:
        return n_order(n, r)
    except ValueError:
        return 0  # Wenn gcd(n, r) ≠ 1 oder Ordnung nicht definiert
    

def find_rao_decomposition(n: int) -> tuple[int, int] | None:
    if n <= 3: return None
    R_minus_1 = n - 1
    n = 1
    while (1 << n) <= R_minus_1:  # 2^n <= R-1
        if R_minus_1 % (1 << n) == 0:
            p = R_minus_1 // (1 << n)
            if p > 0 and isprime(p):
                return (p, n)
        n += 1
    return None

# Find K, p, n such that N = K*p^n + 1 with K < p^n
def find_pocklington_decomposition(n: int) -> tuple:
    if n <= 2: return None
        
    n_minus_1 = n - 1
    
    for p in range(2, n_minus_1 + 1):
        if not isprime(p): continue

        e = 1
        while True:
            p_pow_e = p ** e
            if p_pow_e > n_minus_1:
                break
            if n_minus_1 % p_pow_e == 0:
                K = n_minus_1 // p_pow_e
                if K < p_pow_e:
                    return (K, p, e)
            e += 1
    return None


def find_all_decompositions(n: int) -> list:
    if n < 3: return []
    
    decompositions = []
    N_minus_1 = n - 1
    primes = primefactors(N_minus_1) # Get all prime factors of n-1
    
    for p in primes:
        # Find the maximum exponent n such that p^n divides n-1
        n = 0
        temp = N_minus_1
        while temp % p == 0:
            temp = temp // p
            n += 1
        # For each possible exponent from 1 to max exponent
        for current_n in range(1, n + 1):
            p_power = p ** current_n
            if N_minus_1 % p_power != 0:
                continue
            
            K = N_minus_1 // p_power
            if K < p_power and K > 0:
                decompositions.append((K, p, current_n))
    
    return decompositions

# Find smallest quadratic non-residue modulo p
def find_quadratic_non_residue(p: int) -> int:
    for a in range(2, p):
        if not is_quad_residue(a, p):
            return a
    return None

# Compute nth cyclotomic polynomial at x
def cyclotomic_polynomial(n, x: int) -> int:
    return cyclotomic_poly(n, x)

# Prüft ob n eine Fermat-Zahl der Form n=2^(2^k) + 1 ist
def is_fermat_number(n: int) -> bool:
    if n < 3: return False
    k = 0
    while True:
        fermat_candidate = (1 << (1 << k)) + 1  # Berechnet 2^(2^k) + 1 effizient
        if fermat_candidate == n:
            return True
        if fermat_candidate > n:  # Kein weiteres k kann n ergeben
            return False
        k += 1

# berechnet GF(b, m) = b^{2^m} + 1
def calculate_generalized_fermat_number(b: int, m: int) -> int:
    if b < 2 or m < 1: raise ValueError("Ungültige Parameter: b ≥ 2, m ≥ 1 erforderlich.")
    return pow(b, 1 << m) + 1  # b^{2^m} + 1

# Prüft ob n eine Mersenne-Zahl der Form n=2^p - 1 ist (für beliebiges p ≥ 2)
def is_mersenne_number(n: int) -> bool:
    if n <= 1: return False
    m = n + 1  # M_p + 1 = 2^p
    p = 0
    while m > 1:
        if m % 2 != 0:
            return False
        m = m // 2
        p += 1
    return p >= 2
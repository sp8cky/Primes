import math
from math import gcd
from sympy import is_quad_residue, cyclotomic_poly, isprime, primefactors, perfect_power, n_order
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

# Find K, p, n such that N = K*p^n + 1 with K < p^n
def find_pocklington_decomposition(n: int) -> tuple:
    if n <= 2: return None
        
    n_minus_1 = n - 1
    max_p = int(math.sqrt(n_minus_1)) + 1
    
    for p in range(2, max_p + 1):
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

# check if n is a Fermat number
def is_fermat_number(n: int) -> bool:
    if n < 3: return False
    k = 0
    value = 2
    while value < n:
        value = 2 ** (2 ** k) + 1
        if value == n:
            return True
        k += 1
    return False

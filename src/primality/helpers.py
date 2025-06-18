from math import gcd
from sympy import is_quad_residue
from sympy import cyclotomic_poly


# Helper functions for primality tests and number theory
def divides(a: int, b: int) -> bool:
    if a == 0: raise ValueError("Division by 0.")
    return b % a == 0

# Check if n is a real potency, i.e., n = a^b with a, b > 1
def is_real_potency(n: int):
    if n <= 1: return False
    for b in range(2, int(n**0.5) + 1):
        a = round(n ** (1 / b))
        if a ** b == n:
            return True
    return False

# calculate the order of n modulo r
def order(n: int, r: int) -> int:
    if gcd(n, r) != 1: return 0
    for k in range(1, r):
        if pow(n, k, r) == 1:
            return k
    return 0

# Find K, p, n such that N = K*p^n + 1 with K < p^n
def find_pocklington_decomposition(N):
    if N <= 2: return None
        
    N_minus_1 = N - 1
    # Try small primes first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if N_minus_1 % p == 0:
            n = 0
            temp = N_minus_1
            while temp % p == 0:
                temp //= p
                n += 1
            K = temp
            if K < p**n:
                return (K, p, n)
    return None

# Find smallest quadratic non-residue modulo p
def find_quadratic_non_residue(p):
    for a in range(2, p):
        if not is_quad_residue(a, p):
            return a
    return None

# Compute nth cyclotomic polynomial at x
def cyclotomic_polynomial(n, x):
    return cyclotomic_poly(n, x)
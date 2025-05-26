import helpers
import math
from math import gcd
from sympy import isprime, factorint

def fermat_criterion(a: int, n: int) -> bool:
    if n <= 1 or a <= 1 or a >= n:
        raise ValueError("a must be in the range 2 to n-1 and n must be greater than 1")
    if gcd(a, n) != 1:
        return False
    return pow(a, n - 1, n) == 1

def wilson_criterion(p: int) -> bool:
    if p <= 1:
        raise ValueError("p must be greater than 1")
    return math.factorial(p - 1) % p == p - 1

def initial_lucas_test(a: int, n: int) -> bool:
    if n <= 1 or a <= 1 or a >= n:
        raise ValueError("a must be in the range 2 to n-1 and n must be greater than 1")
    if pow(a, (n-1), n) != 1:
        return False
    for m in range(1, n-1):
        if pow(a, m, n) == 1:
            return False
    return True

def lucas_test(a: int, n: int) -> bool:
    if n <= 1 or a <= 1 or a >= n:
        raise ValueError("a must be in the range 2 to n-1 and n must be greater than 1")
    if pow(a, (n-1), n) != 1:
        return False
    for m in range(1, n):
        if helpers.divides(m, n) and pow(a, m, n) == 1:
            return False
    return True

def optimized_lucas_test(n: int) -> bool:
    if n <= 1:
        raise ValueError("n must be greater than 1")
    for q in factorint(n-1):
        for a in range (2, n):
            if pow(a, (n-1), n) == 1 and pow(a, (n-1) // q, n) != 1:
                break
        else:
            return False
    return True

# TODO: Satz 4.5 - 4.8
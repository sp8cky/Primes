import src.primality.helpers as helpers
import random, math
from math import gcd
from sympy import factorint
from statistics import mean


def fermat_criterion(n: int, k: int = 1) -> bool:
    if n <= 1:
        raise ValueError("n must be greater than 1")
    if n == 2: 
        return True
    
    for _ in range(k):
        a = random.randint(2, n-1)
        if gcd(a, n) != 1:
            return False
        if pow(a, n-1, n) != 1:
            return False
    return True # wahrscheinlich prim

def wilson_criterion(p: int) -> bool:
    if p <= 1:
        raise ValueError("p must be greater than 1")
    return math.factorial(p - 1) % p == p - 1

def initial_lucas_test(n: int) -> bool:
    if n <= 1: 
        raise ValueError("n must be greater than 1")
    if n == 2: return True
    a = random.randint(2, n-2)
    if pow(a, (n-1), n) != 1:
        return False
    for m in range(1, n-1):
        if pow(a, m, n) == 1:
            return False
    return True

def lucas_test(n: int) -> bool:
    if n <= 1:
        raise ValueError("n must be greater than 1")
    if n == 2:
        return True
    a = random.randint(2, n-1)
    if pow(a, (n-1), n) != 1:
        return False
    for m in range(1, n):
        if helpers.divides(m, (n-1)) and pow(a, m, n) == 1:
            return False
    return True

def optimized_lucas_test(n: int) -> bool:
    if n <= 1:
        raise ValueError("n must be greater than 1")
    if n == 2: return True
    for q in factorint(n-1):
        for a in range(2, n):
            if pow(a, (n-1), n) == 1 and pow(a, (n-1) // q, n) != 1:
                break
        else:
            return False
    return True

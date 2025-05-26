import helpers
import random
from math import gcd
from sympy import *

def miller_selfridge_rabin_test(n, rounds=5):
    if (n < 2) or (n % 2 == 0 and n > 2) or helpers.is_real_potency(n):
        return False
    # Form of n-1
    m = n-1
    k = 0
    while m % 2 == 0:
        m //= 2
        k += 1
    # Iterations (rounds)
    for _ in range(rounds):
        if rounds <= 0:
            raise ValueError("Rounds must be a positive integer.")
        rounds -= 1
        a = random.randint(1, n - 1)
        if gcd(a, n) != 1:
            return False
        if pow(a, m, n) == 1:
            continue
        for i in range(k):
            if pow(a, 2**i * m, n) % n == n-1:
                break
        else:
            return False
    return True

def solovay_strassen_test(n, rounds=5):
    if rounds <= 0:
            raise ValueError("Rounds must be a positive integer.")
    if n < 2 or n % 2 == 0 and n > 2:
        raise ValueError("n must be odd and > 1.")
    if n == 2:
        return True

    for _ in range(rounds):
        a = random.randint(2, n - 1)
        if jacobi_symbol(a, n) == 0 or pow(a, (n-1)//2, n) != jacobi_symbol(a, n) % n:
            return False
    return True


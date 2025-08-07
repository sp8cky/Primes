import sympy
from src.primality.constants import USE_NJIT
from typing import List, Optional, Tuple
from statistics import mean
from sympy import jacobi_symbol, is_quad_residue, cyclotomic_poly,  gcd, log, factorint, primerange, isprime, divisors, totient, n_order, perfect_power, cyclotomic_poly, GF, ZZ, rem, symbols, Poly
from numba import njit
import math
import numpy as np

@njit
def is_prime(n: int) -> bool:
    if USE_NJIT:
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    else:
        return sympy.isprime(n)

@njit
def gcd(a: int, b: int) -> int:
    if USE_NJIT:
        while b:
            a, b = b, a % b
        return a
    else:
        return math.gcd(a, b)

@njit
def is_quad_residue(a: int, p: int) -> bool:
    if USE_NJIT:
        if p <= 2:
            return False
        return pow(a, (p - 1) // 2, p) == 1
    else:
        return sympy.is_quad_residue(a, p)

@njit
def find_quad_non_residue(p: int) -> int:
    if USE_NJIT:
        for a in range(2, p):
            if not is_quad_residue(a, p):
                return a
        return -1
    else:
        for a in range(2, p):
            if not sympy.is_quad_residue(a, p):
                return a
        return -1


@njit
def log2(n: int) -> float:
    if USE_NJIT:
        return math.log(n) / math.log(2)
    else:
        return math.log2(n)

@njit
def sqrt(n: int) -> float:
    return math.sqrt(n)

@njit
def jacobi_symbol(a: int, n: int) -> int:
    if n <= 0 or n % 2 == 0:
        return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a = a // 2
            if n % 8 == 3 or n % 8 == 5:
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0

@njit
def is_perfect_power(n: int) -> bool:
    if USE_NJIT:
        if n <= 1:
            return True
        for b in range(2, int(math.log2(n)) + 2):
            low = 2
            high = int(n ** (1.0 / b)) + 2
            while low <= high:
                mid = (low + high) // 2
                power = 1
                for _ in range(b):
                    power *= mid
                    if power > n:
                        break
                if power == n:
                    return True
                elif power < n:
                    low = mid + 1
                else:
                    high = mid - 1
        return False
    else:
        return sympy.perfect_power(n)

@njit
def product(arr):
    if USE_NJIT:
        result = 1
        for x in arr:
            result *= x
        return result
    else:
        return math.prod(arr)


@njit
def divides(a: int, b: int) -> bool:
    if a == 0:
        return False
    return b % a == 0


def totient(n: int) -> int:
    if USE_NJIT:
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result
    else:
        return sympy.totient(n)   


@njit
def order(n: int, r: int) -> int:
    if USE_NJIT:
        if gcd(n, r) != 1:
            return 0
        k = 1
        t = r % n
        while t != 1:
            t = (t * r) % n
            k += 1
            if k > n:
                return 0
        return k
    else:
        if gcd(n, r) != 1:
            return 0
        try:
            return sympy.n_order(n, r)
        except ValueError:
            return 0


@njit
def randint(low: int, high: int) -> int:
    if USE_NJIT:
        return low + int(np.random.rand() * (high - low + 1))
    else:
        return math.randint(low, high)

@njit
def divisors(n: int):
    result = []
    for i in range(1, n + 1):
        if n % i == 0:
            result.append(i)
    return result

@njit
def factorint(n: int):
    if USE_NJIT:
        factors = []
        i = 2
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 1
        if n > 1:
            factors.append(n)
        return factors
    else:
        return sympy.factorint(n)


@njit
def factorial(n: int):
    if USE_NJIT:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    else:
        return sympy.factorial(n)

@njit
def njit_primerange(start: int, end: int):
    if USE_NJIT:
        primes = []
        for n in range(start, end):
            if is_prime(n):
                primes.append(n)
        return primes
    else:
        return sympy.primerange(start, end)



@njit
def prime_factors(n: int) -> List[int]:
    if USE_NJIT:
        factors = []
        if n < 2: return factors

        while n % 2 == 0:
            factors.append(2)
            n //= 2

        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 2

        if n > 1: factors.append(n)
        return factors
    else:
        return sympy.primefactors(n)


@njit
def njit_ceil(x: float) -> int:
    if USE_NJIT:
        i = int(x)
        return i if x == i else i + 1
    else:
        return math.ceil(x)

@njit
def njit_floor(x: float) -> int:
    if USE_NJIT:
        return int(x)
    else:
        return math.floor(x)



def bit_length(n: int) -> int:
    count = 0
    while n:
        count += 1
        n >>= 1
    return count


@njit
def find_proth_decomposition(n: int):
    if n <= 2 or n % 2 == 0: return (-1, -1)
    m = n - 1
    e = 0
    while m % 2 == 0:
        m //= 2
        e += 1
    K = m
    if K % 2 == 1:
        return (K, e)
    return (-1, -1)



# Find K, p, n such that N = K*p^n + 1 with K < p^n
def find_pocklington_decomposition(n: int) -> tuple:
    if n <= 2: return None
        
    n_minus_1 = n - 1
    prime_factors = sorted(prime_factors(n_minus_1), reverse=True)
    
    for p in prime_factors:
        max_e = 0
        # Finde maximale e, sodass p^e | n_minus_1
        while (p ** (max_e + 1)) <= n_minus_1 and n_minus_1 % (p ** (max_e + 1)) == 0:
            max_e += 1
        
        # Prüfe von der maximalen Potenz rückwärts
        for e in range(max_e, 0, -1):
            p_pow_e = p ** e
            if n_minus_1 % p_pow_e != 0:
                continue
            K = n_minus_1 // p_pow_e
            if K < p_pow_e:  # Pocklington-Bedingung
                return (K, p, e)
    
    return None


@njit
def find_rao_decomposition(n: int):
    if n <= 3: return (-1, -1)
    n_minus_1 = n - 1
    if USE_NJIT:
        max_e = bit_length(n)
    else:
        max_e = n.bit_length()
    for e in range(2, max_e + 1):
        power = 1 << e
        if n_minus_1 % power == 0:
            p = n_minus_1 // power
            if p > 1 and is_prime(p):
                return (p, e)
    return (-1, -1)

@njit
def is_fermat_number(n: int) -> bool:
    if n < 3: return False
    k = 0
    while True:
        fermat_candidate = (1 << (1 << k)) + 1
        if fermat_candidate == n:
            return True
        if fermat_candidate > n:
            return False
        k += 1

@njit
def is_mersenne_number(n: int) -> bool:
    if n <= 1: return False
    m = n + 1
    while m > 1:
        if m % 2 != 0:
            return False
        m = m // 2
    return True


@njit
def mean(values):
    total = 0.0
    for v in values:
        total += v
    return total / len(values) if len(values) > 0 else 0.0

@njit
def std_dev(values, mean_val):
    total = 0.0
    for v in values:
        total += (v - mean_val) ** 2
    return math.sqrt(total / len(values)) if len(values) > 0 else 0.0

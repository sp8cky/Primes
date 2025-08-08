import sympy
from src.primality.constants import USE_NJIT
from typing import List, Optional, Tuple
from statistics import mean
from sympy import jacobi_symbol, is_quad_residue, cyclotomic_poly,  gcd, log, factorint, primerange, isprime, divisors, totient, n_order, perfect_power, cyclotomic_poly, GF, ZZ, rem, symbols, Poly
from numba import njit
import math
from numba.typed import Dict
from numba.types import int64
import numpy as np

# -------------------- Primitive Methoden --------------------

USE_NJIT = False
@njit
def power_njit(base: int, exponent: int) -> int:
    result = 1
    e = exponent
    b = base
    while e > 0:
        if e & 1:
            result *= b
        b *= b
        e >>= 1
    return result

def power_py(base: int, exponent: int) -> int:
    return math.pow(base, exponent)

power = power_njit if USE_NJIT else power_py

@njit
def modexp_njit(base: int, exponent: int, modulus: int) -> int:
    result = 1
    base = base % modulus
    e = exponent
    while e > 0:
        if e & 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        e >>= 1
    return result

def modexp_py(base: int, exponent: int, modulus: int) -> int:
    return pow(base, exponent, modulus)

modexp = modexp_njit if USE_NJIT else modexp_py


@njit
def gcd_njit(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def gcd_py(a: int, b: int) -> int:
    return math.gcd(a, b)

gcd = gcd_njit if USE_NJIT else gcd_py


@njit
def is_prime_njit(n: int) -> bool:
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

def is_prime_py(n: int) -> bool:
    return sympy.isprime(n)

is_prime = is_prime_njit if USE_NJIT else is_prime_py


@njit
def is_quad_residue_njit(a: int, p: int) -> bool:
    if p <= 2: return False
    return modexp_njit(a, (p - 1) // 2, p) == 1

def is_quad_residue_py(a: int, p: int) -> bool:
    return sympy.is_quad_residue(a, p)

is_quad_residue = is_quad_residue_njit if USE_NJIT else is_quad_residue_py


@njit
def find_quad_non_residue_njit(p: int) -> int:
    for a in range(2, p):
        if not is_quad_residue(a, p):
            return a
    return -1

def find_quad_non_residue_py(p: int) -> int:
    for a in range(2, p):
        if not sympy.is_quad_residue(a, p):
            return a
    return -1

find_quad_non_residue = find_quad_non_residue_njit if USE_NJIT else find_quad_non_residue_py


@njit
def log2_njit(n: int) -> float:
    return math.log(n) / math.log(2)

def log2_py(n: int) -> float:
    return math.log2(n)

log2 = log2_njit if USE_NJIT else log2_py


@njit
def sqrt(n: int) -> float:
    return math.sqrt(n)


@njit
def jacobisymbol(a: int, n: int) -> int:
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
def is_perfect_power_njit(n: int) -> bool:
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

def is_perfect_power_py(n: int) -> bool:
    return sympy.perfect_power(n)

if USE_NJIT:
    is_perfect_power = is_perfect_power_njit
else:
    is_perfect_power = is_perfect_power_py
#is_perfect_power = is_perfect_power_njit if USE_NJIT else is_perfect_power_py


@njit
def product_njit(arr):
    result = 1
    for x in arr:
        result *= x
    return result

def product_py(arr):
    return math.prod(arr)

product = product_njit if USE_NJIT else product_py


@njit
def divides_njit(a: int, b: int) -> bool:
    if a == 0:
        return False
    return b % a == 0



def euler_totient_njit(n: int) -> int:
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

def euler_totient_py(n: int) -> int:
    return sympy.totient(n)

euler_totient = euler_totient_njit if USE_NJIT else euler_totient_py


@njit
def order_njit(a: int, n: int) -> int:
    if gcd(a, n) != 1: return 0
    k = 1
    t = a % n
    while t != 1:
        t = (t * a) % n
        k += 1
        if k > n:
            return 0
    return k

def order_py(a: int, n: int) -> int:
    if gcd(a, n) != 1: return 0
    try:
        return sympy.n_order(a, n)
    except ValueError:
        return 0

order = order_njit if USE_NJIT else order_py


@njit
def lcg(seed: int) -> int:
    a = 1664525
    c = 1013904223
    m = 2 ** 32
    return (a * seed + c) % m

@njit
def rand_seed_njit(seed: int, low: int, high: int) -> int:
    rnd = lcg(seed)
    return low + rnd % (high - low + 1)

def rand_seed_py(seed: int, low: int, high: int) -> int:
    return np.random.randint(low, high + 1)

rand_seed = rand_seed_njit if USE_NJIT else rand_seed_py


@njit
def divisors(n: int):
    result = []
    for i in range(1, n + 1):
        if n % i == 0:
            result.append(i)
    return result



@njit
def factorint_njit(n: int):
    factors = Dict.empty(key_type=int64, value_type=int64)
    i = 2
    while i * i <= n:
        count = 0
        while n % i == 0:
            count += 1
            n //= i
        if count > 0:
            factors[i] = count
        i += 1
    if n > 1:
        factors[n] = 1
    return factors

def factorint_py(n: int):
    return sympy.factorint(n)

factorint = factorint_njit if USE_NJIT else factorint_py


@njit
def factorial_njit(n: int):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def factorial_py(n: int):
    return math.factorial(n)

factorial = factorial_njit if USE_NJIT else factorial_py


@njit
def primerange_njit(start: int, end: int):
    primes = []
    for n in range(start, end):
        if is_prime(n):
            primes.append(n)
    return primes

primerange = primerange_njit if USE_NJIT else sympy.primerange


@njit
def calc_prime_factors_njit(n: int) -> List[int]:
    factors = []
    if n < 2:
        return factors
    if n % 2 == 0:
        factors.append(2)
        while n % 2 == 0:
            n //= 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            factors.append(i)
            while n % i == 0:
                n //= i
        i += 2
    if n > 1:
        factors.append(n)
    return factors

def calc_prime_factors_py(n: int) -> List[int]:
    return sympy.factorint(n).keys()

calc_prime_factors = calc_prime_factors_njit if USE_NJIT else calc_prime_factors_py


@njit
def next_item_njit(keys: List[int], values: List[int]) -> Tuple[int, int]:
    return keys[0], values[0]

def next_item_py(d: dict) -> Tuple[int, int]:
    return next(iter(d.items()))

next_item = next_item_njit if USE_NJIT else next_item_py


@njit
def product_keys_njit(keys: List[int]) -> int:
    result = 1
    for k in keys:
        result *= k
    return result

def product_keys_py(d) -> int:
    if isinstance(d, dict):
        iterable = d.keys()
    else:
        iterable = d
    result = 1
    for k in iterable:
        result *= k
    return result

product_keys = product_keys_njit if USE_NJIT else product_keys_py


@njit
def bit_length_njit(n: int) -> int:
    count = 0
    while n:
        count += 1
        n >>= 1
    return count

def bit_length_py(n: int) -> int:
    return n.bit_length()

bit_length = bit_length_njit if USE_NJIT else bit_length_py



@njit
def ceil(x: float) -> int:
    i = int(x)
    return i if x == i else i + 1

@njit
def floor(x: float) -> int:
    return int(x)




@njit
def find_proth_decomposition_njit(n: int) -> Tuple[int, int]:
    if n <= 2 or n % 2 == 0:
        return (-1, -1)
    m = n - 1
    e = 0
    while m % 2 == 0:
        m //= 2
        e += 1
    K = m
    if K % 2 == 1:
        return (K, e)
    return (-1, -1)


def find_proth_decomposition_py(n: int) -> Tuple[int, int]:
    if n <= 2 or n % 2 == 0:
        return (-1, -1)
    m = n - 1
    e = 0
    while m % 2 == 0:
        m //= 2
        e += 1
    K = m
    if K % 2 == 1:
        return (K, e)
    return (-1, -1)

find_proth_decomposition = find_proth_decomposition_njit if USE_NJIT else find_proth_decomposition_py


@njit
def find_pocklington_decomposition_njit(n: int) -> Tuple[int, int, int]:
    if n <= 2:
        return (-1, -1, -1)
        
    n_minus_1 = n - 1
    prime_factors = sorted(calc_prime_factors_njit(n_minus_1), reverse=True)

    for p in prime_factors:
        max_e = 0
        while (p ** (max_e + 1)) <= n_minus_1 and n_minus_1 % (p ** (max_e + 1)) == 0:
            max_e += 1
        for e in range(max_e, 0, -1):
            p_pow_e = p ** e
            if n_minus_1 % p_pow_e != 0:
                continue
            K = n_minus_1 // p_pow_e
            if K < p_pow_e:
                return (K, p, e)
    return (-1, -1, -1)


# Find K, p, n such that N = K*p^n + 1 with K < p^n
def find_pocklington_decomposition_py(n: int) -> tuple:
    if n <= 2: return (-1, -1, -1)
        
    n_minus_1 = n - 1
    prime_factors = sorted(sympy.primefactors(n_minus_1), reverse=True)

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
    
    return (-1, -1, -1)


find_pocklington_decomposition = find_pocklington_decomposition_njit if USE_NJIT else find_pocklington_decomposition_py

@njit
def find_rao_decomposition_njit(n: int) -> Tuple[int, int]:
    if n <= 3:
        return (-1, -1)
    n_minus_1 = n - 1
    max_e = bit_length_njit(n)
    for e in range(2, max_e + 1):
        power = 1 << e
        if n_minus_1 % power == 0:
            p = n_minus_1 // power
            if p > 1 and is_prime_njit(p):
                return (p, e)
    return (-1, -1)


def find_rao_decomposition_py(n: int):
    if n <= 3:
        return (-1, -1)
    n_minus_1 = n - 1
    max_e = bit_length_py(n)
    for e in range(2, max_e + 1):
        power = 1 << e
        if n_minus_1 % power == 0:
            p = n_minus_1 // power
            if p > 1 and sympy.isprime(p):
                return (p, e)
    return (-1, -1)


find_rao_decomposition = find_rao_decomposition_njit if USE_NJIT else find_rao_decomposition_py

@njit
def find_ramzy_decomposition_njit(n: int) -> Tuple[int, int, int]:
    if n <= 2:
        return (-1, -1, -1)
        
    n_minus_1 = n - 1
    for p in sorted(calc_prime_factors(n_minus_1)):
        if not is_prime_njit(p):  # <- ebenfalls deine `@njit`-Variante!
            continue
        max_e = 0
        while (p ** (max_e + 1)) <= n_minus_1 and n_minus_1 % (p ** (max_e + 1)) == 0:
            max_e += 1
        if max_e == 0:
            continue
        for e in range(1, max_e + 1):
            p_pow_e = p ** e
            if n_minus_1 % p_pow_e != 0:
                continue
            K = n_minus_1 // p_pow_e
            for j in range(e):
                if (p ** (e - 1)) >= (K * (p ** j)):
                    return (K, p, e)
    return (-1, -1, -1)


def find_ramzy_decomposition_py(n: int) -> tuple[int, int, int]:
    if n <= 2: return (-1, -1, -1)
        
    n_minus_1 = n - 1
    
    # Iteriere über alle möglichen Primzahlen p als Teiler von n-1
    for p in sorted(sympy.primefactors(n_minus_1)):
        if not sympy.isprime(p): continue
        max_e = 0
        # Finde maximale Potenz p^e, die n-1 teilt
        while (p ** (max_e + 1)) <= n_minus_1 and n_minus_1 % (p ** (max_e + 1)) == 0:
            max_e += 1
            
        if max_e == 0: continue
            
        # Für alle möglichen e von 1 bis max_e
        for e in range(1, max_e + 1):
            p_pow_e = p ** e
            if n_minus_1 % p_pow_e != 0: continue
                
            K = n_minus_1 // p_pow_e
            
            # Prüfe Ramzy-Bedingung p^{e-1} >= K*p^j für mindestens ein j
            for j in range(e):
                if (p ** (e - 1)) >= (K * (p ** j)):
                    return (K, p, e)

    return (-1, -1, -1)


find_ramzy_decomposition = find_ramzy_decomposition_njit if USE_NJIT else find_ramzy_decomposition_py


@njit
def is_fermat_number(n: int) -> bool:
    if n < 3:
        return False
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
    if n <= 1:
        return False
    m = n + 1
    while m > 1:
        if m % 2 != 0:
            return False
        m //= 2
    return True

@njit
def mean_njit(values):
    total = 0.0
    for v in values:
        total += v
    return total / len(values) if len(values) > 0 else 0.0

@njit
def std_dev_njit(values, mean_val):
    total = 0.0
    for v in values:
        total += (v - mean_val) ** 2
    return math.sqrt(total / len(values)) if len(values) > 0 else 0.0


# CHANGE MODE #####################################

def change_mode(use_njit: bool):
    global USE_NJIT
    USE_NJIT = use_njit

    # Durchsuche globals nach allen Funktionen mit _njit/_py Suffix
    g = globals()
    for name in list(g.keys()):
        if name.endswith("_njit"):
            base_name = name[:-5]  # "_njit" Länge = 5
            py_name = base_name + "_py"
            if py_name in g:
                g[base_name] = g[name] if use_njit else g[py_name]

    print(f"[INFO] Helpers mode changed to: {'NJIT' if use_njit else 'Pure Python'}")
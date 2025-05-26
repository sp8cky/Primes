import src.primality.helpers as helpers
import random, math
from math import gcd
from sympy import jacobi_symbol, isprime, gcd, log, primerange, nextprime
from sympy.abc import X
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly, invert
from sympy.ntheory.modular import crt

def miller_selfridge_rabin_test(n: int, rounds=5) -> bool:
    if (n < 2) or (n % 2 == 0 and n > 2) or helpers.is_real_potency(n):
        raise ValueError("n must be an odd integer greater than 1 and not a real potency.")
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
            continue # TODO: Pseudocode überprüfen
        for i in range(k):
            if pow(a, 2**i * m, n) % n == n-1:
                break
        else:
            return False
    return True

def solovay_strassen_test(n: int, rounds=5) -> bool:
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

def aks_test(n: int) -> bool:
    if (n <= 1) or helpers.is_real_potency(n):
        raise ValueError("n must be an odd integer greater than 1 and not a real potency.")

    l = math.ceil(log(n, 2))

    # find lowest r with ord_r(n) > l^2
    r = 2
    while True:
        if (gcd(n, r) == 1) and helpers.order(n, r) > l**2:
            break
        r += 1

    # find prime dividor
    for p in primerange(2, l**5 + 1):
        if n % p == 0:
            if p == n:
                return True
            else:
                return False
    # check polynomial condition with polynom arithmetik
    max_a = math.floor(math.sqrt(r) * l)
    domain = ZZ
    for a in range(1, max_a + 1):
        mod_poly = Poly(X**r - 1, X, domain=domain)
        left = Poly((X + a)**n, X, domain=domain).trunc(n).rem(mod_poly)
        right = Poly(X**n + a, X, domain=domain).trunc(n).rem(mod_poly)
        if left != right:
            return False

    return True


print(aks_test(7))     # True
print(aks_test(11))    # True
print(aks_test(13))    # True
print(aks_test(14))    # False
print(aks_test(15))    # False
print(aks_test(97))    # True


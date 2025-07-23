import src.primality.helpers as helpers
from src.primality.test_protocoll import get_global_seed
import random, math, pytest
from math import gcd, log2, sqrt
from statistics import mean
from sympy import jacobi_symbol, gcd, log, factorint, primerange, isprime, divisors, totient, n_order, perfect_power, cyclotomic_poly, GF, ZZ, rem, symbols, Poly
from sympy.abc import X
from typing import Optional, List, Dict, Tuple, Any, Union

def fermat_test(n: int, k: int = 1, seed: Optional[int] = None) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2: return True

    for i in range(k):
        a_seed = get_global_seed(seed, n, "Fermat", i)
        r = random.Random(a_seed)
        a = r.randint(2, n - 1)
        if gcd(a, n) != 1 or pow(a, n - 1, n) != 1:
            return False
    return True

def miller_selfridge_rabin_test(n: int, k: int = 5, seed: Optional[int] = None) -> bool:
    if (n < 2) or (n % 2 == 0 and n > 2) or perfect_power(n): raise ValueError("n must be an odd integer greater than 1 and not a real potency.")
    if n in (2, 3): return True
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
        if gcd(a, n) != 1: return False
        if pow(a, n - 1, n) == 1: continue
        if any(pow(a, 2**j * m, n) == n - 1 for j in range(s)): continue
        return False

    return True

def optimized_miller_selfridge_rabin_test(n: int, k: int = 5, seed: Optional[int] = None) -> bool:
    if (n < 2) or (n % 2 == 0 and n > 2) or perfect_power(n): raise ValueError("n must be an odd integer greater than 1 and not a real potency.")
    if n in (2, 3): return True
    # Zerlegung von n - 1 in 2^s * m
    m = n - 1
    s = 0
    while m % 2 == 0:
        m //= 2
        s += 1

    for i in range(k):
        a_seed = get_global_seed(seed, n, "Optimzed-Miller-Selfridge-Rabin", i)
        r = random.Random(a_seed)
        a = r.randint(2, n - 1)
        if gcd(a, n) != 1: return False
        x = pow(a, m, n)
        if x == 1 or x == n - 1: continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else:
            return False

    return True

def solovay_strassen_test(n: int, k: int = 5, seed: Optional[int] = None) -> bool:
    if n < 2 or (n % 2 == 0 and n > 2): raise ValueError("n must be greater than 1")
    if n == 2 or n == 3: return True

    for i in range(k):
        a_seed = get_global_seed(seed, n, "Solovay-Strassen", i)
        r = random.Random(a_seed)
        a = r.randint(2, n - 1)
        jacobi = helpers.jacobisymbol(a, n)
        if jacobi == 0 or pow(a, (n - 1) // 2, n) != jacobi % n:
            return False

    return True


def optimized_solovay_strassen_test(n: int, k: int = 5, seed: Optional[int] = None) -> bool:
    if n < 2 or (n % 2 == 0 and n > 2): raise ValueError("n must be greater than 1")
    if n == 2: return True

    for i in range(k):
        a_seed = get_global_seed(seed, n, "Optimized-Solovay-Strassen", i)
        r = random.Random(a_seed)
        a = r.randint(2, n - 1)
        jacobi = jacobi_symbol(a, n)
        if jacobi == 0 or pow(a, (n - 1) // 2, n) != jacobi % n:
            return False

    return True


def initial_lucas_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2: return True

    for a in range(2, n):
        if pow(a, n - 1, n) != 1: continue  # Bedingung (i) nicht erfüllt

        for m in range(1, n - 1):
            if pow(a, m, n) == 1:
                break  # Bedingung (ii) verletzt
        else:
            # Alle m getestet, kein Verstoß gegen Bedingung (ii)
            return True  # EIN solches a gefunden → n ist prim

    return False  # Kein a erfüllt beide Bedingungen


def lucas_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2: return True

    for a in range(2, n):
        if pow(a, n - 1, n) != 1:
            continue  # Bedingung (i) verletzt

        for m in divisors(n - 1)[:-1]:
            if pow(a, m, n) == 1:  # Bedingung (ii) verletzt
                break
        else:
            return True  # a erfüllt beide Bedingungen

    return False  # Kein a gefunden

def optimized_lucas_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2: return True

    factors = factorint(n - 1)
    for q in factors:
        for a in range(2, n):
            cond1 = pow(a, n - 1, n) == 1
            cond2 = pow(a, (n - 1) // q, n) != 1
            if cond1 and cond2:
                break
        else:
            return False

    return True


def wilson_criterion(p: int, seed: Optional[int] = None) -> bool:
    if p <= 1: raise ValueError("p must be greater than 1")
    result = math.factorial(p - 1) % p == p - 1
    return result

def aks04_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1 or perfect_power(n): raise ValueError("n muss eine ungerade Zahl > 1 und keine echte Potenz sein")

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
            return False  # echter Teiler => nicht prim

    # Schritt 3: n <= r => n ist prim
    if n <= r: return True

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

        if left != right: return False

    return True



def aks10_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1 or perfect_power(n): raise ValueError("n muss eine ungerade Zahl > 1 und keine echte Potenz sein")

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
            return p == n

    # polynomial condition check
    max_a = math.floor(math.sqrt(r) * l)
    mod_poly = Poly(X**r - 1, X, domain=GF(n))
    
    for a in range(1, max_a + 1):
        # Effizient: (X + a)^n mod (X^r - 1) mit pow()
        left = pow(Poly(X + a, X, domain=GF(n)), n, mod_poly)

        # Berechne X^n + a mod (X^r - 1)
        xn_mod = pow(Poly(X, X, domain=GF(n)), n, mod_poly)
        right = (xn_mod + a) % mod_poly
        if left != right: return False

    return True


# Prüft ob eine Fermat-Zahl n prim ist
def pepin_test(n: int, seed: Optional[int] = None) -> bool:
    if n == 3: return True
    if not helpers.is_fermat_number(n): return False

    if pow(3, (n - 1) // 2, n) != n - 1: 
        return False
    
    return True

# Prüft ob n=2^p-1 eine Mersenne-Primzahl (mit primem p) ist
def lucas_lehmer_test(n: int, seed: Optional[int] = None) -> bool:
    if n <= 2: raise ValueError("n must be greater than 1")
    is_mersenne = helpers.is_mersenne_number(n)
    p = (n + 1).bit_length() - 1
    if not is_mersenne or not isprime(p): return False
    if p == 2: return True
    # Test
    S = 4
    sequence = [S]
    for _ in range(p - 2):
        S = (pow(S, 2, n) - 2) % n
        sequence.append(S)
    return (S == 0)

def proth_test(n: int, seed: Optional[int] = None) -> bool: #4.5
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Check if n is of the form K*2^m + 1 with K < 2^m
    m, temp = 0, n - 1
    while temp % 2 == 0:
        temp //= 2
        m += 1
    K = temp
    if K >= 2**m:
        return False
    
    # Test
    for a in range(2, n):
        if pow(a, (n - 1) // 2, n) == n - 1:
            return True
    return False


def proth_test_variant(n: int, seed: Optional[int] = None) -> bool: #4.8
    if n <= 1: raise ValueError("n must be greater than 1")
    if n % 2 == 0: return False

    for a in range(2, n):
        if pow(a, n - 1, n) != 1: return False
        if pow(a, (n - 1) // 2, n) == n - 1: return True
        
    return False



def pocklington_test(n: int, seed: Optional[int] = None) -> bool: #4.6
    if n <= 1: raise ValueError("n must be greater than 1")

    # Factorize n-1 as q^m * R
    factors = factorint(n - 1)
    if not factors: return False
    
    q, m = next(iter(factors.items()))
    R = (n - 1) // (q ** m)
    if (n - 1) % q != 0 or R % q == 0:
        return False

    # Test
    for a in range(2, n):
        if pow(a, n - 1, n) == 1 and gcd(pow(a, (n - 1) // q, n) - 1, n) == 1:
            return True
    return False

def optimized_pocklington_test(n: int, seed: Optional[int] = None) -> bool: #4.7
    if n <= 1: raise ValueError("n must be greater than 1")

    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n - 1)
    F = math.prod(factors.keys())
    R = (n - 1) // F

    if gcd(F, R) != 1: return False

    # test for each prime factor q of F
    for q in factors:
        found = False

        for a in range(2, n):
            if pow(a, n - 1, n) == 1 and gcd(pow(a, (n - 1) // q, n) - 1, n) == 1:
                found = True
                break
        if not found: return False

    return True


def optimized_pocklington_test_variant(n: int, B: Optional[int] = None, seed: Optional[int] = None) -> bool: #4.9
    if n <= 1: raise ValueError("n must be greater than 1")

    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n - 1)
    F = math.prod(pow(p, e) for p, e in factors.items())
    R = (n - 1) // F

    if B is None: B = int(math.isqrt(n) // F) + 1
    if F * B <= math.isqrt(n): return False

    for p in primerange(2, B): 
        if R % p == 0: return False

    for q in factors:
        found = False
        for a in range(2, n):
            if pow(a, n - 1, n) == 1 and gcd(pow(a, (n - 1) // q, n) - 1, n) == 1:
                found = True
                break
        if not found: return False

    # b-Test
    b = 2
    while b < n and pow(b, (n - 1) // F, n) == 1:
        b += 1
    if b == n: return False

    return True

def generalized_pocklington_test(n: int, seed: Optional[int] = None) -> bool: #6.12
    if n <= 1: raise ValueError("n must be greater than 1")

    decomposition = helpers.find_pocklington_decomposition(n)
    if decomposition is None: return False

    K, p, e = decomposition
    
    for a in range(2, n):
        if pow(a, n - 1, n) == 1 and gcd(pow(a, (n - 1) // p, n) - 1, n) == 1:
            return True
        
    return False

def grau_test(n: int, seed: Optional[int] = None) -> bool: #6.13
    #print(f"Prüfe Grau-Test für {n}...")
    if n <= 1: raise ValueError("n must be greater than 1")

    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition: return False

    K, p, n_exp = decomposition
    a = helpers.find_quadratic_non_residue(p)
    if a is None: return False

    exponent = (n - 1) // p
    base = pow(a, exponent, n)
    phi_p = cyclotomic_poly(p, base) % n
    return phi_p == 0


def grau_probability_test(n: int, seed: Optional[int] = None) -> bool: #6.14
    #print(f"Prüfe Grau-Probability-Test für {n}...")
    if n <= 1:  raise ValueError("n must be greater than 1")
    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition: return False
    
    K, p, n_exp = decomposition
    a = helpers.find_quadratic_non_residue(p)
    if a is None: return False

    log_p_K = math.log(K, p) if K != 0 else float("-inf")

    for j in range(n_exp - 1, -1, -1):
        phi_value = pow(a, K * pow(p, n_exp - j - 1), n)
        phi_p = cyclotomic_poly(p, phi_value) % n
        if (phi_p == 0) and (2 * (n_exp - j) > log_p_K + n_exp):
            return True
    return False

def rao_test(n: int, seed: Optional[int] = None) -> bool: # 6.6
    if n <= 3: raise ValueError("n must be greater than 1")
    
    # Spezielle Zerlegung für Rao-Test (R = p2^n + 1)
    decomposition = helpers.find_rao_decomposition(n)
    if not decomposition: return False
    p, n_exp = decomposition 

    exponent = (n - 1) // 2
    if pow(3, exponent, n) != (n - 1): return False
    cond2 = (pow(3, pow(2, n_exp - 1), n) + 1) % n == 0
    if not cond2: return True

    return False


def ramzy_test(n: int, seed: Optional[int] = None) -> bool: #6.15
    if n <= 1: raise ValueError("n must be greater than 1")
    decomposition = helpers.find_ramzy_decomposition(n)
    if not decomposition: 
        return False

    K, p, n_exp = decomposition  # N = K*p^n + 1
    
    for j in range(0, n_exp): # Finde passendes j gemäß Bedingung p^{n-1} ≥ Kp^j
        if pow(p, n_exp - 1) >= K * pow(p, j):
            for a in range(2, n):
                # Bedingung (i): a^{Kp^{n-j-1}} ≡ L ≠ 1 mod N
                exponent = K * pow(p, n_exp - j - 1)
                L = pow(a, exponent, n)
                if L == 1: continue
                if pow(L, p**(j+1), n) == 1: # Bedingung (ii): L^{p^{j+1}} ≡ 1 mod N
                    return True
    return False


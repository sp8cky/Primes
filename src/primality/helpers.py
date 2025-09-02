from math import gcd
from sympy import is_quad_residue, isprime, primefactors, n_order,primefactors, factorint

# Helper functions for primality tests and number theory
def divides(a: int, b: int) -> bool:
    if a == 0: raise ValueError("Division by 0.")
    return b % a == 0

def order(n: int, r: int) -> int:
    if gcd(n, r) != 1:
        return 0  # Ordnung nicht definiert
    try:
        return n_order(n, r)
    except ValueError:
        return 0  # Andere Fehler
    
def jacobisymbol(a, n):
    if n <= 0 or n % 2 == 0: raise ValueError("n muss eine ungerade, positive Zahl sein.")

    a = a % n
    result = 1

    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]:
                result = -result

        a, n = n, a  # Law of quadratic reciprocity
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n

    return result if n == 1 else 0

# find decomposition of n-1 = K*2^e with odd K
def find_proth_decomposition(n: int) -> tuple[int, int] | None:
    if n <= 2 or n % 2 == 0:
        return None
    
    m = n - 1
    e = 0
    while m % 2 == 0:
        m //= 2
        e += 1
    K = m
    if K % 2 == 1:
        return (K, e)
    return None



# Find K, p, n such that N = K*p^n + 1 with K < p^n
def find_pocklington_decomposition(n: int) -> tuple:
    if n <= 2: return None
        
    n_minus_1 = n - 1
    prime_factors = sorted(primefactors(n_minus_1), reverse=True)
    
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

def fast_pocklington_decomposition(n: int, max_factors: int = 3) -> tuple | None:
    """
    Schnellerer Versuch, n-1 als K*p^e + 1 mit K < p^e zu schreiben.
    max_factors: wie viele Primfaktoren maximal berücksichtigt werden (heuristisch).
    """
    if n <= 2:
        return None

    n_minus_1 = n - 1
    # Nutze sympy.factorint für effizientere Faktorisierung
    factor_map = factorint(n_minus_1)
    sorted_factors = sorted(factor_map.items(), key=lambda x: -x[0])[:max_factors]  # größte zuerst

    for p, max_e in sorted_factors:
        for e in range(max_e, 0, -1):
            p_pow_e = p ** e
            if n_minus_1 % p_pow_e != 0:
                continue
            K = n_minus_1 // p_pow_e
            if K < p_pow_e:
                return (K, p, e)

    return None


# find the smallest prime p and exponent n such that N = p2^e + 1
def find_rao_decomposition(n: int) -> tuple[int, int] | None:
    if n <= 3: return None
    n_minus_1 = n - 1
    for e in range(2, n.bit_length() + 1):
        power = 1 << e
        if n_minus_1 % power == 0:
            p = n_minus_1 // power
            if p > 1 and isprime(p):
                return (p, e)
    return None


# find prime p and exponent n such that N = K*p^n + 1 with p^{n-1} >= K*p^j
def find_ramzy_decomposition(n: int) -> tuple[int, int, int] | None:
    if n <= 2: return None
        
    n_minus_1 = n - 1
    
    # Iteriere über alle möglichen Primzahlen p als Teiler von n-1
    for p in sorted(primefactors(n_minus_1)):
        if not isprime(p): continue
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
                    
    return None



def compute_all_valid_decompositions(N):
    if N <= 2:
        return [None]
    
    decompositions = []
    N_minus_1 = N - 1
    
    # Iterate over all prime factors of N-1
    for p in sorted(primefactors(N_minus_1), reverse=True):
        max_e = 0
        # Find the maximal e such that p^e divides N-1
        while (p ** (max_e + 1)) <= N_minus_1 and N_minus_1 % (p ** (max_e + 1)) == 0:
            max_e += 1
        
        # Try all possible exponents e from 1 to max_e
        for e in range(1, max_e + 1):
            if N_minus_1 % (p ** e) != 0:
                continue
            K = N_minus_1 // (p ** e)
            
            # Check Ramzy condition: p^{e-1} >= K * p^j for some j in 0..e-1
            for j in range(e):
                if (p ** (e - 1)) >= (K * (p ** j)):
                    decompositions.append((K, p, e))
                    break  # Only need one valid j per (K,p,e)
    
    return decompositions if decompositions else [None]


# Find smallest quadratic non-residue modulo p
def find_quadratic_non_residue(p: int) -> int:
    for a in range(2, p):
        if not is_quad_residue(a, p):
            return a
    return None


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

import src.primality.helpers as helpers
from src.primality.constants import *
import random, math, hashlib
from math import gcd, log2, sqrt
from sympy import factorint
from sympy import gcd, primerange, isprime, divisors, perfect_power, cyclotomic_poly, GF, totient
from sympy.abc import X
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from typing import Optional, List, Dict, Tuple, Any, Union

test_data = {}
def init_dictionary() -> Dict[str, Any]:
    """Erzeugt ein standardisiertes Dictionary für Testdaten eines einzelnen n."""
    return {
        "true_prime": None,      # True/False, tatsächlicher Primstatus
        "is_error": None,        # True/False, Fehler bei Testausgabe
        "false_positive": None,  # True/False
        "false_negative": None,  # True/False
        "repeat_count": 0,       # Für Fehleranalyse   
        "error_count": 0,        # Für Fehleranalyse      
        "error_rate": None,      # Für Fehleranalyse
        "a_values": [],          # Liste von Tupeln/Integers (je nach Test)
        "other_fields": None,    # Kann später zu einem Dict/Tuple/List werden
        "result": None,          # True/False/None
        "reason": None,          # String oder None
    }
def init_dictionary_fields(numbers: List[int], test_name: str) -> None:
    """Initialisiert das globale `test_data`-Dictionary für einen bestimmten Test."""
    
    # Testspezifische Felder laden
    test_config = {
        "Fermat": {"a_values": []},
        "Miller-Selfridge-Rabin": {"a_values": []},
        "Solovay-Strassen": {"a_values": []},
        "Initial Lucas": {"a_values": []},
        "Lucas": {"a_values": []},
        "Optimized Lucas": {"a_values": {}},
        "Wilson": {"a_values": None},
        "AKS04": {"a_values": None, "other_fields": {}},
        "AKS10": {"a_values": None, "other_fields": {}},
        "Pepin": {"a_values": None, "other_fields": ()},
        "Lucas-Lehmer": {"a_values": None, "other_fields": ()},
        "Proth": {"a_values": []},
        "Optimized Proth": {"a_values": []},
        "Proth Variant": {"a_values": []},
        "Pocklington": {"a_values": []},
        "Optimized Pocklington": {"a_values": {}},
        "Optimized Pocklington Variant": {"a_values": {}, "other_fields": ()},
        "Generalized Pocklington": {"a_values": [], "other_fields": ()},
        "Grau": {"a_values": [], "other_fields": ()},
        "Grau Probability": {"a_values": [], "other_fields": ()},
        "Rao": {"a_values": [], "other_fields": ()},
        "Ramzy": {"a_values": [], "other_fields": ()},
        "Optimized Ramzy": {"a_values": [], "other_fields": ()},

    }.get(test_name, {})

    if test_name not in test_data:
        test_data[test_name] = {}

    for n in numbers:
        entry = init_dictionary()
        # Spezifische Defaults setzen
        for key, value in test_config.items():
            entry[key] = value
        test_data[test_name][n] = entry


def get_global_seed(global_seed: int, n: int, testname: str = "", repeat_index: int = 0) -> int:
    key = f"{global_seed}|{n}|{testname}|{repeat_index}"
    hash_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(hash_bytes[:4], "big")
    return seed

############################################################################################

def fermat_test_protocoll(n: int, k: int = 1, seed: Optional[int] = None) -> bool:
    if n <= 1: return INVALID
    
    test_data["Fermat"][n]["a_values"] = []

    if n == 2:
        test_data["Fermat"][n]["result"] = PRIME
        test_data["Fermat"][n]["a_values"].append((2, True, True))  # alle Bedingungen erfüllt
        return PRIME

    for i in range(k):
        r = random.Random(get_global_seed(seed, n, "Fermat", i))
        a = r.randint(2, n - 1)
        cond1 = gcd(a, n) == 1 

        if not cond1:
            test_data["Fermat"][n]["a_values"].append((a, False, None))
            test_data["Fermat"][n]["reason"] = "ggT ≠ 1"
            test_data["Fermat"][n]["result"] = COMPOSITE
            return COMPOSITE

        cond2 = pow(a, n - 1, n) == 1
        test_data["Fermat"][n]["a_values"].append((a, cond1, cond2))

        if not cond2:
            test_data["Fermat"][n]["reason"] = "a^{n-1} ≠ 1"
            test_data["Fermat"][n]["result"] = COMPOSITE
            return COMPOSITE

    test_data["Fermat"][n]["result"] = PRIME
    return PRIME

def miller_selfridge_rabin_test_protocoll(n: int, k: int = 5, seed: int | None = None) -> bool:
    if (n < 2) or (n % 2 == 0 and n > 2) or perfect_power(n): return INVALID

    if n in (2, 3):
        test_data["Miller-Selfridge-Rabin"][n]["result"] = PRIME
        test_data["Miller-Selfridge-Rabin"][n]["a_values"] = []
        return PRIME

    m = n - 1
    s = 0
    while m % 2 == 0:
        m //= 2
        s += 1

    test_data["Miller-Selfridge-Rabin"][n]["a_values"] = []

    for i in range(k):
        r = random.Random(get_global_seed(seed, n, "Miller-Selfridge-Rabin", i))
        a = r.randint(2, n - 1)

        if gcd(a, n) != 1:
            test_data["Miller-Selfridge-Rabin"][n]["result"] = COMPOSITE
            test_data["Miller-Selfridge-Rabin"][n]["reason"] = "ggT ≠ 1"
            return COMPOSITE

        cond1 = pow(a, m, n) == 1
        if cond1:
            test_data["Miller-Selfridge-Rabin"][n]["a_values"].append((a, True, None))
            continue

        for j in range(s):
            if pow(a, 2**j * m, n) == n - 1:
                test_data["Miller-Selfridge-Rabin"][n]["a_values"].append((a, cond1, True))
                break
        else:
            test_data["Miller-Selfridge-Rabin"][n]["a_values"].append((a, cond1, False))
            test_data["Miller-Selfridge-Rabin"][n]["reason"] = "Keine passende Potenz gefunden"
            return COMPOSITE

    test_data["Miller-Selfridge-Rabin"][n]["result"] = PRIME
    return PRIME

def solovay_strassen_test_protocoll(n: int, k: int = 5, seed: Optional[int] = None) -> bool:
    if n < 2 or (n % 2 == 0 and n > 2): return INVALID

    if n == 2 or n == 3:
        test_data["Solovay-Strassen"][n]["result"] = PRIME
        test_data["Solovay-Strassen"][n]["a_values"] = [(2, False, True)]
        return PRIME

    test_data["Solovay-Strassen"][n]["a_values"] = []

    for i in range(k):
        r = random.Random(get_global_seed(seed, n, "Solovay-Strassen", i))
        a = r.randint(2, n - 1)
        jacobi = helpers.jacobisymbol(a, n)
        cond1 = (jacobi == 0)
        cond2 = pow(a, (n - 1) // 2, n) == jacobi % n

        test_data["Solovay-Strassen"][n]["a_values"].append((a, cond1, cond2))

        if cond1:
            test_data["Solovay-Strassen"][n]["reason"] = "Jacobi-Symbol = 0"
            test_data["Solovay-Strassen"][n]["result"] = COMPOSITE
            return COMPOSITE

        if not cond2:
            test_data["Solovay-Strassen"][n]["reason"] = "Kongruenzprüfung fehlgeschlagen"
            test_data["Solovay-Strassen"][n]["result"] = COMPOSITE
            return COMPOSITE

    test_data["Solovay-Strassen"][n]["result"] = PRIME
    return PRIME

def initial_lucas_test_protocoll(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: return INVALID
    test_data["Initial Lucas"][n]["a_values"] = []

    if n == 2:
        test_data["Initial Lucas"][n]["result"] = PRIME
        return PRIME

    for a in range(2, n):
        cond1 = pow(a, n - 1, n) == 1

        if not cond1: continue  # Wichtig: nicht abbrechen, sondern nächstes a testen

        for m in range(1, n - 1):
            cond2 = pow(a, m, n) == 1
            test_data["Initial Lucas"][n]["a_values"] = (a, cond1, cond2)
            if cond2:
                break  # a ist ungeeignet → nächstes a
        else:
            test_data["Initial Lucas"][n]["result"] = PRIME
            return PRIME

    test_data["Initial Lucas"][n]["result"] = COMPOSITE
    test_data["Initial Lucas"][n]["reason"] = "Kein passendes a gefunden"
    return COMPOSITE


def lucas_test_protocoll(n: int, seed: int | None = None) -> bool:
    if n <= 1: return INVALID
    if n == 2:
        test_data["Lucas"][n]["result"] = PRIME
        return PRIME

    test_data["Lucas"][n]["a_values"] = []

    for a in range(2, n):
        cond1 = pow(a, n - 1, n) == 1

        if not cond1: continue  # nächstes a versuchen

        for m in divisors(n - 1)[:-1]:
            cond2 = pow(a, m, n) == 1
            test_data["Lucas"][n]["a_values"] = (a, cond1, cond2)
            if cond2:
                break  # Bedingung (ii) verletzt, nächstes a
        else:
            test_data["Lucas"][n]["result"] = PRIME
            return PRIME  # EIN gültiges a gefunden

    test_data["Lucas"][n]["result"] = COMPOSITE
    test_data["Lucas"][n]["reason"] = "Kein passendes a gefunden"
    return COMPOSITE


def optimized_lucas_test_protocoll(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1: return INVALID
    test_data["Optimized Lucas"][n]["a_values"] = {}
    if n == 2:
        test_data["Optimized Lucas"][n]["result"] = PRIME
        return PRIME

    factors = factorint(n - 1)
    num_prime_factors = len(factors)
    test_data["Optimized Lucas"][n]["other_fields"] = {"num_prime_factors": num_prime_factors}
    for q in factors:
        for a in range(2, n):
            cond1 = pow(a, n - 1, n) == 1
            cond2 = pow(a, (n - 1) // q, n) != 1
            test_data["Optimized Lucas"][n]["a_values"][q] = (a, cond1, cond2)
            if cond1 and cond2:
                break
        else:
            test_data["Optimized Lucas"][n]["result"] = COMPOSITE
            test_data["Optimized Lucas"][n]["reason"] = f"Kein valides a für q = {q}"
            return COMPOSITE

    test_data["Optimized Lucas"][n]["result"] = PRIME
    return PRIME


def wilson_criterion_protocoll(p: int, seed: Optional[int] = None) -> bool:
    if p <= 1: return INVALID
    result = math.factorial(p - 1) % p == p - 1
    if result:
        test_data["Wilson"][p]["result"] = PRIME
        return PRIME
    test_data["Wilson"][p]["result"] = COMPOSITE
    return COMPOSITE


def aks04_test_protocoll(n: int, seed: Optional[int] = None) -> bool:
    testname = "AKS04"

    if n <= 1 or perfect_power(n):
        test_data[testname][n]["other_fields"]["initial_check"] = False
        test_data[testname][n]["result"] = INVALID
        test_data[testname][n]["reason"] = "Ungültige Eingabe: ≤ 1 oder echte Potenz"
        return INVALID

    # Initialisiere Protokollstruktur
    test_data[testname][n]["other_fields"] = {
        "initial_check": True,
        "find_r": None,
        "gcd_check": [],
        "early_prime_check": None,
        "polynomial_check": []
    }

    log_n = log2(n)
    log_sq = pow(log_n, 2)
    r = 2
    while True:
        ord_val = helpers.order(n, r)
        if gcd(n, r) == 1 and ord_val > log_sq:
            test_data[testname][n]["other_fields"]["find_r"] = r
            break
        r += 1

    # GCD-Prüfungen: 1 < (a, n) < n für a ≤ r
    found_gcd_witness = False
    for a in range(2, r + 1):
        g = gcd(a, n)
        test_data[testname][n]["other_fields"]["gcd_check"].append((a, g))
        if 1 < g < n:
            test_data[testname][n]["result"] = COMPOSITE
            test_data[testname][n]["reason"] = f"Nichttrivialer Teiler gefunden: gcd({a}, {n}) = {g}"
            return COMPOSITE

    # Frühausstieg, falls n ≤ r
    if n <= r:
        test_data[testname][n]["other_fields"]["early_prime_check"] = True
        test_data[testname][n]["result"] = PRIME
        return PRIME
    else:
        test_data[testname][n]["other_fields"]["early_prime_check"] = False

    # Polynomtest: (X+a)^n ≡ X^n + a mod (X^r−1, n)
    phi_r = totient(r)
    log_n = log2(n)
    max_a = int(sqrt(phi_r) * log_n) + 1
    mod_poly = Poly(X**r - 1, X, domain=GF(n))

    for a in range(1, max_a + 1):
        left = Poly(X + a, X, domain=GF(n)) ** n
        left = left.rem(mod_poly)
        right = Poly(X**n + a, X, domain=GF(n)).rem(mod_poly)

        passed = (left == right)
        test_data[testname][n]["other_fields"]["polynomial_check"].append((a, passed))

        if not passed:
            test_data[testname][n]["result"] = COMPOSITE
            test_data[testname][n]["reason"] = f"Polynomprüfung für a={a} fehlgeschlagen"
            return COMPOSITE

    test_data[testname][n]["result"] = PRIME
    return PRIME



def aks10_test_protocoll(n: int, seed: Optional[int] = None) -> bool:
    if n <= 1 or perfect_power(n):
        test_data["AKS10"][n]["other_fields"]["initial_check"] = False
        test_data["AKS10"][n]["result"] = INVALID
        return INVALID

    # Reset steps if test has to be run again
    test_data["AKS10"][n]["other_fields"] = {
        "initial_check": True,
        "find_r": None,
        "prime_divisor_check": None,
        "polynomial_check": []
    }

    l = math.ceil(math.log2(n))
    l_sq = pow(l, 2)
    r = 2
    while True:
        if gcd(n, r) == 1 and helpers.order(n, r) > l_sq:
            test_data["AKS10"][n]["other_fields"]["find_r"] = r
            break
        r += 1

    # Prüfe kleine Primteiler
    l_pow5 = pow(l, 5)
    for p in primerange(2, l_pow5 + 1):
        if n % p == 0:
            if p == n:
                test_data["AKS10"][n]["result"] = PRIME
                test_data["AKS10"][n]["other_fields"]["prime_divisor_check"] = f"Primfaktor p={p} (n selbst)"
                return PRIME
            else:
                test_data["AKS10"][n]["result"] = COMPOSITE
                test_data["AKS10"][n]["other_fields"]["prime_divisor_check"] = f"Teiler p={p} von n"
                test_data["AKS10"][n]["reason"] = f"n ist durch p={p} teilbar"
                return COMPOSITE

    test_data["AKS10"][n]["other_fields"]["prime_divisor_check"] = "Keine kleinen Teiler gefunden"

    # polynomial condition check
    max_a = math.floor(math.sqrt(r) * l)
    mod_poly = Poly(X**r - 1, X, domain=GF(n))

    for a in range(1, max_a + 1):
        left = Poly(X + a, X, domain=GF(n)) ** n
        left = left.rem(mod_poly) 
        right = Poly(X**n + a, X, domain=GF(n)).rem(mod_poly)

        if left != right:
            test_data["AKS10"][n]["result"] = COMPOSITE
            test_data["AKS10"][n]["reason"] = f"Polynomprüfung für a={a} fehlgeschlagen"
            return COMPOSITE

    test_data["AKS10"][n]["result"] = PRIME
    return PRIME


def pepin_test_protocoll(n: int, seed: Optional[int] = None) -> bool:
    if n == 3:
        test_data["Pepin"][n]["result"] = PRIME
        return PRIME
    if not helpers.is_fermat_number(n):
        test_data["Pepin"][n]["result"] = INVALID
        test_data["Pepin"][n]["reason"] = "n ist keine Fermat-Zahl"
        return INVALID

    if pow(3, (n - 1) // 2, n) != n - 1: 
        test_data["Pepin"][n]["result"] = COMPOSITE
        test_data["Pepin"][n]["reason"] = "3^(n-1)/2 mod n ≠ n - 1"
        return COMPOSITE
    
    test_data["Pepin"][n]["result"] = PRIME
    return PRIME


def lucas_lehmer_test_protocoll(n: int, seed: Optional[int] = None) -> bool:
    if n < 2: raise ValueError("n must be greater than 1")
    is_mersenne = helpers.is_mersenne_number(n)
    p = (n + 1).bit_length() - 1
    if not is_mersenne or not isprime(p): 
        test_data["Lucas-Lehmer"][n]["result"] = INVALID
        test_data["Lucas-Lehmer"][n]["reason"] = "Keine Mersenne-Zahl"
        return INVALID
    if p == 2: 
        test_data["Lucas-Lehmer"][n]["result"] = PRIME
        return PRIME

    # Test
    S = 4
    sequence = [S]
    for _ in range(p - 2):
        S = (pow(S, 2, n) - 2) % n
        sequence.append(S)
    if S != 0:
        test_data["Lucas-Lehmer"][n]["result"] = COMPOSITE
        test_data["Lucas-Lehmer"][n]["reason"] = "S != 0"
        return COMPOSITE
    test_data["Lucas-Lehmer"][n]["other_fields"] = [p, sequence, S]
    test_data["Lucas-Lehmer"][n]["result"] = PRIME
    return PRIME


def proth_test_protocoll(n: int, seed: Optional[int] = None) -> bool: #4.5
    if n <= 1: return INVALID
    
    # Check if n is of the form K*2^m + 1 with K < 2^m
    m, temp = 0, n - 1
    while temp % 2 == 0:
        temp //= 2
        m += 1
    K = temp
    if K >= 2**m:
        test_data["Proth"][n]["result"] = NOT_APPLICABLE
        test_data["Proth"][n]["reason"] = "Erfüllt nicht K < 2^m"
        return NOT_APPLICABLE

    # Test
    for a in range(2, n):
        cond = pow(a, (n - 1) // 2, n) == n - 1
        if cond:
            test_data["Proth"][n]["a_values"].append((a, cond))
            test_data["Proth"][n]["result"] = PRIME
            return PRIME
    test_data["Proth"][n]["a_values"].append((a, cond))
    test_data["Proth"][n]["result"] = COMPOSITE
    return COMPOSITE


def proth_test_variant_protocoll(n: int, seed: Optional[int] = None) -> bool: #4.8
    if n <= 1: return INVALID
    if n % 2 == 0:
        test_data["Proth Variant"][n]["result"] = COMPOSITE
        test_data["Proth Variant"][n]["reason"] = "n ist nicht ungerade"
        return COMPOSITE

    decomposition = helpers.find_proth_decomposition(n)
    if decomposition is None: 
        test_data["Proth Variant"][n]["reason"] = "Keine Zerlegung gefunden"
        test_data["Proth Variant"][n]["result"] = NOT_APPLICABLE
        return NOT_APPLICABLE
    
    K, e = decomposition
    if 2 ** e <= K: 
        test_data["Proth Variant"][n]["reason"] = "K muss kleiner als 2^e sein"
        test_data["Proth Variant"][n]["result"] = NOT_APPLICABLE
        return NOT_APPLICABLE

    for a in range(2, n):
        if pow(a, n - 1, n) != 1:
            test_data["Proth Variant"][n]["result"] = COMPOSITE
            test_data["Proth Variant"][n]["reason"] = f"a^(n-1) ≠ 1 mod n"
            return COMPOSITE

        if pow(a, (n - 1) // 2, n) == n - 1:
            test_data["Proth Variant"][n]["a_values"] = [(a, True)]
            test_data["Proth Variant"][n]["result"] = PRIME
            return PRIME

    test_data["Proth Variant"][n]["result"] = COMPOSITE
    test_data["Proth Variant"][n]["reason"] = "Kein passendes a gefunden"
    return COMPOSITE


def pocklington_test_protocoll(n: int, seed: Optional[int] = None) -> bool: #4.6
    if n <= 1: return INVALID

    # Factorize n-1 as q^m * R
    factors = factorint(n - 1)
    if not factors:
        test_data["Pocklington"][n]["result"] = NOT_APPLICABLE
        test_data["Pocklington"][n]["reason"] = "Keine Faktorisierung gefunden"
        return NOT_APPLICABLE

    q, m = next(iter(factors.items()))
    R = (n - 1) // (q ** m)
    if (n - 1) % q != 0 or R % q == 0:
        test_data["Pocklington"][n]["result"] = NOT_APPLICABLE
        test_data["Pocklington"][n]["reason"] = "q erfüllt Bedingungen nicht"
        return NOT_APPLICABLE
    # Test
    for a in range(2, n):
        cond1 = pow(a, n - 1, n) == 1
        cond2 = gcd(pow(a, (n - 1) // q, n) - 1, n) == 1
        if cond1 and cond2:
            test_data["Pocklington"][n]["a_values"].append((a, cond1, cond2))
            test_data["Pocklington"][n]["result"] = PRIME
            return PRIME
        
    test_data["Pocklington"][n]["result"] = COMPOSITE
    return COMPOSITE


def optimized_pocklington_test_protocoll(n: int, seed: Optional[int] = None) -> bool: #4.7
    if n <= 1: return INVALID

    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n - 1)
    test_data["Optimized Pocklington"][n]["other_fields"] = {"num_prime_factors": len(factors)}
    F = math.prod(factors.keys())
    R = (n - 1) // F

    if gcd(F, R) != 1:
        test_data["Optimized Pocklington"][n]["result"] = NOT_APPLICABLE
        test_data["Optimized Pocklington"][n]["reason"] = "ggT(F, R) != 1"
        return NOT_APPLICABLE

    # test for each prime factor q of F
    test_data["Optimized Pocklington"][n]["a_values"] = {}
    for q in factors:
        found = False
        for a in range(2, n):
            cond1 = pow(a, n - 1, n) == 1
            cond2 = gcd(pow(a, (n - 1) // q, n) - 1, n) == 1
            if cond1 and cond2:
                #test_data["Optimized Pocklington"][n]["a_values"] = {q: [(a, cond1, cond2)]}
                test_data["Optimized Pocklington"][n]["a_values"].setdefault(q, []).append((a, cond1, cond2))
                found = True
                break
        if not found:
            test_data["Optimized Pocklington"][n]["result"] = COMPOSITE
            return COMPOSITE

    test_data["Optimized Pocklington"][n]["result"] = PRIME
    return PRIME


def optimized_pocklington_test_variant_protocoll(n: int, B: Optional[int] = None, seed: Optional[int] = None) -> bool: #4.9
    if n <= 1: return INVALID

    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n - 1)
    test_data["Optimized Pocklington"][n]["other_fields"] = {"num_prime_factors": len(factors)}
    F = math.prod(pow(p, e) for p, e in factors.items())
    R = (n - 1) // F

    if B is None:
        B = int(math.isqrt(n) // F) + 1

    if F * B <= math.isqrt(n):
        test_data["Optimized Pocklington Variant"][n]["result"] = NOT_APPLICABLE
        test_data["Optimized Pocklington Variant"][n]["reason"] = "FB !≤ √n"
        return NOT_APPLICABLE

    for p in primerange(2, B):
        if R % p == 0:
            test_data["Optimized Pocklington Variant"][n]["result"] = NOT_APPLICABLE
            test_data["Optimized Pocklington Variant"][n]["reason"] = f"R hat Primfaktor < B: {p}"
            return NOT_APPLICABLE

    test_data["Optimized Pocklington Variant"][n]["a_values"] = {}
    for q in factors:
        found = False
        for a in range(2, n):
            cond1 = pow(a, n - 1, n) == 1
            cond2 = gcd(pow(a, (n - 1) // q, n) - 1, n) == 1
            if cond1 and cond2:
                test_data["Optimized Pocklington Variant"][n]["a_values"][q] = [(a, cond1, cond2)]
                found = True
                break
        if not found:
            test_data["Optimized Pocklington Variant"][n]["result"] = COMPOSITE
            return COMPOSITE

    # b-Test
    b = 2
    found_b = False
    while b < n:
        if pow(b, n-1, n) == 1 and gcd(pow(b, F, n) - 1, n) == 1:
            found_b = True
            break
        b += 1
    if not found_b:
        test_data["Optimized Pocklington Variant"][n]["result"] = NOT_APPLICABLE
        test_data["Optimized Pocklington Variant"][n]["reason"] = "Kein passendes b gefunden"
        return NOT_APPLICABLE

    test_data["Optimized Pocklington Variant"][n]["other_fields"] = [b]
    test_data["Optimized Pocklington Variant"][n]["result"] = PRIME
    return PRIME


def generalized_pocklington_test_protocoll(n: int, seed: Optional[int] = None) -> bool: #6.12
    if n <= 1: return INVALID

    decomposition = helpers.find_pocklington_decomposition(n)
    if decomposition is None:
        test_data["Generalized Pocklington"][n]["result"] = NOT_APPLICABLE
        test_data["Generalized Pocklington"][n]["reason"] = "Keine Zerlegung gefunden"
        return NOT_APPLICABLE

    K, p, e = decomposition
    test_data["Generalized Pocklington"][n]["other_fields"] = [K, p, e]

    for a in range(2, n):
        cond1 = pow(a, n - 1, n) == 1
        cond2 = gcd(pow(a, (n - 1) // p, n) - 1, n) == 1
        if cond1 and cond2:
            test_data["Generalized Pocklington"][n]["a_values"] = [(a, cond1, cond2)]
            test_data["Generalized Pocklington"][n]["result"] = PRIME
            return PRIME

    test_data["Generalized Pocklington"][n]["result"] = COMPOSITE
    test_data["Generalized Pocklington"][n]["reason"] = "Kein geeignetes a gefunden"
    return COMPOSITE


def grau_test_protocoll(n: int, seed: Optional[int] = None) -> bool: #6.13
    if n <= 1: return INVALID

    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition:
        test_data["Grau"][n]["result"] = NOT_APPLICABLE
        test_data["Grau"][n]["reason"] = "Keine Zerlegung gefunden"
        return NOT_APPLICABLE

    K, p, n_exp = decomposition
    a = helpers.find_quadratic_non_residue(p)
    if a is None:
        test_data["Grau"][n]["result"] = NOT_APPLICABLE
        test_data["Grau"][n]["reason"] = f"Kein QNR für p={p} gefunden"
        return NOT_APPLICABLE

    exponent = (n - 1) // p
    base = pow(a, exponent, n)
    phi_p = cyclotomic_poly(p, base) % n
    
    test_data["Grau"][n]["a_values"] = [a]
    test_data["Grau"][n]["other_fields"] = [K, p, n_exp, phi_p]
    if phi_p != 0:
        test_data["Grau"][n]["result"] = COMPOSITE
        
    test_data["Grau"][n]["result"] = PRIME
    return PRIME 


def grau_probability_test_protocoll(n: int, seed: Optional[int] = None) -> bool: #6.14
    if n <= 1: return INVALID

    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition:
        test_data["Grau Probability"][n]["result"] = NOT_APPLICABLE
        test_data["Grau Probability"][n]["reason"] = "Keine Zerlegung gefunden"
        return NOT_APPLICABLE

    K, p, exp = decomposition
    test_data["Grau Probability"][n]["other_fields"] = [K, p, exp]
    log_p_K = math.log(K, p) if K != 0 else float("-inf")
    a = helpers.find_quadratic_non_residue(p)
    if a is None: 
        test_data["Grau Probability"][n]["result"] = NOT_APPLICABLE
        test_data["Grau Probability"][n]["reason"] = "Keine a gefunden"
        return NOT_APPLICABLE

    for j in range(exp - 1, -1, -1):
        phi_value = pow(a, K * pow(p, exp - j - 1), n)
        phi_p = cyclotomic_poly(p, phi_value) % n

        cond1 = (phi_p == 0)
        cond2 = (2 * (exp - j) > math.log(K, p) + exp)

        if cond1 and cond2:
            test_data["Grau Probability"][n]["a_values"] = [a]
            test_data["Grau Probability"][n]["other_fields"].extend([j])
            test_data["Grau Probability"][n]["result"] = PRIME
            return PRIME
        
    test_data["Grau Probability"][n]["result"] = COMPOSITE
    test_data["Grau Probability"][n]["reason"] = "Kein geeignetes (a,j)-Paar gefunden"
    return COMPOSITE


def rao_test_protocoll(n: int, seed: Optional[int] = None) -> bool: #6.6
    if n <= 1: return INVALID
    if n == 2:
        test_data["Rao"][n]["result"] = PRIME
        return PRIME
    
    # Spezielle Zerlegung für Rao-Test (R = p2^n + 1)
    decomposition = helpers.find_rao_decomposition(n)
    if not decomposition:
        test_data["Rao"][n]["result"] = NOT_APPLICABLE
        test_data["Rao"][n]["reason"] = "Keine Zerlegung gefunden"
        return NOT_APPLICABLE

    p, n_exp = decomposition
    test_data["Rao"][n]["other_fields"] = [p, 2, n_exp]

    exponent = (n - 1) // 2
    cond1 = pow(3, exponent, n) == (n - 1)
    if not cond1: 
        test_data["Rao"][n]["result"] = COMPOSITE
        test_data["Rao"][n]["a_values"].append((3, False, None))
        test_data["Rao"][n]["reason"] = "3^{(R-1)/2} ≠ -1 mod R"
        return COMPOSITE

    cond2 = (pow(3, pow(2, n_exp - 1), n) + 1) % n == 0
    if not cond2: 
        test_data["Rao"][n]["result"] = PRIME
        test_data["Rao"][n]["a_values"].append((3, cond1, cond2))
        test_data["Rao"][n]["reason"] = "3^{(R-1)/2} ≡ -1 und R ∤ GF(3, n-1)"
        return PRIME

    test_data["Rao"][n]["result"] = COMPOSITE
    test_data["Rao"][n]["reason"] = "3^{(R-1)/2} ≡ -1 und R | GF(3, n-1)"
    return COMPOSITE


def ramzy_test_protocoll(n: int, seed: Optional[int] = None) -> bool: #6.15
    if n <= 1: return INVALID
    if n == 2:
        test_data["Ramzy"][n]["result"] = PRIME
        return PRIME
    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition:
        test_data["Ramzy"][n]["result"] = NOT_APPLICABLE    
        test_data["Ramzy"][n]["reason"] = "Keine Zerlegung gefunden"
        return NOT_APPLICABLE

    K, p, n_exp = decomposition  # N = K*p^n + 1
    test_data["Ramzy"][n]["other_fields"] = [K, p, n_exp]
    
    for j in range(n_exp): # Finde passendes j gemäß Bedingung p^{n-1} ≥ Kp^j
        if pow(p, n_exp - 1) >= K * pow(p, j):
            for a in range(2, n):
                # Bedingung (i): a^{Kp^{n-j-1}} ≡ L ≠ 1 mod N
                exponent = K * pow(p, n_exp - j - 1)
                L = pow(a, exponent, n)
                cond1 = (L != 1)
                cond2 = (pow(L, pow(p, j+1), n) == 1)

                if cond1 and cond2:
                    test_data["Ramzy"][n]["a_values"].append((a, cond1, cond2))
                    test_data["Ramzy"][n]["result"] = PRIME
                    return PRIME
    
    test_data["Ramzy"][n]["result"] = COMPOSITE
    test_data["Ramzy"][n]["reason"] = "Kein geeignetes (a,j)-Paar gefunden"
    return COMPOSITE


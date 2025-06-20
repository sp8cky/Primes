import src.primality.helpers as helpers
import random, math, pytest
from math import gcd
from sympy import factorint
from statistics import mean
from sympy import jacobi_symbol, gcd, log, primerange
from sympy.abc import X
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from typing import Optional, List, Dict, Tuple, Any, Union

test_data = {}
def init_dictionary() -> Dict[str, Any]:
    """Erzeugt ein standardisiertes Dictionary für Testdaten eines einzelnen n."""
    return {
        "a_values": [],          # Liste von Tupeln/Integers (je nach Test)
        "other_fields": None,    # Kann später zu einem Dict/Tuple/List werden
        "result": None,          # True/False/None
        "reason": None,          # String oder None
    }
def init_dictionary_fields(numbers: List[int]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Initialisiert das globale `test_data`-Dictionary für alle Tests."""
    
    # Liste aller Tests mit ihren spezifischen Anpassungen
    tests = {
        "Fermat": {"a_values": []},
        "Wilson": {"a_values": None},  # Wilson benötigt keine a_values-Liste
        "Initial Lucas": {"a_values": [], "other_fields": ()},
        "Lucas": {"a_values": [], "other_fields": ()},
        "Optimized Lucas": {"a_values": {}},  # Als Dictionary für faktorabhängige Werte
        "Pepin": {"a_values": None, "other_fields": ()},
        "Lucas-Lehmer": {"a_values": None, "other_fields": ()},
        "Proth": {"a_values": []},
        "Pocklington": {"a_values": []},
        "Optimized Pocklington": {"a_values": {}},
        "Proth Variant": {"a_values": []},
        "Optimized Pocklington Variant": {"a_values": {}, "other_fields": ()},
        "Generalized Pocklington": {"a_values": [], "other_fields": ()},
        "Grau": {"a_values": [], "other_fields": ()},
        "Grau Probability": {"a_values": [], "other_fields": ()},
        "Miller-Rabin": {"a_values": []},
        "Solovay-Strassen": {"a_values": []},
        "AKS": {"a_values": None, "other_fields": {}},  # AKS speichert Schritte als Dict
    }
    
    for test_name, test_config in tests.items():
        test_data[test_name] = {}
        for n in numbers:
            entry = init_dictionary()
            # Überschreibe Defaults mit testspezifischen Werten
            for key, value in test_config.items():
                entry[key] = value
            test_data[test_name][n] = entry
    
    return test_data

############################################################################################

def fermat_test_protocoll(n: int, k: int = 1) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    test_data["Fermat"][n]["a_values"] = []
    if n == 2:
        test_data["Fermat"][n]["result"] = True
        test_data["Fermat"][n]["a_values"].append((2, True))
        return True

    for _ in range(k):
        a = random.randint(2, n - 1)
        cond = gcd(a, n) == 1 or pow(a, n - 1, n) == 1
        test_data["Fermat"][n]["a_values"].append((a, cond))
        if not cond:
            test_data["Fermat"][n]["result"] = False
            test_data["Fermat"][n]["reason"] = "GCD ≠ 1 or Fermat failed"
            return False

    test_data["Fermat"][n]["result"] = True
    return True


def wilson_criterion_protocoll(p: int) -> bool:
    if p <= 1: raise ValueError("p must be greater than 1")
    result = math.factorial(p - 1) % p == p - 1
    test_data["Wilson"][p]["result"] = result
    return result


def initial_lucas_test_protocoll(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    test_data["Initial Lucas"][n]["a_values"] = []
    if n == 2:
        test_data["Initial Lucas"][n]["result"] = True
        return True

    a = random.randint(2, n - 1)
    cond1 = pow(a, n - 1, n) == 1
    test_data["Initial Lucas"][n]["a_values"] = [(a, cond1, None)]
    if not cond1:
        test_data["Initial Lucas"][n]["result"] = False
        test_data["Initial Lucas"][n]["reason"] = "a^{n-1} ≠ 1"
        return False

    for m in range(1, n - 1):
        cond2 = pow(a, m, n) == 1
        test_data["Initial Lucas"][n]["a_values"] = [(a, cond1, cond2)]
        if not cond2:
            test_data["Initial Lucas"][n]["result"] = False
            test_data["Initial Lucas"][n]["reason"] = f"early break at m = {m}"
            return False

    test_data["Initial Lucas"][n]["result"] = True
    return True


def lucas_test_protocoll(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2:
        test_data["Lucas"][n]["result"] = True
        return True

    a = random.randint(2, n - 1)
    cond1 = pow(a, n - 1, n) == 1
    test_data["Lucas"][n]["a_values"] = [(a, cond1, None)]
    if not cond1:
        test_data["Lucas"][n]["result"] = False
        test_data["Lucas"][n]["reason"] = "a^{n-1} ≠ 1"
        return False

    for m in range(1, n):
        cond2 = (n - 1) % m == 0 and pow(a, m, n) == 1
        test_data["Lucas"][n]["a_values"] = [(a, cond1, cond2)]
        if not cond2:
            test_data["Lucas"][n]["result"] = False
            test_data["Lucas"][n]["reason"] = f"early break at m = {m}"
            return False

    test_data["Lucas"][n]["result"] = True
    return True


def optimized_lucas_test_protocoll(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    test_data["Optimized Lucas"][n]["a_values"] = {}
    if n == 2:
        test_data["Optimized Lucas"][n]["result"] = True
        return True

    for q in factorint(n - 1):
        for a in range(2, n):
            cond1 = pow(a, n - 1, n) == 1
            cond2 = pow(a, (n - 1) // q, n) != 1
            test_data["Optimized Lucas"][n]["a_values"][q] = (a, cond1, cond2)
            if cond1 and cond2:
                break
        else:
            test_data["Optimized Lucas"][n]["result"] = False
            test_data["Optimized Lucas"][n]["reason"] = f"No valid a for q = {q}"
            return False

    test_data["Optimized Lucas"][n]["result"] = True
    return True


def pepin_test_protocoll(n: int) -> bool:
    if not helpers.is_fermat_number(n):
        test_data["Pepin"][n]["result"] = False
        test_data["Pepin"][n]["reason"] = "n ist keine Fermat-Zahl"
        return False

    if pow(3, (n - 1) // 2, n) != n - 1: 
        test_data["Pepin"][n]["result"] = False
        test_data["Pepin"][n]["reason"] = "3^(n-1)/2 mod n ≠ n - 1"
        return False
    
    test_data["Pepin"][n]["result"] = True
    return True


def lucas_lehmer_test_protocoll(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")

    # Check if n is a Mersenne number M_p = 2^p - 1
    p = next((p for p in range(2, 32) if 2**p - 1 == n), None)
    if p is None:
        test_data["Lucas-Lehmer"][n]["result"] = False
        test_data["Lucas-Lehmer"][n]["reason"] = "Keine Mersenne-Zahl"
        return False
    
    # Test
    S = 4
    sequence = [S]
    for _ in range(p - 2):
        S = (S**2 - 2) % n
        sequence.append(S)
    is_prime = (S == 0)
    test_data["Lucas-Lehmer"][n]["other_fields"] = [p, sequence, S]
    test_data["Lucas-Lehmer"][n]["result"] = is_prime
    return is_prime


def proth_test_protocoll(n: int) -> bool: #4.5
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Check if n is of the form K*2^m + 1 with K < 2^m
    m, temp = 0, n - 1
    while temp % 2 == 0:
        temp //= 2
        m += 1
    K = temp
    if K >= 2**m:
        test_data["Proth"][n]["result"] = False
        test_data["Proth"][n]["reason"] = "Erfüllt nicht K < 2^m"
        return False
    
    # Test
    a_values = []
    for a in range(2, n):
        cond = pow(a, (n - 1) // 2, n) == n - 1
        a_values.append((a, cond))
        if cond:
            test_data["Proth"][n]["a_values"] = a_values
            test_data["Proth"][n]["result"] = True
            return True
    test_data["Proth"][n]["a_values"] = a_values
    test_data["Proth"][n]["result"] = False
    return False


def pocklington_test_protocoll(n: int) -> bool: #4.6
    if n <= 1: raise ValueError("n must be greater than 1")

    # Factorize n-1 as q^m * R
    factors = factorint(n - 1)
    if len(factors) != 1:
        test_data["Pocklington"][n]["result"] = False
        test_data["Pocklington"][n]["reason"] = "n-1 muss genau einen Primfaktor haben"
        return False
    
    q, m = next(iter(factors.items()))
    R = (n - 1) // (q ** m)
    if (n - 1) % q != 0 or R % q == 0:
        test_data["Pocklington"][n]["result"] = False
        test_data["Pocklington"][n]["reason"] = "q muss n - 1 genau m mal teilen"
        return False
    # Test
    a_values = []
    for a in range(2, n):
        cond1 = pow(a, n - 1, n) == 1
        cond2 = gcd(pow(a, (n - 1) // q, n) - 1, n) == 1
        a_values.append((a, cond1, cond2))
        if cond1 and cond2:
            test_data["Pocklington"][n]["a_values"] = a_values
            test_data["Pocklington"][n]["other_fields"] = []
            test_data["Pocklington"][n]["result"] = True
            return True
        
    test_data["Pocklington"][n]["a_values"] = a_values
    test_data["Pocklington"][n]["other_fields"] = []
    test_data["Pocklington"][n]["result"] = False
    return False


def optimized_pocklington_test_protocoll(n: int) -> bool: #4.7
    if n <= 1: raise ValueError("n must be greater than 1")

    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n - 1)
    F = math.prod(factors.keys())
    R = (n - 1) // F

    if gcd(F, R) != 1:
        test_data["Optimized Pocklington"][n]["result"] = False
        test_data["Optimized Pocklington"][n]["reason"] = "F und R müssen teilerfremd sein"
        return False

    # test for each prime factor q of F
    test_data["Optimized Pocklington"][n]["a_values"] = {}
    for q in factors:
        found = False
        test_data["Optimized Pocklington"][n]["a_values"][q] = []

        for a in range(2, n):
            cond1 = pow(a, n - 1, n) == 1
            cond2 = gcd(pow(a, (n - 1) // q, n) - 1, n) == 1
            test_data["Optimized Pocklington"][n]["a_values"][q].append((a, cond1, cond2))

            if cond1 and cond2:
                found = True
                break
        if not found:
            test_data["Optimized Pocklington"][n]["result"] = False
            return False

    test_data["Optimized Pocklington"][n]["result"] = True
    return True


def proth_test_variant_protocoll(n: int) -> bool: #4.8
    if n <= 1: raise ValueError("n must be greater than 1")
    if n % 2 == 0:
        test_data["Proth Variant"][n]["result"] = False
        test_data["Proth Variant"][n]["reason"] = "n must be odd"
        return False

    a_values = []
    for a in range(2, n):
        if pow(a, n - 1, n) != 1:
            test_data["Proth Variant"][n]["result"] = False
            test_data["Proth Variant"][n]["reason"] = f"a={a} fails a^(n-1) ≡ 1 mod n"
            return False
        
        if pow(a, (n - 1) // 2, n) == n - 1:
            a_values.append((a, True))
            test_data["Proth Variant"][n]["a_values"] = a_values
            test_data["Proth Variant"][n]["result"] = True
            return True
        else:
            a_values.append((a, False))

    test_data["Proth Variant"][n]["a_values"] = a_values
    test_data["Proth Variant"][n]["result"] = False
    test_data["Proth Variant"][n]["reason"] = "Kein passendes a gefunden"
    return False


def optimized_pocklington_test_variant_protocoll(n: int, B: Optional[int] = None) -> bool: #4.9
    if n <= 1: raise ValueError("n must be greater than 1")

    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n - 1)
    F = math.prod(p**e for p, e in factors.items())
    R = (n - 1) // F

    if B is None:
        B = int(math.isqrt(n) // F) + 1

    if F * B <= math.isqrt(n):
        test_data["Optimized Pocklington Variant"][n]["result"] = False
        test_data["Optimized Pocklington Variant"][n]["reason"] = "FB ≤ √n condition not met"
        return False

    for p in primerange(2, B):
        if R % p == 0:
            test_data["Optimized Pocklington Variant"][n]["result"] = False
            test_data["Optimized Pocklington Variant"][n]["reason"] = f"R has prime factor < B: {p}"
            return False

    test_data["Optimized Pocklington Variant"][n]["a_values"] = {}
    for q in factors:
        found = False
        test_data["Optimized Pocklington Variant"][n]["a_values"][q] = []
        for a in range(2, n):
            cond1 = pow(a, n - 1, n) == 1
            cond2 = gcd(pow(a, (n - 1) // q, n) - 1, n) == 1
            test_data["Optimized Pocklington Variant"][n]["a_values"][q].append((a, cond1, cond2))
            if cond1 and cond2:
                found = True
                break
        if not found:
            test_data["Optimized Pocklington Variant"][n]["result"] = False
            return False

    # b-Test
    b = 2
    while b < n and pow(b, (n - 1) // F, n) == 1:
        b += 1
    if b == n:
        test_data["Optimized Pocklington Variant"][n]["result"] = False
        test_data["Optimized Pocklington Variant"][n]["reason"] = "Kein b gefunden mit b^{(n-1)/F} ≠ 1 mod n"
        return False

    test_data["Optimized Pocklington Variant"][n]["other_fields"] = [b, pow(b, (n - 1) // F, n)]
    test_data["Optimized Pocklington Variant"][n]["result"] = True
    return True


def generalized_pocklington_test_protocoll(n: int) -> bool: #6.12
    if n <= 1: raise ValueError("n must be greater than 1")

    decomposition = helpers.find_pocklington_decomposition(n)
    if decomposition is None:
        test_data["Generalized Pocklington"][n]["result"] = False
        test_data["Generalized Pocklington"][n]["reason"] = "Keine Zerlegung N = K*p^n + 1 mit K < p^n gefunden"
        return False

    K, p, e = decomposition
    test_data["Generalized Pocklington"][n]["other_fields"] = [K, p, e]

    for a in range(2, n):
        cond1 = pow(a, n - 1, n) == 1
        cond2 = gcd(pow(a, (n - 1) // p, n) - 1, n) == 1
        test_data["Generalized Pocklington"][n]["a_values"].append((a, cond1, cond2))

        if cond1 and cond2:
            test_data["Generalized Pocklington"][n]["result"] = True
            return True

    test_data["Generalized Pocklington"][n]["result"] = False
    test_data["Generalized Pocklington"][n]["reason"] = "Kein geeignetes a gefunden"
    return False


def grau_test_protocoll(n: int) -> bool: #6.13
    if n <= 1: raise ValueError("n must be greater than 1")

    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition:
        test_data["Grau"][n]["result"] = False
        test_data["Grau"][n]["reason"] = "Keine Zerlegung n=K*p^n+1 gefunden"
        return False

    K, p, n_exp = decomposition
    a = helpers.find_quadratic_non_residue(p)
    if a is None:
        test_data["Grau"][n]["result"] = False
        test_data["Grau"][n]["reason"] = f"Kein quadratischer Nichtrest für p={p} gefunden"
        return False

    exponent = (n - 1) // p
    base = pow(a, exponent, n)
    phi_p = helpers.cyclotomic_polynomial(p, base) % n
    is_prime = (phi_p == 0)
    test_data["Grau"][n]["a_values"] = [a]
    test_data["Grau"][n]["other_fields"] = [K, p, n_exp, phi_p]
    test_data["Grau"][n]["result"] = is_prime
    return is_prime


def grau_probability_test_protocoll(n: int) -> bool: #6.14
    if n <= 1:  raise ValueError("n must be greater than 1")

    decomposition = helpers.find_pocklington_decomposition(n)
    if not decomposition:
        test_data["Grau Probability"][n]["result"] = False
        test_data["Grau Probability"][n]["reason"] = "Keine Zerlegung N=K*p^n+1 gefunden"
        return False

    K, p, n_exp = decomposition
    test_data["Grau Probability"][n]["reason"] = None
    test_data["Grau Probability"][n]["other_fields"] = [K, p, n_exp]

    for a in range(2, n):
        for j in range(n - 1, -1, -1):
            try:
                log_term = math.log(K, p)
            except ValueError:
                log_term = float("inf")

            exp = K * (p ** (n - j - 1))
            base = pow(a, exp, n)
            phi_p = helpers.cyclotomic_polynomial(p, base) % n

            cond1 = (phi_p == 0)
            cond2 = (2 * (n - j) > log_term + n)

            if cond1 and cond2:
                test_data["Grau Probability"][n]["a_values"] = [a]
                test_data["Grau Probability"][n]["other_fields"].extend([j])
                test_data["Grau Probability"][n]["result"] = True
                return True

    test_data["Grau Probability"][n]["result"] = False
    test_data["Grau Probability"][n]["reason"] = "Kein geeignetes (a,j)-Paar gefunden"
    return False

#############################################################################################
def miller_selfridge_rabin_test_protocoll(n: int, k: int = 5) -> bool:
    if (n < 2) or (n % 2 == 0 and n > 2) or helpers.is_real_potency(n): raise ValueError("n must be an odd integer greater than 1 and not a real potency.")
    
    if n in (2, 3):
        test_data["Miller-Rabin"][n]["result"] = True
        return True

    # Zerlegung von n - 1 in 2^r * m
    m = n-1
    r = 0 
    while m % 2 == 0:
        m //= 2
        r += 1

    for _ in range(k):
        a = random.randint(2, n - 1)
        if gcd(a, n) != 1:
            test_data["Miller-Rabin"][n]["result"] = False
            return False

        cond1 = pow(a, n - 1, n) == 1
        test_data["Miller-Rabin"][n]["a_values"].append((a, cond1))
        if cond1:
            continue

        for i in range(r):
            if pow(a, 2**i * m, n) == n - 1: #cond2
                break
        else:
            test_data["Miller-Rabin"][n]["a_values"].append((a, False))
            test_data["Miller-Rabin"][n]["result"] = False
            return False

        test_data["Miller-Rabin"][n]["a_values"].append((a, True)) # cond2
    test_data["Miller-Rabin"][n]["result"] = True
    return True


def solovay_strassen_test_protocoll(n: int, k: int = 5) -> bool:
    if n < 2 or (n % 2 == 0 and n > 2): raise ValueError("n must be greater than 1")
    if n == 2:
        test_data["Solovay-Strassen"][n]["result"] = True
        return True
    
    for _ in range(k):
        a = random.randint(2, n - 1)
        jacobi = jacobi_symbol(a, n)
        cond1 = (jacobi == 0)
        test_data["Solovay-Strassen"][n]["a_values"].append((a, cond1))

        if cond1:
            test_data["Solovay-Strassen"][n]["result"] = False
            test_data["Solovay-Strassen"][n]["reason"] = "Jacobi-Symbol ist 0"
            return False
        
        cond2 = pow(a, (n - 1) // 2, n) != jacobi % n
        if cond2:
            test_data["Solovay-Strassen"][n]["a_values"].append((a, cond2))
            test_data["Solovay-Strassen"][n]["reason"] = "Kongruenzprüfung fehlgeschlagen"
            test_data["Solovay-Strassen"][n]["result"] = False
            return False

    test_data["Solovay-Strassen"][n]["result"] = True
    return True


def aks_test_protocoll(n: int) -> bool:
    if n <= 1 or helpers.is_real_potency(n):
        test_data["AKS"][n]["other_fields"]["initial_check"] = False
        test_data["AKS"][n]["result"] = False
        raise ValueError("n muss eine ungerade Zahl > 1 und keine echte Potenz sein")

    # Reset steps if test has to be run again
    test_data["AKS"][n]["other_fields"] = {
        "initial_check": True,
        "find_r": None,
        "prime_divisor_check": None,
        "polynomial_check": []
    }

    l = math.ceil(math.log2(n))
    r = 2
    while True:
        if gcd(n, r) == 1 and helpers.order(n, r) > l ** 2:
            test_data["AKS"][n]["other_fields"]["find_r"] = r
            break
        r += 1

    # Prüfe kleine Primteiler
    for p in primerange(2, l ** 5 + 1):
        if n % p == 0:
            if p == n:
                test_data["AKS"][n]["result"] = True
                test_data["AKS"][n]["other_fields"]["prime_divisor_check"] = f"Primfaktor {p} (n selbst)"
                return True
            else:
                test_data["AKS"][n]["result"] = False
                test_data["AKS"][n]["other_fields"]["prime_divisor_check"] = f"Teiler {p} von n"
                test_data["AKS"][n]["reason"] = f"n ist durch {p} teilbar"
                return False

    test_data["AKS"][n]["other_fields"]["prime_divisor_check"] = "Keine kleinen Teiler gefunden"

    # polynomial condition check
    max_a = math.floor(math.sqrt(r) * l)
    domain = ZZ

    for a in range(1, max_a + 1):
        mod_poly = Poly(X**r - 1, X, domain=domain)
        left = Poly((X + a) ** n, X, domain=domain).trunc(n).rem(mod_poly)
        right = Poly(X ** n + a, X, domain=domain).trunc(n).rem(mod_poly)

        test_passed = (left == right)
        test_data["AKS"][n]["other_fields"]["polynomial_check"].append((a, test_passed))

        if not test_passed:
            test_data["AKS"][n]["result"] = False
            return False

    test_data["AKS"][n]["result"] = True
    return True


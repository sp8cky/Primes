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
    print("Initialisiere Testdaten für alle Tests...")
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
    print(f"Fermat-Test für n={n} mit k={k} Wiederholungen")
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



#############################################################################################

"""
# Einheitliches Format für Zeitmessung
def format_timing(times: List[float]) -> str:
    return f"⏱ Time: {times[0]*1000:.2f}ms"

# Vereinheitlichte Ausgabe aller Tests
def print_test_protocoll(numbers: List[int], timings: Optional[Dict[str, List[Dict]]] = None, selected_tests: Optional[List[str]] = None):

    # Alle Testnamen aus test_data
    all_test_names = list(test_data.keys())

    # Wenn keine Auswahl angegeben, dann alle Tests
    if selected_tests is None:
        selected_tests = all_test_names
    else:
        selected_tests_lower = [name.lower() for name in selected_tests]
        name_map = {name.lower(): name for name in all_test_names}
        selected_tests = [name_map[name] for name in selected_tests_lower if name in name_map]

    def print_result_line(name: str, result: bool):
        print(f"{name}: {'✅ Prim' if result else '❌ Zusammengesetzt'}")

    def print_timing_line(name: str, n: int):
        if timings:
            times = [d["avg_time"] for d in timings.get(name, []) if d["n"] == n]
            if times:
                print("    ", format_timing(times))

    def print_test_detail(name: str, n: int, data: Dict):
        if name == "Fermat":
            print("    ", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in zip(data["a_values"], data["results"])))
        
        elif name in {"Initial Lucas", "Lucas"}:
            print(f"    a={data['a']}: Bedingung 1 {'✓' if data['condition1'] else '✗'}")
            if data.get("early_break"):
                print(f"    ⚠️ Abbruch bei m={data['early_break']}")
        
        elif name == "Optimized Lucas":
            for q, tests in data["tests"].items():
                print(f"    q={q}:", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in tests))
        
        elif name == "Pepin":
            if data.get("k") is not None:
                print(f"    F_{data['k']} = 2^(2^{data['k']}) + 1 = {n}")
                print(f"    {data.get('calculation', 'Keine Berechnung verfügbar')}")
                if data.get("result"):
                    print(f"    ≡ -1 mod {n} → Primzahl")
                else:
                    print(f"    ≢ -1 mod {n} → Zusammengesetzt")
            else:
                print(f"    {data.get('reason', 'Unbekannter Fehler')}")
        
        elif name == "Lucas-Lehmer":
            p = data.get("p")
            if p is not None:
                print(f"    M_{p} = 2^{p} - 1 = {n}")
                print(f"    Sequenz S_k: {', '.join(map(str, data.get('sequence', [])))}")
                final_S = data.get("final_S")
                if final_S is not None:
                    print(f"    Finales S_{p-2} = {final_S}")
                if data.get("result"):
                    print(f"    ≡ 0 mod {n} → Primzahl")
                else:
                    print(f"    ≢ 0 mod {n} → Zusammengesetzt")
            else:
                print(f"    {data.get('reason', 'Unbekannter Fehler')}")
        
        elif name == "Proth":
            if data.get("reason"):
                print(f"\n📌{data['reason']}")
        
        elif name == "Pocklington":
            if data.get("reason"):
                print(f"\n📌{data['reason']}")
        
        elif name == "Proth Variant":
            if data.get("a"):
                print(f"    a={data['a']}→✓")
            elif data.get("reason"):
                print(f"\n📌{data['reason']}")
        
        elif name == "Optimized Pocklington Variant":
            if data.get("tests"):
                for q, (a, res) in data["tests"].items():
                    print(f"    q={q}: a={a}→{'✓' if res else '✗'}")
            if data.get("b_test"):
                b, res = data["b_test"]
                print(f"    b={b}→{'✓' if res else '✗'}")
            if data.get("reason"):
                print(f"\n📌{data['reason']}")
        elif name == "Optimized Pocklington":
            for q, tests in data["tests"].items():
                print(f"    q={q}:", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in tests))

        elif name == "Generalized Pocklington":
            if data.get("K") is not None:
                print(f"    N = {data['K']}*{data['p']}^{data['n']} + 1")
                if data.get("a"):
                    print(f"    Found a = {data['a']} satisfying conditions")
                else:
                    print(f"    Attempted a values: {len(data['attempts'])}")
            print(f"    {data.get('reason', '')}")

        elif name == "Grau":
            if data.get("K") is not None:
                print(f"    N = {data['K']}*{data['p']}^{data['n']} + 1")
                print(f"    Quadratic non-residue a = {data.get('a', '?')}")
                exponent = f"(N-1)/{data['p']}" if data.get('p') else "?"
                print(f"    φ_{data['p']}(a^{exponent}) ≡ {data.get('phi_p', '?')} mod N")
            print(f"    {data.get('reason', '')}")

        elif name == "Grau Probability":
            if data.get("K") is not None:
                print(f"    N = {data['K']}*{data['p']}^{data['n']} + 1")
                if data.get("a") is not None:
                    print(f"    Found (a,j) = ({data['a']},{data['j']}) satisfying conditions")
                else:
                    print(f"    Attempted (a,j) pairs: {len(data['attempts'])}")
                print(f"    log_p(K) + n = {math.log(data['K'], data['p']) + data['n'] if data.get('K') else '?'}")
            print(f"    {data.get('reason', '')}")
        
        elif name == "Miller-Rabin":
            print("    ", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in data["repeats"]))
        
        elif name == "Solovay-Strassen":
            print("    ", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in data["repeats"]))
        
        elif name == "AKS":
            steps = data.get("steps", {})
            if "find_r" in steps:
                print(f"    r = {steps['find_r']}")
            if "prime_divisor_check" in steps:
                print(f"    Primteiler-Check: {steps['prime_divisor_check']}")
            if "polynomial_check" in steps:
                print("    Polynom-Tests:", " | ".join(f"a={a}→{'✓' if res else '✗'}"
                    for a, res in steps["polynomial_check"]))

    for n in numbers:
        print(f"\n\033[1mTesting n = {n}\033[0m")
        for name in selected_tests:
            if n not in test_data[name]:
                continue
            data = test_data[name][n]
            # Standard: wenn "result" nicht vorhanden, dann schauen, ob alle "results" True sind
            result = data.get("result", all(data.get("results", [])))
            print_result_line(name, result)
            print_test_detail(name, n, data)
            print_timing_line(name, n)

"""

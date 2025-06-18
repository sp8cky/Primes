import src.primality.helpers as helpers
import random, math
from math import gcd
from sympy import factorint
from statistics import mean
from sympy import jacobi_symbol, gcd, log, primerange
from sympy.abc import X
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from typing import Optional, List, Dict, Tuple


test_data = {}

def init_all_test_data(numbers: List[int]):
    global test_data
    test_data = {
        # Kriterien
        "Fermat": {n: {"a_values": [], "results": []} for n in numbers},
        "Wilson": {n: {"result": None} for n in numbers},
        "Initial Lucas": {n: {"a": None, "condition1": None, "early_break": None, "result": None} for n in numbers},
        "Lucas": {n: {"a": None, "condition1": None, "early_break": None, "result": None} for n in numbers},
        "Optimized Lucas": {n: {"factors": factorint(n-1), "tests": {}, "result": None} for n in numbers},
        "Pepin": {n: {"k": None, "result": None, "calculation": None, "reason": None} for n in numbers},
        "Lucas-Lehmer": {n: {"p": None, "sequence": [], "final_S": None, "result": None, "reason": None} for n in numbers},
        "Proth": {n: {"a_values": [], "results": [], "result": None, "reason": None} for n in numbers},
        "Pocklington": {n: {"a_values": [], "condition1": [], "condition2": [], "result": None, "reason": None} for n in numbers},
        "Optimized Pocklington": {n: {"tests": {}, "result": None, "reason": None} for n in numbers},
        "Proth Variant": {n: {"a": None, "result": None, "reason": None} for n in numbers},
        "Optimized Pocklington Variant": {n: {"tests": {}, "b_test": None, "result": None, "reason": None} for n in numbers},
        "Generalized Pocklington": {n: {"K": None, "p": None, "n": None, "a": None, "attempts": [], "result": None, "reason": None} for n in numbers},
        "Grau": {n: {"K": None, "p": None, "n": None, "a": None, "phi_p": None, "exponent": None, "result": None, "reason": None} for n in numbers},
        "Grau Probability": {n: {"K": None, "p": None, "n": None, "a": None, "j": None, "attempts": [], "result": None, "reason": None} for n in numbers},
        "Miller-Rabin": {n: {"repeats": [], "results": []} for n in numbers},
        "Solovay-Strassen": {n: {"repeats": [], "results": []} for n in numbers},
        "AKS": {n: {"steps": {
                        "initial_check": None,
                        "find_r": None,
                        "prime_divisor_check": None,
                        "polynomial_check": []
                    },
                    "result": None
                } for n in numbers
        }
    }

############################################################################################

def fermat_test(n: int, k: int = 1) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2: return True

    for _ in range(k):
        a = random.randint(2, n-1)
        gcd_ok = gcd(a, n) == 1
        test_ok = pow(a, n-1, n) == 1

        test_data["Fermat"][n]["a_values"].append(a)
        test_data["Fermat"][n]["results"].append(gcd_ok and test_ok)

        if not gcd_ok or not test_ok:
            return False
    return True


def wilson_criterion(p: int) -> bool:
    if p <= 1: raise ValueError("p must be greater than 1")
    result = math.factorial(p - 1) % p == p - 1
    test_data["Wilson"][p]["result"] = result
    return result


def initial_lucas_test(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2:
        test_data["Initial Lucas"][n]["result"] = True
        return True

    a = random.randint(2, n-2)
    condition1 = pow(a, n-1, n) == 1
    test_data["Initial Lucas"][n]["a"] = a
    test_data["Initial Lucas"][n]["condition1"] = condition1

    if not condition1:
        test_data["Initial Lucas"][n]["result"] = False
        return False

    for m in range(1, n-1):
        if pow(a, m, n) == 1:
            test_data["Initial Lucas"][n]["early_break"] = m
            test_data["Initial Lucas"][n]["result"] = False
            return False
    test_data["Initial Lucas"][n]["result"] = True
    return True


def lucas_test(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2:
        test_data["Lucas"][n]["result"] = True
        return True

    a = random.randint(2, n-1)
    condition1 = pow(a, n-1, n) == 1
    test_data["Lucas"][n]["a"] = a
    test_data["Lucas"][n]["condition1"] = condition1

    if not condition1:
        test_data["Lucas"][n]["result"] = False
        return False

    for m in range(1, n):
        if (n-1) % m == 0 and pow(a, m, n) == 1:
            test_data["Lucas"][n]["early_break"] = m
            test_data["Lucas"][n]["result"] = False
            return False
    test_data["Lucas"][n]["result"] = True
    return True


def optimized_lucas_test(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2:
        test_data["Optimized Lucas"][n]["result"] = True
        return True

    factors = test_data["Optimized Lucas"][n]["factors"]
    for q in factors:
        for a in range(2, n):
            condition1 = pow(a, n-1, n) == 1
            condition2 = pow(a, (n-1)//q, n) != 1

            if q not in test_data["Optimized Lucas"][n]["tests"]:
                test_data["Optimized Lucas"][n]["tests"][q] = []
            test_data["Optimized Lucas"][n]["tests"][q].append((a, condition1 and condition2))

            if condition1 and condition2:
                break
        else:
            test_data["Optimized Lucas"][n]["result"] = False
            return False
    test_data["Optimized Lucas"][n]["result"] = True
    return True

def pepin_test(n: int) -> bool: # 3.36
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Check if n is a Fermat number F_k = 2^(2^k) + 1
    k = None
    for k_candidate in range(1, 32):  # Praktischer Grenzwert
        F_k = 2**(2**k_candidate) + 1
        if F_k == n:
            k = k_candidate
            break
        elif F_k > n:
            break
    
    if k is None:
        test_data["Pepin"][n]["result"] = False
        test_data["Pepin"][n]["reason"] = "Keine Fermat-Zahl (n ≠ 2^(2^k)+1)"
        return False
    
    test_data["Pepin"][n]["k"] = k
    
    # Durchführung des Tests
    exponent = (n - 1) // 2
    result = pow(3, exponent, n)
    
    is_prime = (result == n - 1)  # ≡ -1 mod n
    test_data["Pepin"][n]["result"] = is_prime
    test_data["Pepin"][n]["calculation"] = f"3^({exponent}) ≡ {result} mod {n}"
    
    return is_prime

def lucas_lehmer_test(n: int) -> bool: #3.32
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Check if n is a Mersenne number M_p = 2^p - 1
    p = None
    for p_candidate in range(2, 32):  # Praktischer Grenzwert
        M_p = 2**p_candidate - 1
        if M_p == n:
            p = p_candidate
            break
        elif M_p > n:
            break
    
    if p is None:
        test_data["Lucas-Lehmer"][n]["result"] = False
        test_data["Lucas-Lehmer"][n]["reason"] = "Keine Mersenne-Zahl (n ≠ 2^p-1)"
        return False
    
    test_data["Lucas-Lehmer"][n]["p"] = p
    
    # Durchführung des Tests
    S = 4
    test_data["Lucas-Lehmer"][n]["sequence"] = [S]
    
    for _ in range(p - 2):
        S = (S**2 - 2) % n
        test_data["Lucas-Lehmer"][n]["sequence"].append(S)
    
    is_prime = (S == 0)
    test_data["Lucas-Lehmer"][n]["result"] = is_prime
    test_data["Lucas-Lehmer"][n]["final_S"] = S
    
    return is_prime


def proth_test(n: int) -> bool: # 4.5
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Check if n is of the form K*2^m + 1 with K < 2^m
    m = 0
    temp = n - 1
    while temp % 2 == 0:
        temp //= 2
        m += 1
    K = temp
    
    if K >= 2**m:
        test_data["Proth"][n]["result"] = False
        test_data["Proth"][n]["reason"] = "Erfüllt nicht K < 2^m"
        return False
    
    # perform the test
    for a in range(2, n):
        condition = pow(a, (n-1)//2, n) == n-1
        test_data["Proth"][n]["a_values"].append(a)
        test_data["Proth"][n]["results"].append(condition)
        
        if condition:
            test_data["Proth"][n]["result"] = True
            return True
    
    test_data["Proth"][n]["result"] = False
    return False


def pocklington_test(n: int) -> bool: # 4.6
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Factorize n-1 as q^m * R
    factors = factorint(n-1)
    if len(factors) != 1:
        test_data["Pocklington"][n]["result"] = False
        test_data["Pocklington"][n]["reason"] = "n-1 muss genau einen Primfaktor haben"
        return False
    
    q, m = next(iter(factors.items()))
    R = (n-1) // (q**m)
    
    if (n-1) % q != 0 or R % q == 0:
        test_data["Pocklington"][n]["result"] = False
        test_data["Pocklington"][n]["reason"] = "q muss n - 1 genau m mal teilen"
        return False
    
    # perform the test
    for a in range(2, n):
        condition1 = pow(a, n-1, n) == 1
        condition2 = gcd(pow(a, (n-1)//q, n) - 1, n) == 1
        
        test_data["Pocklington"][n]["a_values"].append(a)
        test_data["Pocklington"][n]["condition1"].append(condition1)
        test_data["Pocklington"][n]["condition2"].append(condition2)
        
        if condition1 and condition2:
            test_data["Pocklington"][n]["result"] = True
            return True
    
    test_data["Pocklington"][n]["result"] = False
    return False


def optimized_pocklington_test(n: int) -> bool: # 4.7
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Factorize n-1 as F*R with gcd(F,R)=1
    # We need to find a suitable F where we know all its prime factors
    # For simplicity, we'll take F as the largest square-free factor of n-1
    # whose prime factors we know (in practice, this would need to be provided)
    factors = factorint(n-1)
    F = 1
    for p in factors:
        F *= p
    
    R = (n-1) // F
    
    if gcd(F, R) != 1:
        test_data["Optimized Pocklington"][n]["result"] = False
        test_data["Optimized Pocklington"][n]["reason"] = "F und R müssen teilerfremd sein"
        return False
    
    # Now perform the test for each prime factor of F
    for q in factors:
        found = False
        for a in range(2, n):
            condition1 = pow(a, n-1, n) == 1
            condition2 = gcd(pow(a, (n-1)//q, n) - 1, n) == 1
            
            if q not in test_data["Optimized Pocklington"][n]["tests"]:
                test_data["Optimized Pocklington"][n]["tests"][q] = []
            test_data["Optimized Pocklington"][n]["tests"][q].append((a, condition1 and condition2))
            
            if condition1 and condition2:
                found = True
                break
        
        if not found:
            test_data["Optimized Pocklington"][n]["result"] = False
            return False
    
    test_data["Optimized Pocklington"][n]["result"] = True
    return True


def proth_test_variant(n: int) -> bool: # 4.8
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Check if n is of form 2^m * h + 1 with h odd
    if n % 2 == 0:
        test_data["Proth Variant"][n]["result"] = False
        test_data["Proth Variant"][n]["reason"] = "n must be odd"
        return False
    
    m = 0
    h = n - 1
    while h % 2 == 0:
        h //= 2
        m += 1
    
    # Check a^(n-1) ≡ 1 mod n first
    for a in range(2, n):
        if pow(a, n-1, n) != 1:
            test_data["Proth Variant"][n]["result"] = False
            test_data["Proth Variant"][n]["reason"] = f"a={a} fails a^(n-1) ≡ 1 mod n"
            return False
        
        # Now check a^((n-1)/2) ≡ -1 mod n
        if pow(a, (n-1)//2, n) == n-1:
            test_data["Proth Variant"][n]["result"] = True
            test_data["Proth Variant"][n]["a"] = a
            return True
    
    test_data["Proth Variant"][n]["result"] = False
    test_data["Proth Variant"][n]["reason"] = "Kein passendes a gefunden"
    return False


def optimized_pocklington_test_variant(n: int, B: Optional[int] = None) -> bool: # 4.9
    if n <= 1: raise ValueError("n must be greater than 1")
    
    # Factorize n-1 as F*R with gcd(F,R)=1
    factors = factorint(n-1)
    F = 1
    for p in factors:
        F *= p**factors[p]
    R = (n-1) // F
    
    # Automatic B selection if not provided
    if B is None:
        B = int(math.isqrt(n) // F) + 1
    
    # Check FB > sqrt(n) and R has no prime factors < B
    if F * B <= math.isqrt(n):
        test_data["Optimized Pocklington Variant"][n]["result"] = False
        test_data["Optimized Pocklington Variant"][n]["reason"] = "FB ≤ √n condition not met"
        return False
    
    # Check R has no small prime factors
    for p in primerange(2, B):
        if R % p == 0:
            test_data["Optimized Pocklington Variant"][n]["result"] = False
            test_data["Optimized Pocklington Variant"][n]["reason"] = f"R has prime factor {p} < B"
            return False
    
    # Condition (i): For each prime factor q of F
    for q in factors:
        found = False
        for a in range(2, n):
            condition1 = pow(a, n-1, n) == 1
            condition2 = gcd(pow(a, (n-1)//q, n) - 1, n) == 1
            
            test_data["Optimized Pocklington Variant"][n]["tests"][q] = (a, condition1 and condition2)
            
            if condition1 and condition2:
                found = True
                break
        
        if not found:
            test_data["Optimized Pocklington Variant"][n]["result"] = False
            test_data["Optimized Pocklington Variant"][n]["reason"] = f"Kein passendes a gefunden für q={q}"
            return False
    
    # Condition (ii): Find b satisfying special condition
    for b in range(2, n):
        condition1 = pow(b, n-1, n) == 1
        condition2 = gcd(pow(b, F, n) - 1, n) == 1
        
        test_data["Optimized Pocklington Variant"][n]["b_test"] = (b, condition1 and condition2)
        
        if condition1 and condition2:
            test_data["Optimized Pocklington Variant"][n]["result"] = True
            return True
    
    test_data["Optimized Pocklington Variant"][n]["result"] = False
    test_data["Optimized Pocklington Variant"][n]["reason"] = "Kein passendes b gefunden"
    return False


def generalized_pocklington_test(N: int) -> bool: # 6.12
    if N <= 1: raise ValueError("N must be greater than 1")
    
    # Find decomposition N = K*p^n + 1
    decomposition = helpers.find_pocklington_decomposition(N)
    if not decomposition:
        test_data["Generalized Pocklington"][N]["result"] = False
        test_data["Generalized Pocklington"][N]["reason"] = "No valid decomposition N=K*p^n+1 found"
        return False
    
    K, p, n = decomposition
    test_data["Generalized Pocklington"][N]["K"] = K
    test_data["Generalized Pocklington"][N]["p"] = p
    test_data["Generalized Pocklington"][N]["n"] = n
    
    # Find suitable a
    for a in range(2, N):
        condition1 = pow(a, N-1, N) == 1
        exponent = (N-1) // p
        condition2 = gcd(pow(a, exponent, N) - 1, N) == 1
        
        test_data["Generalized Pocklington"][N]["attempts"].append((a, condition1, condition2))
        
        if condition1 and condition2:
            test_data["Generalized Pocklington"][N]["a"] = a
            test_data["Generalized Pocklington"][N]["result"] = True
            return True
    
    test_data["Generalized Pocklington"][N]["result"] = False
    test_data["Generalized Pocklington"][N]["reason"] = "No suitable a found"
    return False


def grau_test(N: int) -> bool: # 6.13
    if N <= 1: raise ValueError("N must be greater than 1")
    
    decomposition = helpers.find_pocklington_decomposition(N)
    if not decomposition:
        test_data["Grau"][N]["result"] = False
        test_data["Grau"][N]["reason"] = "No valid decomposition N=K*p^n+1 found"
        return False
    
    K, p, n = decomposition
    test_data["Grau"][N]["K"] = K
    test_data["Grau"][N]["p"] = p
    test_data["Grau"][N]["n"] = n
    
    # Find quadratic non-residue modulo p
    a = helpers.find_quadratic_non_residue(p)
    if a is None:
        test_data["Grau"][N]["result"] = False
        test_data["Grau"][N]["reason"] = f"No quadratic non-residue found for p={p}"
        return False
    
    test_data["Grau"][N]["a"] = a
    
    # Compute cyclotomic polynomial condition
    exponent = (N-1) // p
    value = pow(a, exponent, N)
    phi_p = helpers.cyclotomic_polynomial(p, value) % N
    
    test_data["Grau"][N]["phi_p"] = phi_p
    test_data["Grau"][N]["exponent"] = exponent
    
    is_prime = (phi_p == 0)
    test_data["Grau"][N]["result"] = is_prime
    return is_prime


def grau_probability_test(N: int) -> bool: # 6.14
    if N <= 1: raise ValueError("N must be greater than 1")
    
    decomposition = helpers.find_pocklington_decomposition(N)
    if not decomposition:
        test_data["Grau Probability"][N]["result"] = False
        test_data["Grau Probability"][N]["reason"] = "No valid decomposition N=K*p^n+1 found"
        return False
    
    K, p, n = decomposition
    test_data["Grau Probability"][N]["K"] = K
    test_data["Grau Probability"][N]["p"] = p
    test_data["Grau Probability"][N]["n"] = n
    
    # Find suitable a and j
    for a in range(2, N):
        for j in range(n-1, -1, -1):
            exponent = K * (p**(n-j-1))
            value = pow(a, exponent, N)
            phi_p = helpers.cyclotomic_polynomial(p, value) % N
            
            condition1 = (phi_p == 0)
            condition2 = (2*(n - j) > math.log(K, p) + n)
            
            test_data["Grau Probability"][N]["attempts"].append((a, j, condition1, condition2))
            
            if condition1 and condition2:
                test_data["Grau Probability"][N]["a"] = a
                test_data["Grau Probability"][N]["j"] = j
                test_data["Grau Probability"][N]["result"] = True
                return True
    
    test_data["Grau Probability"][N]["result"] = False
    test_data["Grau Probability"][N]["reason"] = "No suitable (a,j) pair found"
    return False


############################################################################################

def miller_selfridge_rabin_test(n: int, k=5) -> bool:
    if (n < 2) or (n % 2 == 0 and n > 2) or helpers.is_real_potency(n):
        raise ValueError("n must be an odd integer greater than 1 and not a real potency.")
    
    # Zerlegung von n - 1 in 2^r * m
    m = n - 1
    r = 0
    while m % 2 == 0:
        m //= 2
        r += 1

    for round_num in range(k):
        a = random.randint(1, n - 1)
        if gcd(a, n) != 1:
            test_data["Miller-Rabin"][n]["repeats"].append((a, False))
            test_data["Miller-Rabin"][n]["results"].append(False)
            return False
        
        if pow(a, m, n) == 1:
            test_data["Miller-Rabin"][n]["repeats"].append((a, True))
            continue
        
        for i in range(r):
            if pow(a, 2**i * m, n) % n == n - 1:
                break
        else:
            test_data["Miller-Rabin"][n]["repeats"].append((a, False))
            test_data["Miller-Rabin"][n]["results"].append(False)
            return False

        test_data["Miller-Rabin"][n]["repeats"].append((a, True))

    test_data["Miller-Rabin"][n]["results"].append(True)
    return True


def solovay_strassen_test(n: int, k=5) -> bool:
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if n < 2 or (n % 2 == 0 and n > 2):
        raise ValueError("n must be odd and > 1.")
    if n == 2:
        test_data["Solovay-Strassen"][n]["results"].append(True)
        return True

    for round_num in range(k):
        a = random.randint(2, n - 1)
        jacobi = jacobi_symbol(a, n)
        condition = (jacobi == 0) or (pow(a, (n - 1) // 2, n) != jacobi % n)

        test_data["Solovay-Strassen"][n]["repeats"].append((a, not condition))

        if condition:
            test_data["Solovay-Strassen"][n]["results"].append(False)
            return False

    test_data["Solovay-Strassen"][n]["results"].append(True)
    return True


def aks_test(n: int) -> bool:
    if (n <= 1) or helpers.is_real_potency(n):
        test_data["AKS"][n]["steps"]["initial_check"] = False
        test_data["AKS"][n]["result"] = False
        raise ValueError("n must be an odd integer greater than 1 and not a real potency.")

    # Reset steps (falls der Test wiederholt wird)
    test_data["AKS"][n]["steps"] = {
        "initial_check": True,
        "find_r": None,
        "prime_divisor_check": None,
        "polynomial_check": []
    }

    l = math.ceil(log(n, 2))

    # find lowest r with ord_r(n) > l^2
    r = 2
    while True:
        if (gcd(n, r) == 1) and helpers.order(n, r) > l**2:
            test_data["AKS"][n]["steps"]["find_r"] = r
            break
        r += 1

    # check for prime divisors
    for p in primerange(2, l**5 + 1):
        if n % p == 0:
            if p == n:
                test_data["AKS"][n]["result"] = True
                test_data["AKS"][n]["steps"]["prime_divisor_check"] = f"Prime {p}"
                return True
            else:
                test_data["AKS"][n]["result"] = False
                test_data["AKS"][n]["steps"]["prime_divisor_check"] = f"Divisor {p}"
                return False
    
    test_data["AKS"][n]["steps"]["prime_divisor_check"] = "No small divisors found"

    # polynomial condition check
    max_a = math.floor(math.sqrt(r) * l)
    domain = ZZ
    for a in range(1, max_a + 1):
        mod_poly = Poly(X**r - 1, X, domain=domain)
        left = Poly((X + a)**n, X, domain=domain).trunc(n).rem(mod_poly)
        right = Poly(X**n + a, X, domain=domain).trunc(n).rem(mod_poly)
        
        test_passed = (left == right)
        test_data["AKS"][n]["steps"]["polynomial_check"].append((a, test_passed))
        
        if not test_passed:
            test_data["AKS"][n]["result"] = False
            return False
    
    test_data["AKS"][n]["result"] = True
    return True


#############################################################################################


# Einheitliches Format für Zeitmessung
def format_timing(times: List[float]) -> str:
    return f"⏱ Time: {times[0]*1000:.2f}ms"

# Vereinheitlichte Ausgabe aller Tests
def test_protocoll(numbers: List[int], timings: Optional[Dict[str, List[Dict]]] = None, selected_tests: Optional[List[str]] = None):

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
                print(f"    φ_{data['p']}(a^{(n-1)/p}) ≡ {data.get('phi_p', '?')} mod N")
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
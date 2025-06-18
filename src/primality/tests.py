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

        # Tests
        "Miller-Rabin": {n: {"rounds": [], "results": []} for n in numbers},
        "Solovay-Strassen": {n: {"rounds": [], "results": []} for n in numbers},
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

############################################################################################


def miller_selfridge_rabin_test(n: int, rounds=5) -> bool:
    if (n < 2) or (n % 2 == 0 and n > 2) or helpers.is_real_potency(n):
        raise ValueError("n must be an odd integer greater than 1 and not a real potency.")
    
    # Form von n-1 zerlegen
    m = n-1
    k = 0
    while m % 2 == 0:
        m //= 2
        k += 1
    
    # Iterationen (Runden)
    for round_num in range(rounds):
        a = random.randint(1, n - 1)
        gcd_ok = gcd(a, n) == 1
        test_ok = True
        
        if not gcd_ok:
            test_data["Miller-Rabin"][n]["rounds"].append((a, False))
            test_data["Miller-Rabin"][n]["results"].append(False)
            return False
        
        if pow(a, m, n) == 1:
            test_data["Miller-Rabin"][n]["rounds"].append((a, True))
            continue
        
        for i in range(k):
            if pow(a, 2**i * m, n) % n == n-1:
                break
        else:
            test_data["Miller-Rabin"][n]["rounds"].append((a, False))
            test_data["Miller-Rabin"][n]["results"].append(False)
            return False
        
        test_data["Miller-Rabin"][n]["rounds"].append((a, True))
    
    test_data["Miller-Rabin"][n]["results"].append(True)
    return True

def solovay_strassen_test(n: int, rounds=5) -> bool:
    if rounds <= 0: raise ValueError("Rounds must be a positive integer.")
    if n < 2 or (n % 2 == 0 and n > 2):
        raise ValueError("n must be odd and > 1.")
    if n == 2:
        test_data["Solovay-Strassen"][n]["results"].append(True)
        return True

    for round_num in range(rounds):
        a = random.randint(2, n - 1)
        jacobi = jacobi_symbol(a, n)
        condition = (jacobi == 0) or (pow(a, (n-1)//2, n) != jacobi % n)
        
        test_data["Solovay-Strassen"][n]["rounds"].append((a, not condition))
        
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
            print("   ", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in zip(data["a_values"], data["results"])))
        elif name in {"Initial Lucas", "Lucas"}:
            print(f"   a={data['a']}: Bedingung 1 {'✓' if data['condition1'] else '✗'}")
            if data.get("early_break"):
                print(f"   ⚠️ Abbruch bei m={data['early_break']}")
        elif name == "Optimized Lucas":
            for q, tests in data["tests"].items():
                print(f"   q={q}:", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in tests))
        elif name == "Miller-Rabin":
            print("   ", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in data["rounds"]))
        elif name == "Solovay-Strassen":
            print("   ", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in data["rounds"]))
        elif name == "AKS":
            steps = data.get("steps", {})
            if "find_r" in steps:
                print(f"   r = {steps['find_r']}")
            if "prime_divisor_check" in steps:
                print(f"   Primteiler-Check: {steps['prime_divisor_check']}")
            if "polynomial_check" in steps:
                print("   Polynom-Tests:", " | ".join(f"a={a}→{'✓' if res else '✗'}"
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
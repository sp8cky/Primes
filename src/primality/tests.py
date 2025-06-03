import src.primality.helpers as helpers
import random, math
from math import gcd
from statistics import mean
from sympy import jacobi_symbol, gcd, log, primerange
from sympy.abc import X
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from typing import Optional, List, Dict, Tuple

tests_data = {}

def init_tests_data(numbers: List[int]):
    global tests_data
    tests_data = {
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
            tests_data["Miller-Rabin"][n]["rounds"].append((a, False))
            tests_data["Miller-Rabin"][n]["results"].append(False)
            return False
        
        if pow(a, m, n) == 1:
            tests_data["Miller-Rabin"][n]["rounds"].append((a, True))
            continue
        
        for i in range(k):
            if pow(a, 2**i * m, n) % n == n-1:
                break
        else:
            tests_data["Miller-Rabin"][n]["rounds"].append((a, False))
            tests_data["Miller-Rabin"][n]["results"].append(False)
            return False
        
        tests_data["Miller-Rabin"][n]["rounds"].append((a, True))
    
    tests_data["Miller-Rabin"][n]["results"].append(True)
    return True

def solovay_strassen_test(n: int, rounds=5) -> bool:
    if rounds <= 0: raise ValueError("Rounds must be a positive integer.")
    if n < 2 or (n % 2 == 0 and n > 2):
        raise ValueError("n must be odd and > 1.")
    if n == 2:
        tests_data["Solovay-Strassen"][n]["results"].append(True)
        return True

    for round_num in range(rounds):
        a = random.randint(2, n - 1)
        jacobi = jacobi_symbol(a, n)
        condition = (jacobi == 0) or (pow(a, (n-1)//2, n) != jacobi % n)
        
        tests_data["Solovay-Strassen"][n]["rounds"].append((a, not condition))
        
        if condition:
            tests_data["Solovay-Strassen"][n]["results"].append(False)
            return False
    
    tests_data["Solovay-Strassen"][n]["results"].append(True)
    return True

def aks_test(n: int) -> bool:
    if (n <= 1) or helpers.is_real_potency(n):
        tests_data["AKS"][n]["steps"]["initial_check"] = False
        tests_data["AKS"][n]["result"] = False
        raise ValueError("n must be an odd integer greater than 1 and not a real potency.")

    # Reset steps (falls der Test wiederholt wird)
    tests_data["AKS"][n]["steps"] = {
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
            tests_data["AKS"][n]["steps"]["find_r"] = r
            break
        r += 1

    # check for prime divisors
    for p in primerange(2, l**5 + 1):
        if n % p == 0:
            if p == n:
                tests_data["AKS"][n]["result"] = True
                tests_data["AKS"][n]["steps"]["prime_divisor_check"] = f"Prime {p}"
                return True
            else:
                tests_data["AKS"][n]["result"] = False
                tests_data["AKS"][n]["steps"]["prime_divisor_check"] = f"Divisor {p}"
                return False
    
    tests_data["AKS"][n]["steps"]["prime_divisor_check"] = "No small divisors found"

    # polynomial condition check
    max_a = math.floor(math.sqrt(r) * l)
    domain = ZZ
    for a in range(1, max_a + 1):
        mod_poly = Poly(X**r - 1, X, domain=domain)
        left = Poly((X + a)**n, X, domain=domain).trunc(n).rem(mod_poly)
        right = Poly(X**n + a, X, domain=domain).trunc(n).rem(mod_poly)
        
        test_passed = (left == right)
        tests_data["AKS"][n]["steps"]["polynomial_check"].append((a, test_passed))
        
        if not test_passed:
            tests_data["AKS"][n]["result"] = False
            return False
    
    tests_data["AKS"][n]["result"] = True
    return True


def format_timing(times: List[float]) -> str:
    return f"⏱ Best: {min(times)*1000:.2f}ms | Avg: {mean(times)*1000:.2f}ms | Worst: {max(times)*1000:.2f}ms"

def tests_protocoll(numbers: List[int], selected_tests: str = 'msa', timings: Optional[Dict[str, List[Dict]]] = None):
    if timings is None:
        timings = {}
    
    selected_tests = selected_tests.lower()
    
    for n in numbers:
        print(f"\n\033[1mTeste n = {n}\033[0m")
        
        # Miller-Rabin Test
        if 'm' in selected_tests and 'Miller-Rabin' in tests_data and n in tests_data['Miller-Rabin']:
            data = tests_data["Miller-Rabin"][n]
            print(f"Miller-Rabin: {'✅ Prim' if all(data['results']) else '❌ Zusammengesetzt'}")
            print("   ", " | ".join(f"a={a}→{'✓' if res else '✗'}" 
                 for a, res in data["rounds"]))
            if 'Miller-Rabin' in timings:
                times = [d["avg_time"] for d in timings["Miller-Rabin"] if d["n"] == n]
                if times: print("    ", format_timing(times))

        # Solovay-Strassen Test
        if 's' in selected_tests and 'Solovay-Strassen' in tests_data and n in tests_data['Solovay-Strassen']:
            data = tests_data["Solovay-Strassen"][n]
            print(f"Solovay-Strassen: {'✅ Prim' if all(data['results']) else '❌ Zusammengesetzt'}")
            print("   ", " | ".join(f"a={a}→{'✓' if res else '✗'}" 
                 for a, res in data["rounds"]))
            if 'Solovay-Strassen' in timings:
                times = [d["avg_time"] for d in timings["Solovay-Strassen"] if d["n"] == n]
                if times: print("    ", format_timing(times))

        # AKS Test
        if 'a' in selected_tests and 'AKS' in tests_data and n in tests_data['AKS']:
            data = tests_data["AKS"][n]
            print(f"AKS: {'✅ Prim' if data['result'] else '❌ Zusammengesetzt'}")
            if 'steps' in data and 'find_r' in data['steps']:
                print(f"   r = {data['steps']['find_r']}")
            if 'steps' in data and 'prime_divisor_check' in data['steps']:
                print(f"   Primteiler-Check: {data['steps']['prime_divisor_check']}")
            if 'steps' in data and 'polynomial_check' in data['steps']:
                print("   Polynom-Tests:", " | ".join(f"a={a}→{'✓' if res else '✗'}" 
                     for a, res in data["steps"]["polynomial_check"]))
            if 'AKS' in timings:
                times = [d["avg_time"] for d in timings["AKS"] if d["n"] == n]
                if times: print("    ", format_timing(times))
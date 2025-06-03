import src.primality.helpers as helpers
import random, math
from math import gcd
from statistics import mean
from sympy import jacobi_symbol, gcd, log, primerange
from sympy.abc import X
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from typing import Optional, List, Dict, Tuple

# same tests as in src.primality.tests.py, but with detailed output for each step

def miller_selfridge_rabin_test_detail(n: int, rounds=5) -> Tuple[bool, List[Tuple[str, bool]]]:
    details = []
    if n < 2 or (n % 2 == 0 and n > 2) or helpers.is_real_potency(n):
        raise ValueError("Ungültige Eingabe für Miller-Rabin: n muss ungerade >1 und keine echte Potenz sein.")
    
    # Zerlegung n-1 = 2^k * m
    m = n - 1
    k = 0
    while m % 2 == 0:
        m //= 2
        k += 1
    details.append((f"Zerlegung: {n-1} = 2^{k} * {m}", True))

    for i in range(rounds):
        a = random.randint(2, n - 2)
        details.append((f"Runde {i+1}: a = {a}", True))
        
        # Check gcd(a, n)
        gcd_val = gcd(a, n)
        if gcd_val != 1:
            details.append((f"gcd({a}, {n}) = {gcd_val} ≠ 1", False))
            return (False, details)
        
        x = pow(a, m, n)
        details.append((f"a^{m} mod {n} = {x}", x == 1 or x == n - 1))
        
        if x == 1 or x == n - 1:
            continue
            
        for j in range(k - 1):
            x = pow(x, 2, n)
            details.append((f"Quadrierung {j+1}: {x}", x == n - 1))
            if x == n - 1:
                break
        else:
            return (False, details)
    
    return (True, details)

def solovay_strassen_test_detail(n: int, rounds=5) -> Tuple[bool, List[Tuple[str, bool]]]:
    details = []
    if n < 2 or (n % 2 == 0 and n > 2):
        raise ValueError("n muss ungerade und >1 sein.")
    
    for i in range(rounds):
        a = random.randint(2, n - 1)
        details.append((f"Runde {i+1}: a = {a}", True))
        
        jacobi = jacobi_symbol(a, n)
        mod = pow(a, (n - 1) // 2, n)
        condition = mod == jacobi % n
        
        details.append((f"Jacobi-Symbol: ({a}/{n}) = {jacobi}", True))
        details.append((f"Bedingung: {a}^(({n}-1)/2) ≡ {mod} ≡ {jacobi} mod {n}", condition))
        
        if jacobi == 0 or not condition:
            return (False, details)
    
    return (True, details)

def aks_test_detail(n: int) -> Tuple[bool, List[Tuple[str, bool]]]:
    details = []
    if n <= 1 or helpers.is_real_potency(n):
        raise ValueError("n muss >1 sein und keine echte Potenz.")

    # Schritt 1: Log-Bedingung
    l = math.ceil(log(n, 2))
    details.append((f"Log-Bedingung: log₂({n}) = {l:.2f}", True))

    # Schritt 2: Finde kleinstes r mit ord_r(n) > l²
    r = 1
    while True:
        r += 1
        if gcd(n, r) == 1:
            ord_r = helpers.order(n, r)
            details.append((f"Teste r={r}: ord_{r}({n}) = {ord_r}", ord_r > l**2))
            if ord_r > l**2:
                break

    # Schritt 3: Prüfe kleine Teiler
    for p in primerange(2, min(n, l + 1)):
        if n % p == 0:
            details.append((f"Teiler gefunden: {p} | {n}", p == n))
            return (p == n, details)

    # Schritt 4: Polynomprüfung
    max_a = math.floor(math.sqrt(r) * l)
    details.append((f"Polynomtest für a ∈ 1..{max_a}", True))
    
    for a in range(1, max_a + 1):
        left = Poly((X + a)**n, X).trunc(n).rem(Poly(X**r - 1, X))
        right = Poly(X**n + a, X).trunc(n).rem(Poly(X**r - 1, X))
        condition = (left == right)
        details.append((f"a={a}: (X+{a})^{n} ≡ X^{n}+{a} mod (X^{r}-1)?", condition))
        if not condition:
            return (False, details)

    return (True, details)


# Utility function to format timing results
def format_timing(times: List[float]) -> str:
    return f"⏱ Best: {min(times)*1000:.2f}ms | Avg: {mean(times)*1000:.2f}ms | Worst: {max(times)*1000:.2f}ms"


# function for running tests and logging results
def tests_protocoll(numbers: List[int], tests_to_run: str = "msa", timings: Optional[Dict[str, List[Dict]]] = None):
    test_name_mapping = {
        'm': "Miller–Rabin",
        's': "Solovay–Strassen",
        'a': "AKS"
    }
    
    for n in numbers:
        print(f"\n\033[1mTeste n = {n}\033[0m")

        for test_code in tests_to_run.lower():
            if test_code == 'm':
                # Miller-Rabin Test
                try:
                    result, details = miller_selfridge_rabin_test_detail(n, 3)
                    print(f"Miller-Rabin: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
                
                    round_data = {0: []}
                    current_round = 0
                    for desc, cond in details:
                        if desc.startswith("Runde"):
                            current_round = int(desc.split()[1][:-1])
                            round_data[current_round] = []
                        round_data[current_round].append((desc, cond))
                    
                    for round_num in sorted(round_data.keys()):
                        items = round_data[round_num]
                        if round_num == 0:
                            print(f"  {items[0][0]}: {'✓' if items[0][1] else '✗'}")
                        else:
                            parts = []
                            for desc, cond in items:
                                if "Runde" in desc:
                                    parts.append(desc)
                                else:
                                    parts.append(f"{desc}→{'✓' if cond else '✗'}")
                            print("  ", " | ".join(parts))
                    
                    if timings:
                            timing_key = test_name_mapping[test_code]
                            times = [d["avg_time"] for d in timings.get(timing_key, []) if d["n"] == n]
                            if times:
                                print("   ", format_timing(times))
                            
                except ValueError as e:
                    print(f"Miller-Rabin: ⚠️ {str(e)}")

            # Solovay-Strassen Test
            elif test_code == 's':
                    # Solovay-Strassen Test
                    try:
                        result, details = solovay_strassen_test_detail(n, 3)
                        print(f"Solovay-Strassen: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
                    
                        round_data = {}
                        current_round = 0
                        for desc, cond in details:
                            if desc.startswith("Runde"):
                                current_round = int(desc.split()[1][:-1])
                                round_data[current_round] = []
                            round_data[current_round].append((desc, cond))
                        
                        for round_num in sorted(round_data.keys()):
                            parts = []
                            for desc, cond in round_data[round_num]:
                                if "Runde" in desc:
                                    parts.append(desc)
                                else:
                                    parts.append(f"{desc}→{'✓' if cond else '✗'}")
                            print("  ", " | ".join(parts))
                        
                        if timings:
                            timing_key = test_name_mapping[test_code]
                            times = [d["avg_time"] for d in timings.get(timing_key, []) if d["n"] == n]
                            if times:
                                print("   ", format_timing(times))
                                
                    except ValueError as e:
                        print(f"Solovay-Strassen: ⚠️ {str(e)}")
                            


            # AKS Test
            elif test_code == 'a':
                # AKS Test
                try:
                    result, details = aks_test_detail(n)
                    print(f"AKS: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
                
                    current_group = []
                    for desc, cond in details:
                        if "Teste r=" in desc or "a=" in desc:
                            current_group.append(f"{desc}→{'✓' if cond else '✗'}")
                        else:
                            if current_group:
                                print("  ", " | ".join(current_group))
                                current_group = []
                            print(f"  {desc}: {'✓' if cond else '✗'}")
                    
                    if current_group:
                        print("  ", " | ".join(current_group))
                    
                    if timings:
                        timing_key = test_name_mapping[test_code]
                        times = [d["avg_time"] for d in timings.get(timing_key, []) if d["n"] == n]
                        if times:
                            print("   ", format_timing(times))
                            
                except ValueError as e:
                    print(f"AKS: ⚠️ {str(e)}")


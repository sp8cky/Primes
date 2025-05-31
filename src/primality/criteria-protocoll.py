import src.primality.helpers as helpers
import random
import math
from math import gcd
from sympy import factorint
from typing import List, Dict, Tuple

# Erweiterte Testfunktionen mit Debug-Ausgabe
def fermat_criterion(n: int, k: int = 1) -> Tuple[bool, List[Tuple[int, bool]]]:
    details = []
    if n <= 1: return (False, details)
    if n == 2: return (True, details)
    
    for _ in range(k):
        a = random.randint(2, n-1)
        gcd_val = gcd(a, n)
        test_result = pow(a, n-1, n) == 1
        details.append((a, test_result and gcd_val == 1))
        if gcd_val != 1: return (False, details)
        if not test_result: return (False, details)
    return (True, details)

def wilson_criterion(n: int) -> Tuple[bool, None]:
    if n <= 1: return (False, None)
    result = (math.factorial(n-1) % n) == n-1
    return (result, None)  # Wilson verwendet kein a

def initial_lucas_test(n: int) -> Tuple[bool, List[Tuple[int, bool]]]:
    details = []
    if n <= 1: return (False, details)
    if n == 2: return (True, details)
    
    a = random.randint(2, n-1)
    condition1 = pow(a, n-1, n) == 1
    details.append((a, condition1))
    if not condition1: return (False, details)
    
    for m in range(1, n-1):
        if pow(a, m, n) == 1:
            details.append((f"m={m}", False))
            return (False, details)
    return (True, details)

def lucas_test(n: int) -> Tuple[bool, List[Tuple[str, bool]]]:
    details = []
    if n <= 1: return (False, details)
    if n == 2: return (True, details)
    
    a = random.randint(2, n-1)
    condition1 = pow(a, n-1, n) == 1
    details.append((a, condition1))
    if not condition1: return (False, details)
    
    for m in range(1, n):
        if (n-1) % m == 0 and pow(a, m, n) == 1:
            details.append((f"m={m}", False))
            return (False, details)
    return (True, details)

def optimized_lucas_test(n: int) -> Tuple[bool, Dict[int, List[Tuple[int, bool]]]]:
    details = {}
    if n <= 1: return (False, details)
    if n == 2: return (True, details)
    
    factors = factorint(n-1)
    for q in factors:
        details[q] = []
        for a in range(2, min(n, 100)):
            condition1 = pow(a, n-1, n) == 1
            condition2 = pow(a, (n-1)//q, n) != 1
            details[q].append((a, condition1 and condition2))
            if condition1 and condition2:
                break
        else:
            return (False, details)
    return (True, details)

# Testausführung mit Protokollierung
def run_tests(numbers: List[int]):
    for n in numbers:
        print(f"\n\033[1mTeste n = {n}\033[0m")
        
        # Fermat
        result, details = fermat_criterion(n, k=3)
        print(f"Fermat: {'Prim' if result else 'Zusammengesetzt'}")
        for i, (a, res) in enumerate(details, 1):
            print(f"  Iteration {i}: a={a} → {'Bestanden' if res else 'Durchgefallen'}")
        
        # Wilson
        result, _ = wilson_criterion(n)
        print(f"Wilson: {'Prim' if result else 'Zusammengesetzt'} (kein a)")
        
        # Initial Lucas
        result, details = initial_lucas_test(n)
        print(f"Initial Lucas: {'Prim' if result else 'Zusammengesetzt'}")
        a, res = details[0]
        print(f"  a={a}: Bedingung 1 {'OK' if res else 'Fehler'}")
        if len(details) > 1: print(f"  Abbruch bei {details[1][0]}")
        
        # Lucas
        result, details = lucas_test(n)
        print(f"Lucas: {'Prim' if result else 'Zusammengesetzt'}")
        a, res = details[0]
        print(f"  a={a}: Bedingung 1 {'OK' if res else 'Fehler'}")
        if len(details) > 1: print(f"  Abbruch bei {details[1][0]}")
        
        # Optimierter Lucas
        result, details = optimized_lucas_test(n)
        print(f"Optimierter Lucas: {'Prim' if result else 'Zusammengesetzt'}")
        for q, tests in details.items():
            print(f"  Faktor q={q}:")
            for a, res in tests:
                print(f"    a={a}: {'OK' if res else 'Fehler'}")

# Testzahlen (6-7 stellige Primzahlen)
test_numbers = [100003, 100019, 100043, 100103, 100151]

# Ausführung
run_tests(test_numbers)
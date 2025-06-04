import src.primality.helpers as helpers
import random, math
from math import gcd
from sympy import factorint
from statistics import mean
from typing import Optional, List, Dict, Tuple

def init_criteria_data(numbers: List[int]):
    global criteria_data
    criteria_data = {
        "Fermat": {n: {"a_values": [], "results": []} for n in numbers},
        "Wilson": {n: {"result": None} for n in numbers},
        "Initial Lucas": {n: {"a": None, "condition1": None, "early_break": None, "result": None} for n in numbers},
        "Lucas": {n: {"a": None, "condition1": None, "early_break": None, "result": None} for n in numbers},
        "Optimized Lucas": {n: {"factors": factorint(n-1), "tests": {}, "result": None} for n in numbers}
    }

def fermat_criterion(n: int, k: int = 1) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2: return True
    
    for _ in range(k):
        a = random.randint(2, n-1)
        gcd_ok = gcd(a, n) == 1
        test_ok = pow(a, n-1, n) == 1
        
        criteria_data["Fermat"][n]["a_values"].append(a)
        criteria_data["Fermat"][n]["results"].append(gcd_ok and test_ok)
        
        if not gcd_ok:
            return False
        if not test_ok:
            return False
    return True

def wilson_criterion(p: int) -> bool:
    if p <= 1: raise ValueError("p must be greater than 1")
    result = math.factorial(p - 1) % p == p - 1
    criteria_data["Wilson"][p]["result"] = result
    return result

def initial_lucas_test(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2:
        criteria_data["Initial Lucas"][n]["result"] = True
        return True
    
    a = random.randint(2, n-2)
    condition1 = pow(a, n-1, n) == 1
    criteria_data["Initial Lucas"][n]["a"] = a
    criteria_data["Initial Lucas"][n]["condition1"] = condition1
    
    if not condition1:
        criteria_data["Initial Lucas"][n]["result"] = False
        return False
    
    for m in range(1, n-1):
        if pow(a, m, n) == 1:
            criteria_data["Initial Lucas"][n]["early_break"] = m
            criteria_data["Initial Lucas"][n]["result"] = False
            return False
    criteria_data["Initial Lucas"][n]["result"] = True
    return True

def lucas_test(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2:
        criteria_data["Lucas"][n]["result"] = True
        return True
    
    a = random.randint(2, n-1)
    condition1 = pow(a, n-1, n) == 1
    criteria_data["Lucas"][n]["a"] = a
    criteria_data["Lucas"][n]["condition1"] = condition1
    
    if not condition1:
        criteria_data["Lucas"][n]["result"] = False
        return False
    
    for m in range(1, n):
        if (n-1) % m == 0 and pow(a, m, n) == 1:
            criteria_data["Lucas"][n]["early_break"] = m
            criteria_data["Lucas"][n]["result"] = False
            return False
    criteria_data["Lucas"][n]["result"] = True
    return True

def optimized_lucas_test(n: int) -> bool:
    if n <= 1: raise ValueError("n must be greater than 1")
    if n == 2:
        criteria_data["Optimized Lucas"][n]["result"] = True
        return True
    
    factors = criteria_data["Optimized Lucas"][n]["factors"]
    for q in factors:
        for a in range(2, n):
            condition1 = pow(a, n-1, n) == 1
            condition2 = pow(a, (n-1)//q, n) != 1
            
            if q not in criteria_data["Optimized Lucas"][n]["tests"]:
                criteria_data["Optimized Lucas"][n]["tests"][q] = []
            criteria_data["Optimized Lucas"][n]["tests"][q].append((a, condition1 and condition2))
            
            if condition1 and condition2:
                break
        else:
            criteria_data["Optimized Lucas"][n]["result"] = False
            return False
    criteria_data["Optimized Lucas"][n]["result"] = True
    return True


def format_timing(times: List[float]) -> str:
    return f"⏱ Time: {times[0]*1000:.2f}ms"
    #return f"⏱ Best: {min(times)*1000:.2f}ms | Avg: {mean(times)*1000:.2f}ms | Worst: {max(times)*1000:.2f}ms"

def criteria_protocoll(numbers: List[int], timings: Optional[Dict[str, List[Dict]]] = None):
    for n in numbers:
        print(f"\n\033[1mTesting n = {n}\033[0m")
        
        # Fermat
        if n in criteria_data["Fermat"]:
            data = criteria_data["Fermat"][n]
            print(f"Fermat: {'✅ Prim' if all(data['results']) else '❌ Zusammengesetzt'}")
            print("   ", " | ".join(f"a={a}→{'✓' if res else '✗'}" 
                 for a, res in zip(data["a_values"], data["results"])))
            if timings:
                times = [d["avg_time"] for d in timings["Fermat"] if d["n"] == n]
                if times: print("    ", format_timing(times))

        # Wilson
        if n in criteria_data["Wilson"]:
            result = criteria_data["Wilson"][n]["result"]
            print(f"Wilson: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
            if timings:
                times = [d["avg_time"] for d in timings["Wilson"] if d["n"] == n]
                if times: print("    ", format_timing(times))

        # Initial Lucas
        if n in criteria_data["Initial Lucas"]:
            data = criteria_data["Initial Lucas"][n]
            print(f"Initial Lucas: {'✅ Prim' if data['result'] else '❌ Zusammengesetzt'}")
            print(f"   a={data['a']}: Bedingung 1 {'✓' if data['condition1'] else '✗'}")
            if "early_break" in data and data["early_break"]:
                print(f"   ⚠️ Abbruch bei m={data['early_break']}")
            if timings:
                times = [d["avg_time"] for d in timings["Initial Lucas"] if d["n"] == n]
                if times: print("    ", format_timing(times))

        # Lucas
        if n in criteria_data["Lucas"]:
            data = criteria_data["Lucas"][n]
            print(f"Lucas: {'✅ Prim' if data['result'] else '❌ Zusammengesetzt'}")
            print(f"   a={data['a']}: Bedingung 1 {'✓' if data['condition1'] else '✗'}")
            if "early_break" in data and data["early_break"]:
                print(f"   ⚠️ Abbruch bei m={data['early_break']}")
            if timings:
                times = [d["avg_time"] for d in timings["Lucas"] if d["n"] == n]
                if times: print("    ", format_timing(times))

        # Optimierter Lucas
        if n in criteria_data["Optimized Lucas"]:
            data = criteria_data["Optimized Lucas"][n]
            print(f"Optimierter Lucas: {'✅ Prim' if data['result'] else '❌ Zusammengesetzt'}")
            for q, tests in data["tests"].items():
                print(f"   q={q}:", " | ".join(f"a={a}→{'✓' if res else '✗'}" for a, res in tests))
            if timings:
                times = [d["avg_time"] for d in timings["Optimized Lucas"] if d["n"] == n]
                if times: print("    ", format_timing(times))









"""

def fermat_criterion_detail(n: int, k: int = 1) -> Tuple[bool, List[Tuple[int, bool]]]:
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

def wilson_criterion_detail(n: int) -> Tuple[bool, None]:
    if n <= 1: return (False, None)
    result = (math.factorial(n-1) % n) == n-1
    return (result, None)

def initial_lucas_test_detail(n: int) -> Tuple[bool, List[Tuple[int, bool]]]:
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

def lucas_test_detail(n: int) -> Tuple[bool, List[Tuple[str, bool]]]:
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

def optimized_lucas_test_detail(n: int) -> Tuple[bool, Dict[int, List[Tuple[int, bool]]]]:
    details = {}
    if n <= 1: return (False, details)
    if n == 2: return (True, details)
    
    factors = factorint(n-1)
    for q in factors:
        details[q] = []
        for a in range(2, min(n, 100)): # TODO: change for big a
            condition1 = pow(a, n-1, n) == 1
            condition2 = pow(a, (n-1)//q, n) != 1
            details[q].append((a, condition1 and condition2))
            if condition1 and condition2:
                break
        else:
            return (False, details)
    return (True, details)



# Format timing results for output
def format_timing(times: List[float]) -> str:
    return f"⏱ Best: {min(times)*1000:.2f}ms | Avg: {mean(times)*1000:.2f}ms | Worst: {max(times)*1000:.2f}ms"

# Main function to print the protocol for each criterion
def criteria_protocoll(numbers: List[int], timings: Optional[Dict[str, List[Dict]]] = None):
    for n in numbers:
        print(f"\n\033[1mTeste n = {n}\033[0m")

        # Fermat
        result, details = fermat_criterion_detail(n, k=3)
        print(f"Fermat: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
        print("  ", " | ".join([f"a={a}→{'✓' if res else '✗'}" for a, res in details]))
        if timings:
            times = [d["avg_time"] for d in timings["Fermat"] if d["n"] == n]
            if times:
                print("   ", format_timing(times))

        # Wilson
        result, _ = wilson_criterion_detail(n)
        print(f"Wilson: {'✅ Prim' if result else '❌ Zusammengesetzt'} (kein a)")
        if timings:
            times = [d["avg_time"] for d in timings["Wilson"] if d["n"] == n]
            if times:
                print("   ", format_timing(times))

        # Initial Lucas
        result, details = initial_lucas_test_detail(n)
        print(f"Initial Lucas: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
        if details:
            a, res = details[0]
            print(f"  a={a}: Bedingung 1 {'✓' if res else '✗'}")
            if len(details) > 1:
                print(f"  ⚠️ Abbruch bei {details[1][0]}")
        if timings:
            times = [d["avg_time"] for d in timings["Initial Lucas"] if d["n"] == n]
            if times:
                print("   ", format_timing(times))

        # Lucas
        result, details = lucas_test_detail(n)
        print(f"Lucas: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
        if details:
            a, res = details[0]
            print(f"  a={a}: Bedingung 1 {'✓' if res else '✗'}")
            if len(details) > 1:
                print(f"  ⚠️ Abbruch bei {details[1][0]}")
        if timings:
            times = [d["avg_time"] for d in timings["Lucas"] if d["n"] == n]
            if times:
                print("   ", format_timing(times))

        # Optimierter Lucas
        result, details = optimized_lucas_test_detail(n)
        print(f"Optimierter Lucas: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
        for q, tests in details.items():
            row = " | ".join([f"a={a}→{'✓' if res else '✗'}" for a, res in tests])
            print(f"  q={q}: {row}")
        if timings:
            times = [d["avg_time"] for d in timings["Optimized Lucas"] if d["n"] == n]
            if times:
                print("   ", format_timing(times))
"""
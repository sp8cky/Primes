from math import gcd

# Helper functions for primality tests and number theory
def divides(a: int, b: int) -> bool:
    if a == 0:
        raise ValueError("Division by 0.")
    return b % a == 0

# Check if n is a real potency, i.e., n = a^b with a, b > 1
def is_real_potency(n: int):
    """Check, if n is a real potency: n = a^b with a, b > 1. """
    if n <= 1:
        return False
    for b in range(2, int(n**0.5) + 1):
        a = round(n ** (1 / b))
        if a ** b == n:
            return True
    return False

# calculate the order of n modulo r
def order(n: int, r: int) -> int:
    if gcd(n, r) != 1:
        return 0
    for k in range(1, r):
        if pow(n, k, r) == 1:
            return k
    return 0
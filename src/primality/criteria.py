from math import gcd

def fermat_criterion(a: int, n: int) -> bool:
    if n <= 1 or a <= 1 or a >= n:
        raise ValueError("a must be in the range 2 to n-1 and n must be greater than 1")
    if gcd(a, n) != 1:
        return False
    return pow(a, n - 1, n) == 1

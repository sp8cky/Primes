def divides(a: int, b: int) -> bool:
    if a == 0:
        raise ValueError("Division by 0.")
    return b % a == 0

def is_real_potency(n: int):
    """Check, if n is a real potency: n = a^b with a, b > 1. """
    if n <= 1:
        return False
    for b in range(2, int(n**0.5) + 1):
        a = round(n ** (1 / b))
        if a ** b == n:
            return True
    return False
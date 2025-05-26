def divides(a: int, b: int) -> bool:
    if a == 0:
        raise ValueError("Division by 0.")
    return b % a == 0
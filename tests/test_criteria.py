from src.primality.criteria import *
import math, pytest

# Test cases
test_cases = [
    (2, True),
    (3, True),
    (4, False),
    (5, True),
    (6, False),
    (7, True),
    (8, False),
    (9, False),
    (10, False),
    (11, True),
    (12, False),
]

def test_wilson_criterion():
    failed_cases = 0
    total_cases = len(test_cases)

    for p, expected in test_cases:
        try:
            result = wilson_criterion(p)
            assert result == expected
        except Exception as e:
            failed_cases += 1
            print(f"Test for p={p} raised exception: {e}")
            continue
        
        if result != expected:
            failed_cases += 1
            print(f"Error for p={p}: Expected {expected}, got {result}")

    print(f"Wilson criterion: {total_cases - failed_cases} of {total_cases} tests passed.")
    assert failed_cases == 0, f"{failed_cases} tests failed."

print(wilson_criterion(5))  # True
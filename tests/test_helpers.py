import pytest
import math
from src.primality.helpers import *

# Test data
test_numbers = [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 17, 19, 21, 25, 31, 37]
prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 31, 37]
composite_numbers = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 21, 25]

# Test for divides()
@pytest.mark.parametrize("a,b,expected", [
    (2, 4, True),
    (3, 10, False),
    (5, 25, True),
    (7, 49, True),
    (3, 7, False),
])
def test_divides(a, b, expected):
    if a == 0:
        with pytest.raises(ValueError):
            divides(a, b)
    else:
        assert divides(a, b) == expected

# Test for is_real_potency()
@pytest.mark.parametrize("n,expected", [
    (4, True),     # 2^2
    (8, True),     # 2^3
    (9, True),     # 3^2
    (16, True),    # 2^4 and 4^2
    (25, True),    # 5^2
    (27, True),    # 3^3
    (1, False),
    (2, False),
    (3, False),
    (5, False),
    (7, False),
    (10, False),
])
def test_is_real_potency(n, expected):
    assert is_real_potency(n) == expected

# Test for order()
@pytest.mark.parametrize("n,r,expected", [
    (2, 3, 2),     # 2^2 ≡ 1 mod 3
    (2, 5, 4),     # 2^4 ≡ 1 mod 5
    (3, 5, 4),     # 3^4 ≡ 1 mod 5
    (2, 7, 3),     # 2^3 ≡ 1 mod 7
    (3, 7, 6),     # 3^6 ≡ 1 mod 7
    (5, 7, 6),     # 5^6 ≡ 1 mod 7
    (2, 9, 6),     # 2^6 ≡ 1 mod 9
    (4, 5, 0),     # gcd(4,5) = 1 but should be tested
    (3, 9, 0),     # gcd(3,9) ≠ 1
])
def test_order(n, r, expected):
    assert order(n, r) == expected

# Test for find_pocklington_decomposition()
@pytest.mark.parametrize("n,expected_options", [
    (5, [(1, 2, 2)]),
    (17, [(2, 2, 3), (1, 2, 4)]),
    (19, [(9, 2, 1), (2, 3, 2)]),
    (37, [(9, 2, 2), (4, 3, 2)]),
    (1093, [(13, 3, 5), (3, 7, 3)]),
    (7, [(2, 3, 1)]),
    (13, [(3, 2, 2)]),
])
def test_find_pocklington_decomposition(n, expected_options):
    result = find_pocklington_decomposition(n)
    assert result in expected_options


@pytest.mark.parametrize("N,expected_decompositions", [
    # Primzahlen mit gültigen Zerlegungen
    (5, [(1, 2, 2)]),          # 5 = 1*2² + 1 (1 < 4)
    (17, [(1, 2, 4), (2, 2, 3)]),         # 17 = 1*2⁴ + 1 (1 < 16)
    (37, [(4, 3, 2)]),         # 37 = 9*2² + 1 (9 < 4? Nein → Test soll fehlschlagen)
    (1093, [(1, 3, 6)]),  # 1093 = 273*2²+1 (273 < 4? Nein) und 1*3⁶+1 (1 < 729)
    (3511, []),    # 3511 = 1755*2¹ + 1 (1755 < 2? Nein)
    (7, []),                   # 7-1=6 → Keine Zerlegung mit K < p^n
    (13, []),                  # 13-1=12 → Keine gültige Zerlegung
    (31, []),                  # 31-1=30 → Keine gültige Zerlegung
    (2, []),                   # N < 3 → Leere Liste
    (1, []),                   # N < 3 → Leere Liste
])
def test_find_all_decompositions(N, expected_decompositions):
    decompositions = find_all_decompositions(N)

    assert len(decompositions) == len(expected_decompositions)

    for (K, p, n), (exp_K, exp_p, exp_n) in zip(decompositions, expected_decompositions):
        computed_N = K * (p**n) + 1
        assert computed_N == N
        assert K < (p**n)
        assert (K, p, n) == (exp_K, exp_p, exp_n)



# Test for find_quadratic_non_residue()
@pytest.mark.parametrize("p,expected", [
    (3, 2),     # 2 is QNR mod 3
    (5, 2),     # 2 is QNR mod 5
    (7, 3),     # 3 is QNR mod 7
    (11, 2),    # 2 is QNR mod 11
    (13, 2),    # 2 is QNR mod 13
    (17, 3),    # 3 is QNR mod 17
    (2, None),  # No QNR for p=2
])
def test_find_quadratic_non_residue(p, expected):
    assert find_quadratic_non_residue(p) == expected

# Test for cyclotomic_polynomial()
@pytest.mark.parametrize("n,x,expected", [
    (1, 2, 1),          # Φ₁(x) = x - 1 → Φ₁(2) = 1
    (2, 2, 3),          # Φ₂(x) = x + 1 → Φ₂(2) = 3
    (3, 2, 7),          # Φ₃(x) = x² + x + 1 → Φ₃(2) = 7
    (4, 2, 5),          # Φ₄(x) = x² + 1 → Φ₄(2) = 5
    (5, 2, 31),         # Φ₅(x) = x⁴ + x³ + x² + x + 1 → Φ₅(2) = 31
    (6, 2, 3),          # Φ₆(x) = x² - x + 1 → Φ₆(2) = 3
    (3, 3, 13),         # Φ₃(3) = 9 + 3 + 1 = 13
])
def test_cyclotomic_polynomial(n, x, expected):
    assert cyclotomic_polynomial(n, x) == expected

# Test for is_fermat_number()
@pytest.mark.parametrize("n,expected", [
    (3, True),      # F₀ = 3
    (5, True),      # F₁ = 5
    (17, True),     # F₂ = 17
    (257, True),    # F₃ = 257
    (65537, True),  # F₄ = 65537
    (1, False),
    (2, False),
    (4, False),
    (6, False),
    (9, False),
    (15, False),
    (33, False),    # Not a Fermat number
])
def test_is_fermat_number(n, expected):
    assert is_fermat_number(n) == expected
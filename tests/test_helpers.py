import pytest
import math
from src.primality.helpers import *

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


@pytest.mark.parametrize("n,expected", [
    # Gültige Zerlegungen (n = p * 2^k + 1, mit p prim)
    (5, None),    # No valid decomposition (p=1 is not prime)
    (7, None),    # No valid decomposition (no integer p for n≥2)
    (13, (3, 2)),     # 13 = 3 * 2² + 1 (3 ist prim)
    (17, (2, 3)),     # 17 = 2 * 2³ + 1 (2 ist prim)
    (41, (5, 3)),     # 41 = 5 * 2³ + 1 (5 ist prim)
    (97, (3, 5)),     # 97 = 3 * 2⁵ + 1 (3 ist prim)
    (113, (7, 4)),    # 113 = 7 * 2⁴ + 1 (7 ist prim)

    # Ungültige Fälle (kein prim p in Zerlegung möglich)
    (37, None),       # 37 = 9 * 2² + 1 (9 nicht prim)
    (19, None),       # 19-1 = 18 → 9*2¹ (9 nicht prim), 3*2² (3 prim, aber 3 < 4? Nein, weil K < p^n nicht geprüft wird)
    (31, None),       # 31-1 = 30 → 15*2¹ (15 nicht prim)
    (1093, None),     # 1093-1 = 1092 → 273*2² (273 nicht prim)
    (3511, None),     # 3511-1 = 3510 → 1755*2¹ (1755 nicht prim)

    # Sonderfälle (n ≤ 3)
    (3, None),
    (2, None),
    (1, None),
])
def test_find_rao_decomposition(n, expected):
    result = find_rao_decomposition(n)
    assert result == expected


# Test for find_ramzy_decomposition()
@pytest.mark.parametrize("n, expected", [
    # Valid cases (all possible decompositions)
    (5, (1, 2, 2)),                 # Only 4 = 1*2^2 works (j=0: 2 >= 1)
    (9, (2, 2, 2)),                  # Only 8 = 2*2^2 works (j=0: 2 >= 2)
    (17, (2, 2, 3)),      # 16 = 1*2^4 (j=0: 8 >= 1) or 2*2^3 (j=0: 4 >= 2)
    (25, (3, 2, 3)),                 # Only 24 = 3*2^3 works (j=0: 4 >= 3)
    (3, (1, 2, 1)),                  # Only 2 = 1*2^1 works (j=0: 1 >= 1)
    (2, None),
    (4, (1, 3, 1)),
    (7, None),                       # 6=3*2^1 fails (1 >= 3? No)
    (11, None),                      # No valid (K,p,n) pair
    (19, (2, 3, 2)),                 # 18=9*2^1 fails (1 >= 9? No)
    (37, None),                      # 36=9*2^2 fails (2 >= 9? No)
    (13, None),                 # 12=3*2^2 works (2 >= 3? No)
    (1093, None),            # 1092=2*2^6 works (6 >= 2? Yes)
    (3, (1, 2, 1)),                  # 2=1*2^1 works (1 >= 1? Yes)
])
def test_find_ramzy_decomposition(n, expected):
    result = find_ramzy_decomposition(n)
    assert result == expected
 
# Test for find_pocklington_decomposition()
@pytest.mark.parametrize("n, expected", [
    (7, (2, 3, 1)),
    (13, (3, 2, 2)),
    (17, (1, 2, 4)),
    (31, None),
    (5, (1, 2, 2)),
    (19, (2, 3, 2)),
    (11, (2, 5, 1)),
    (23, (2, 11, 1)),
    (3, (1, 2, 1)),
])
def test_find_pocklington_decomposition(n, expected):
    result = find_pocklington_decomposition(n)
    assert result == expected


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
    (6, False),
    (10, False),
    (15, False),
    (18, False),
    (20, False),
    (100, False),
    (1000, False),
    (10000, False),
    (100000, False),
    (1000000, False),
    (0, False),
    (1, False),
    (2, False),
    (4, False),
    (8, False),
    (9, False),
    (16, False),
    (32, False),
    (64, False),
    (128, False),
    (256, False),
    (258, False),
    (65536, False),
    (65538, False),
    (999999, False),
    (4294967297, True),  # F₅ = 2^(2^5) + 1
    (4294967296, False),
    (4294967298, False),
])
def test_is_fermat_number(n, expected):
    assert is_fermat_number(n) == expected


# Test cases for is_mersenne_number()
@pytest.mark.parametrize("n,expected", [
    # Valid Mersenne numbers (both prime and composite)
    (3, True),       # M₂ = 3 (prime)
    (7, True),       # M₃ = 7 (prime)
    (15, True),      # M₄ = 15 (composite)
    (31, True),      # M₅ = 31 (prime)
    (63, True),      # M₆ = 63 (composite)
    (127, True),     # M₇ = 127 (prime)
    (255, True),     # M₈ = 255 (composite)
    (511, True),     # M₉ = 511 (composite)
    (1023, True),    # M₁₀ = 1023 (composite)
    (2047, True),    # M₁₁ = 2047 (composite)
    (8191, True),    # M₁₃ = 8191 (prime)
    (131071, True),  # M₁₇ = 131071 (prime)
    
    # Non-Mersenne numbers
    (0, False),
    (1, False),
    (2, False),
    (4, False),
    (5, False),
    (6, False),
    (8, False),
    (9, False),
    (10, False),
    (16, False),
    (32, False),
    (64, False),
    (100, False),
    (1000, False),
    
    # Large numbers
    (2**17-1, True),    # M₁₇ = 131071
    (2**19-1, True),    # M₁₉ = 524287
    (2**31-1, True),    # M₃₁ = 2147483647
    (2**61-1, True),    # M₆₁ = 2305843009213693951
    (2**89-1, True),    # M₈₉ (composite)
    (2**107-1, True),   # M₁₀₇ (prime)
    
    # Numbers just below/above Mersenne numbers
    (2**5-2, False),    # 30 (M₅ = 31)
    (2**7+1, False),    # 129 (M₇ = 127)
    (2**13-2, False),   # 8190 (M₁₃ = 8191)
])
def test_is_mersenne_number(n, expected):
    assert is_mersenne_number(n) == expected
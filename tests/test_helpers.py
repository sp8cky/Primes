import pytest
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

# Test for order()
@pytest.mark.parametrize("n,r,expected", [
    (2, 3, 2),
    (2, 5, 4),
    (3, 5, 4),
    (2, 7, 3),
    (3, 7, 6),
    (5, 7, 6),
    (2, 9, 6),
    (4, 5, 2),
    (3, 9, 0),
])
def test_order(n, r, expected):
    assert order(n, r) == expected


@pytest.mark.parametrize("n,expected", [
    # Gültige Zerlegungen (n = p * 2^k + 1, mit p prim)
    (5, None),
    (7, None),
    (13, (3, 2)),
    (17, (2, 3)),
    (41, (5, 3)),
    (97, (3, 5)),
    (113, (7, 4)),

    # Ungültige Fälle (kein prim p in Zerlegung möglich)
    (37, None),
    (19, None),
    (31, None),
    (1093, None),
    (3511, None),

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
    (5, (1, 2, 2)),
    (9, (2, 2, 2)),
    (17, (2, 2, 3)),
    (25, (3, 2, 3)),
    (3, (1, 2, 1)),
    (2, None),
    (4, (1, 3, 1)),
    (7, None),
    (11, None),
    (19, (2, 3, 2)),
    (37, None),
    (13, None),
    (1093, None),
    (3, (1, 2, 1)),
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
    (3, 2),
    (5, 2),
    (7, 3),
    (11, 2),
    (13, 2),
    (17, 3),
    (2, None),
])
def test_find_quadratic_non_residue(p, expected):
    assert find_quadratic_non_residue(p) == expected

# Test for is_fermat_number()
@pytest.mark.parametrize("n,expected", [
    (3, True),
    (5, True),
    (17, True),
    (257, True),
    (65537, True),
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
    (3, True),
    (7, True),
    (15, True),
    (31, True),
    (63, True),
    (127, True),
    (255, True),
    (511, True),
    (1023, True),
    (2047, True),
    (8191, True),
    (131071, True),
    
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
    (2**17-1, True),
    (2**19-1, True),
    (2**31-1, True),
    (2**61-1, True),
    (2**89-1, True),
    (2**107-1, True),
    
    # Numbers just below/above Mersenne numbers
    (2**5-2, False), 
    (2**7+1, False), 
    (2**13-2, False),
])
def test_is_mersenne_number(n, expected):
    assert is_mersenne_number(n) == expected
import math, pytest
from src.primality.criteria import *

# === Fermat criterion ===
@pytest.mark.parametrize("a, n, expected", [
    (2, 3, True),     # 3 ist prim, 2^2 ≡ 1 mod 3
    (2, 4, False),    # 2 und 4 nicht teilerfremd → gcd = 2 → False
    (3, 5, True),     # 3^4 = 81 ≡ 1 mod 5 → Fermat-Test besteht
    (2, 6, False),    # 6 keine Primzahl, 2^5 = 32 ≡ 2 mod 6
    (2, 7, True),     # 2^6 = 64 ≡ 1 mod 7 → besteht
    (4, 7, True),     # 4^6 = 4096 ≡ 1 mod 7
    (5, 341, False),   # Achtung: 341 ist Carmichael-Zahl! → Fermat-Test besteht fälschlich!
    (2, 341, True),   # Zeigt Schwäche des Tests: 341 = 11 * 31
    (2, 9, False),    # 2^8 = 256 ≡ 4 mod 9 → kein Fermat-Zeuge
])
def test_fermat_criterion(a, n, expected):
    result = fermat_criterion(a, n)
    assert result == expected, f"fermat_criterion(a={a}, n={n}) fehlgeschlagen: Erwartet {expected}, erhalten {result}"


# === wilson criterion ===
test_cases = [
    (2, True),
    (3, True),
    (7, True),
    (8, False),
    (9, False),
    (11, True),
    (12, False),
]

@pytest.mark.parametrize("p,expected", test_cases)
def test_wilson_criterion(p, expected):
    result = wilson_criterion(p)
    assert result == expected, f"Error for p={p}: expected {expected}, got {result}"

# === Initial Lucas Test ===
@pytest.mark.parametrize("a, n, expected", [
    (2, 3, True),     # 2^2 ≡ 1 mod 3; kein m < 2 erfüllt a^m ≡ 1
    (2, 4, False),    # 2^3 ≡ 0 mod 4 → sofort False
    (3, 5, True),     # 3^4 ≡ 1 mod 5, kein m < 4 mit a^m ≡ 1
    (2, 7, False),    # Lucas-Test mit a=2 für Primzahl 7 TODO: gibt fälschlicherweise False zurück
    (4, 7, False),    # 4^3 = 64 ≡ 1 mod 7, obwohl 3 kein Teiler von 6
])
def test_initial_lucas_test(a, n, expected):
    result = initial_lucas_test(a, n)
    assert result == expected, f"initial_lucas_test(a={a}, n={n}) fehlgeschlagen: Erwartet {expected}, erhalten {result}"

# === Lucas Test ===
@pytest.mark.parametrize("a, n, expected", [
    (2, 3, True),
    (3, 7, True),
    (2, 9, False),     # 2^6 ≡ 1 mod 9, aber 2^3 ≡ 8 ≠ 1 — keine gute Wahl
    (2, 11, True),
    (2, 15, False),    # 15 ist keine Primzahl → Lucas-Test schlägt fehl
])
def test_lucas_test(a, n, expected):
    result = lucas_test(a, n)
    assert result == expected, f"lucas_test(a={a}, n={n}) fehlgeschlagen: Erwartet {expected}, erhalten {result}"

# === Optimized Lucas Test ===
@pytest.mark.parametrize("n, expected", [
    (2, True),
    (3, True),
    (5, True),
    (7, True),
    (11, True),
    (9, False),       # 9 = 3^2, keine Primzahl
    (15, False),      # zusammengesetzt
    (21, False),      # zusammengesetzt
    (17, True),
    (19, True),
])
def test_optimized_lucas_test(n, expected):
    result = optimized_lucas_test(n)
    assert result == expected, f"optimized_lucas_test(n={n}) fehlgeschlagen: Erwartet {expected}, erhalten {result}"
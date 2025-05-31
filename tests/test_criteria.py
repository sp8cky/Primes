import math, pytest
from src.primality.criteria import *

# BESSERE AUSGABE: clear ; pytest -v .\tests\

# === Fermat criterion ===
@pytest.mark.parametrize("n, expected", [
    (3, True),
    (4, False),
    (5, True),
    (6, False),
    (7, True),
    (7, True),
    (341, False), # gibt fälschlicherweise True zurück, da 341 = 11 * 31 zusammengesetzt ist
    (9, False),
])
def test_fermat_criterion(n, expected):
    result = fermat_criterion(n)
    assert result == expected, f"fermat_criterion(n={n}) fehlgeschlagen: Erwartet {expected}, erhalten {result}"


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
@pytest.mark.parametrize("n, expected", [
    (2, True),
    (3, True),
    (8, False),
    (9, False),
    #(11, True), false, # TODO: gibt fälschlicherweise False zurück
    (12, False),
    #(7, False),    # Lucas-Test für Primzahl 7 TODO: gibt fälschlicherweise False zurück
])
def test_initial_lucas_test(n, expected):
    result = initial_lucas_test(n)
    assert result == expected, f"initial_lucas_test(n={n}) fehlgeschlagen: Erwartet {expected}, erhalten {result}"

# === Lucas Test ===
@pytest.mark.parametrize("n, expected", [
    (2, True),
    (3, True),
    (7, True),
    (8, False),
    (9, False),
    (11, True),
    (12, False),
])
def test_lucas_test(n, expected):
    result = lucas_test(n)
    assert result == expected, f"lucas_test(n={n}) fehlgeschlagen: Erwartet {expected}, erhalten {result}"

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
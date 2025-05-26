import math, pytest
from src.primality.tests import *

# === Miller-Rabin Test ===
@pytest.mark.parametrize("n, expected", [
    (3, True),        # Primzahl
    (5, True),        # Primzahl
    (7, True),        # Primzahl
    (11, True),
    (15, False),
    (17, True),
    (561, False),     # Carmichael-Zahl → sollte aussortiert werden
    (2047, False),    # 23 * 89
])
def test_miller_selfridge_rabin_test(n, expected):
    result = miller_selfridge_rabin_test(n, rounds=10)
    assert result == expected, f"Miller-Rabin-Test fehlgeschlagen für n={n}: Erwartet {expected}, erhalten {result}"

# === Solovay-Strassen Test ===
@pytest.mark.parametrize("n, expected", [
    (3, True),
    (5, True),
    (7, True),
    (9, False),
    (11, True),
    (15, False),
    (17, True),
    (561, False),     # Carmichael-Zahl
    (1729, False),    # Carmichael-Zahl
])
def test_solovay_strassen_test(n, expected):
    result = solovay_strassen_test(n, rounds=10)
    assert result == expected, f"Solovay-Strassen-Test fehlgeschlagen für n={n}: Erwartet {expected}, erhalten {result}"

# === AKS Test ===
@pytest.mark.parametrize("n, expected", [
    (2, True),
    (3, True),
    (5, True),
    (6, False),
    (7, True),
    (11, True),
])
def test_aks_test(n, expected):
    result = aks_test(n)
    assert result == expected, f"AKS-Test fehlgeschlagen für n={n}: Erwartet {expected}, erhalten {result}"
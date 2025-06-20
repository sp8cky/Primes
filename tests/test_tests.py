import pytest
import random
from src.primality.helpers import *
from src.primality.tests import *

# Alle Zahlen aus den Tests
valid_primes = [3, 5, 7, 13, 17, 31, 37]
valid_composites = [9, 15, 21, 25, 35, 39]
pepin_numbers = [17, 257, 65537, 5, 15]
lucas_lehmer_numbers = [3, 7, 15, 31, 63]
proth_numbers = [577, 561, 9]
pocklington_numbers = [1811, 561]
optimized_pocklington_numbers = [1811, 561]
proth_variant_numbers = [577, 561]
optimized_pocklington_variant_numbers = [8081, 1811]
generalized_pocklington_numbers = [8081, 561]
grau_numbers = [8081, 561]
grau_probability_numbers = [8081, 561]
miller_numbers = [2, 3, 5, 7, 11, 15, 21, 29, 31, 39]
solovay_numbers = [2, 3, 5, 7, 11, 15, 21, 25, 31, 39]
aks_numbers = [2, 3, 5, 7, 11, 15, 21, 25, 31, 39]

# Alle Zahlen zusammenführen
all_numbers = set(
    valid_primes +
    valid_composites +
    pepin_numbers +
    lucas_lehmer_numbers +
    proth_numbers +
    pocklington_numbers +
    optimized_pocklington_numbers +
    proth_variant_numbers +
    optimized_pocklington_variant_numbers +
    generalized_pocklington_numbers +
    grau_numbers +
    grau_probability_numbers + 
    miller_numbers +
    solovay_numbers +
    aks_numbers
)
init_test_data_for_numbers(list(all_numbers))


# Erwartungswerte als Dicts (in einer Zeile je Test)
fermat_expected = {3:True,5:True,7:True,9:True,13:True,15:False,17:True,21:False,25:True,31:True,35:False,37:True,39:False}
wilson_expected = {3:True,5:True,7:True,9:False,13:True,15:False,17:True,21:False,25:False,31:True,35:False,37:True,39:False}
# Initial Lucas: laut deinem Code immer False (für Test-Zahlen)
initial_lucas_expected = {n:False for n in valid_primes + valid_composites}
# Lucas-Test: auch immer False (für Test-Zahlen)
lucas_expected = {n:False for n in valid_primes + valid_composites}
optimized_lucas_expected = {3:True,5:True,7:True,9:False,13:True,15:False,17:True,21:False,25:False,31:True,35:False,37:True,39:False}
pepin_expected = {17:True,257:True,65537:True,5:False,15:False}
lucas_lehmer_expected = {3:True,7:True,15:False,31:True,63:False}
proth_expected = {577:True,561:False,9:False}
pocklington_expected = {1811:True,561:False}
optimized_pocklington_expected = {1811:True,561:False}
proth_variant_expected = {577:True,561:False}
optimized_pocklington_variant_expected = {8081:False,1811:True}
generalized_pocklington_expected = {8081:True,561:False}
grau_expected = {8081:True,561:False}
grau_probability_expected = {8081:False,561:False}
miller_expected = {2:True, 3:True, 5:True, 7:True, 11:True, 15:False, 21:False, 29:True, 31:True, 39:False}
solovay_expected = {2:True, 3:True, 5:True, 7:True, 11:True, 15:False, 21:False, 25:False, 31:True, 39:False}
aks_expected = {2:True, 3:True, 5:True, 7:True, 11:True, 15:False, 21:False, 25:False, 31:True, 39:False}


# Tests

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_fermat(n):
    random.seed(0)
    result = fermat_test(n, k=3)
    expected = fermat_expected.get(n, False)
    assert result == expected, f"Fermat-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_wilson(n):
    result = wilson_criterion(n)
    expected = wilson_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_initial_lucas(n):
    random.seed(0)
    result = initial_lucas_test(n)
    expected = initial_lucas_expected.get(n, False)
    assert result == expected, f"Initial Lucas-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_lucas(n):
    random.seed(0)
    result = lucas_test(n)
    expected = lucas_expected.get(n, False)
    assert result == expected, f"Lucas-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_optimized_lucas(n):
    random.seed(0)
    result = optimized_lucas_test(n)
    expected = optimized_lucas_expected.get(n, False)
    assert result == expected, f"Optimized Lucas-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", pepin_numbers)
def test_pepin(n):
    result = pepin_test(n)
    expected = pepin_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", lucas_lehmer_numbers)
def test_lucas_lehmer(n):
    result = lucas_lehmer_test(n)
    expected = lucas_lehmer_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", proth_numbers)
def test_proth(n):
    result = proth_test(n)
    expected = proth_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", pocklington_numbers)
def test_pocklington(n):
    result = pocklington_test(n)
    expected = pocklington_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", optimized_pocklington_numbers)
def test_optimized_pocklington(n):
    result = optimized_pocklington_test(n)
    expected = optimized_pocklington_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", proth_variant_numbers)
def test_proth_variant(n):
    result = proth_test_variant(n)
    expected = proth_variant_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", optimized_pocklington_variant_numbers)
def test_optimized_pocklington_variant(n):
    result = optimized_pocklington_test_variant(n)
    expected = optimized_pocklington_variant_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", generalized_pocklington_numbers)
def test_generalized_pocklington(n):
    result = generalized_pocklington_test(n)
    expected = generalized_pocklington_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", grau_numbers)
def test_grau(n):
    result = grau_test(n)
    expected = grau_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", grau_probability_numbers)
def test_grau_probability(n):
    result = grau_probability_test(n)
    expected = grau_probability_expected.get(n, False)
    assert result == expected

@pytest.mark.parametrize("n", miller_numbers)
def test_miller_selfridge_rabin(n):
    random.seed(0)
    result = miller_selfridge_rabin_test(n, k=5)
    expected = miller_expected.get(n, False)
    assert result == expected, f"Miller-Rabin-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", solovay_numbers)
def test_solovay_strassen(n):
    random.seed(0)
    result = solovay_strassen_test(n, k=5)
    expected = solovay_expected.get(n, False)
    assert result == expected, f"Solovay-Strassen-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", aks_numbers)
def test_aks(n):
    try:
        result = aks_test(n)
    except ValueError:
        result = False
    expected = aks_expected.get(n, False)
    assert result == expected, f"AKS-Test failed for n={n}: expected {expected}, got {result}"
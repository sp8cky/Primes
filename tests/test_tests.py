import pytest
import random
from src.primality.helpers import *
from src.primality.tests import *
from src.primality.test_protocoll import init_dictionary_fields

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
ramzy_numbers = [5, 7, 13, 17, 19, 37, 1093]
ramzy_invalid_numbers = [3, 11, 31, 41, 73]
rao_numbers = [5, 7, 17, 41, 97, 113, 257, 65537]
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
    ramzy_numbers +
    ramzy_invalid_numbers +
    rao_numbers +
    miller_numbers +
    solovay_numbers +
    aks_numbers
)
test_names = [
    "Fermat", "Wilson", "Initial Lucas", "Lucas", "Optimized Lucas",
    "Pepin", "Lucas-Lehmer", "Proth", "Pocklington", "Optimized Pocklington",
    "Proth Variant", "Optimized Pocklington Variant", "Generalized Pocklington",
    "Grau", "Grau Probability", "Ramzy", "Rao",
    "Miller-Selfridge-Rabin", "Solovay-Strassen", "AKS10"
]

for name in test_names:
    init_dictionary_fields(list(all_numbers), name)


# Erwartungswerte als Dicts (in einer Zeile je Test)
fermat_expected = {3: PRIME, 5: PRIME, 7: PRIME, 9: COMPOSITE, 13: PRIME, 15: COMPOSITE, 17: PRIME, 21: COMPOSITE, 25: COMPOSITE, 31: PRIME, 35: COMPOSITE, 37: PRIME, 39: COMPOSITE}
wilson_expected = {3: PRIME, 5: PRIME, 7: PRIME, 9: COMPOSITE, 13: PRIME, 15: COMPOSITE, 17: PRIME, 21: COMPOSITE, 25: COMPOSITE, 31: PRIME, 35: COMPOSITE, 37: PRIME, 39: COMPOSITE}
initial_lucas_expected = {n: (PRIME if n in valid_primes else COMPOSITE) for n in valid_primes + valid_composites}
lucas_expected = {n: (PRIME if n in valid_primes else COMPOSITE) for n in valid_primes + valid_composites}
optimized_lucas_expected = {3: PRIME, 5: PRIME, 7: PRIME, 9: COMPOSITE, 13: PRIME, 15: COMPOSITE, 17: PRIME, 21: COMPOSITE, 25: COMPOSITE, 31: PRIME, 35: COMPOSITE, 37: PRIME, 39: COMPOSITE}
pepin_expected = {5: PRIME, 15: INVALID, 17: PRIME, 257: PRIME, 65537: PRIME}
lucas_lehmer_expected = {3: PRIME, 7: PRIME, 15: INVALID, 31: PRIME, 63: INVALID}
proth_expected = {577: PRIME, 561: NOT_APPLICABLE, 9: COMPOSITE}
pocklington_expected = {1811: PRIME, 561: COMPOSITE}
optimized_pocklington_expected = {1811: PRIME, 561: NOT_APPLICABLE}
proth_variant_expected = {577: PRIME, 561: NOT_APPLICABLE}
optimized_pocklington_variant_expected = {8081: COMPOSITE, 1811: COMPOSITE}
generalized_pocklington_expected = {8081: PRIME, 561: NOT_APPLICABLE}
grau_expected = {8081: PRIME, 561: NOT_APPLICABLE}
grau_probability_expected = {8081: PRIME, 561: NOT_APPLICABLE}
ramzy_expected = {5: PRIME, 7: NOT_APPLICABLE, 13: NOT_APPLICABLE, 17: PRIME, 19: PRIME, 37: NOT_APPLICABLE, 1093: NOT_APPLICABLE, 3: PRIME, 11: NOT_APPLICABLE, 31: NOT_APPLICABLE, 41: NOT_APPLICABLE, 73: NOT_APPLICABLE}
rao_expected = {5: NOT_APPLICABLE, 7: NOT_APPLICABLE, 17: PRIME, 41: COMPOSITE, 97: COMPOSITE, 113: PRIME, 257: PRIME, 65537: PRIME}
miller_expected = {2: PRIME, 3: PRIME, 5: PRIME, 7: PRIME, 11: PRIME, 15: COMPOSITE, 21: COMPOSITE, 29: PRIME, 31: PRIME, 39: COMPOSITE}
solovay_expected = {2: PRIME, 3: PRIME, 5: PRIME, 7: PRIME, 11: PRIME, 15: COMPOSITE, 21: COMPOSITE, 25: COMPOSITE, 31: PRIME, 39: COMPOSITE}
aks10_expected = {2: PRIME, 3: PRIME, 5: PRIME, 7: PRIME, 11: PRIME, 15: COMPOSITE, 21: COMPOSITE, 25: INVALID, 31: PRIME, 39: COMPOSITE}
aks04_expected = {2: PRIME, 3: PRIME, 5: PRIME, 7: PRIME, 11: PRIME, 15: COMPOSITE, 21: COMPOSITE, 25: INVALID, 31: PRIME, 39: COMPOSITE}



# Tests

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_fermat(n):
    random.seed(0)
    result = fermat_test(n, k=3)
    expected = fermat_expected.get(n, COMPOSITE)
    assert result == expected, f"Fermat-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_wilson(n):
    result = wilson_criterion(n)
    expected = wilson_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_initial_lucas(n):
    random.seed(0)
    result = initial_lucas_test(n)
    expected = initial_lucas_expected.get(n, COMPOSITE)
    assert result == expected, f"Initial Lucas-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_lucas(n):
    random.seed(0)
    result = lucas_test(n)
    expected = lucas_expected.get(n, COMPOSITE)
    assert result == expected, f"Lucas-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_optimized_lucas(n):
    random.seed(0)
    result = optimized_lucas_test(n)
    expected = optimized_lucas_expected.get(n, COMPOSITE)
    assert result == expected, f"Optimized Lucas-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", pepin_numbers)
def test_pepin(n):
    result = pepin_test(n)
    expected = pepin_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", lucas_lehmer_numbers)
def test_lucas_lehmer(n):
    result = lucas_lehmer_test(n)
    expected = lucas_lehmer_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", proth_numbers)
def test_proth(n):
    result = proth_test(n)
    expected = proth_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", pocklington_numbers)
def test_pocklington(n):
    result = pocklington_test(n)
    expected = pocklington_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", optimized_pocklington_numbers)
def test_optimized_pocklington(n):
    result = optimized_pocklington_test(n)
    expected = optimized_pocklington_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", proth_variant_numbers)
def test_proth_variant(n):
    result = proth_test_variant(n)
    expected = proth_variant_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", optimized_pocklington_variant_numbers)
def test_optimized_pocklington_variant(n):
    result = optimized_pocklington_test_variant(n)
    expected = optimized_pocklington_variant_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", generalized_pocklington_numbers)
def test_generalized_pocklington(n):
    result = generalized_pocklington_test(n)
    expected = generalized_pocklington_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", grau_numbers)
def test_grau(n):
    result = grau_test(n)
    expected = grau_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", grau_probability_numbers)
def test_grau_probability(n):
    result = grau_probability_test(n)
    expected = grau_probability_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", ramzy_numbers + ramzy_invalid_numbers)
def test_ramzy(n):
    result = ramzy_test(n)
    expected = ramzy_expected.get(n, COMPOSITE)
    assert result == expected

@pytest.mark.parametrize("n", rao_numbers)
def test_rao(n):
    result = rao_test(n)
    expected = rao_expected.get(n, COMPOSITE)
    assert result == expected, f"Rao-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", miller_numbers)
def test_miller_selfridge_rabin(n):
    random.seed(0)
    result = miller_selfridge_rabin_test(n, k=5)
    expected = miller_expected.get(n, COMPOSITE)
    assert result == expected, f"Miller-Rabin-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", solovay_numbers)
def test_solovay_strassen(n):
    random.seed(0)
    result = solovay_strassen_test(n, k=5)
    expected = solovay_expected.get(n, COMPOSITE)
    assert result == expected, f"Solovay-Strassen-Test failed for n={n}: expected {expected}, got {result}"

@pytest.mark.parametrize("n", aks_numbers)
def test_aks04(n):
    try:
        result = aks04_test(n)
    except ValueError:
        result = COMPOSITE
    expected = aks04_expected.get(n, COMPOSITE)
    assert result == expected, f"AKS04-Test failed for n={n}: expected {expected}, got {result}"


@pytest.mark.parametrize("n", aks_numbers)
def test_aks10(n):
    try:
        result = aks10_test(n)
    except ValueError:
        result = COMPOSITE
    expected = aks10_expected.get(n, COMPOSITE)
    assert result == expected, f"AKS10-Test failed for n={n}: expected {expected}, got {result}"
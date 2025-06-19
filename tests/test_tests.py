# tests/test_tests.py
import pytest
from src.primality.helpers import *
from src.primality.tests import *

valid_primes = [3, 5, 7, 13, 17, 31, 37]
valid_composites = [9, 15, 21, 25, 35, 39]
special_numbers = [257, 65537, 127, 255, 511]

proth_numbers = [13, 41, 97]              # z.B. 3*2^2+1=13, 5*2^3+1=41, 3*2^5+1=97
non_proth_numbers = [15, 27, 33]

pocklington_numbers = [22, 46, 70]        # Beispiele, die n-1 einen Primfaktor hat
non_pocklington_numbers = [20, 30]

optimized_pocklington_numbers = [22, 46]
non_optimized_pocklington_numbers = [20, 30]

proth_variant_numbers = [13, 41, 97]       # Ungerade Proth-ähnliche Zahlen

generalized_pocklington_numbers = [22, 46]
grau_test_numbers = [22, 46]
grau_probability_numbers = [22, 46]

all_numbers = list(set(
    valid_primes +
    valid_composites +
    special_numbers +
    proth_numbers +
    non_proth_numbers +
    pocklington_numbers +
    non_pocklington_numbers +
    optimized_pocklington_numbers +
    non_optimized_pocklington_numbers +
    proth_variant_numbers +
    generalized_pocklington_numbers +
    grau_test_numbers +
    grau_probability_numbers
))

init_all_test_data(all_numbers)

# ----------- Beispielhafte Tests -----------

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_fermat(n):
    result = fermat_test(n, k=3)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_wilson(n):
    result = wilson_criterion(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_lucas(n):
    result = lucas_test(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_initial_lucas(n):
    result = initial_lucas_test(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_optimized_lucas(n):
    result = optimized_lucas_test(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", [3, 5, 17, 257, 65537, 15, 31])
def test_pepin(n):
    result = pepin_test(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", [3, 7, 31, 127, 255, 511])
def test_lucas_lehmer(n):
    result = lucas_lehmer_test(n)
    assert isinstance(result, bool)


@pytest.mark.parametrize("n", proth_numbers)
def test_proth_test_valid(n):
    result = proth_test(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", non_proth_numbers)
def test_proth_test_invalid(n):
    result = proth_test(n)
    assert result is False

@pytest.mark.parametrize("n", pocklington_numbers)
def test_pocklington_test_valid(n):
    result = pocklington_test(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", non_pocklington_numbers)
def test_pocklington_test_invalid(n):
    result = pocklington_test(n)
    assert result is False

@pytest.mark.parametrize("n", optimized_pocklington_numbers)
def test_optimized_pocklington_test_valid(n):
    result = optimized_pocklington_test(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", non_optimized_pocklington_numbers)
def test_optimized_pocklington_test_invalid(n):
    result = optimized_pocklington_test(n)
    assert result is False

@pytest.mark.parametrize("n", proth_variant_numbers)
def test_proth_test_variant(n):
    result = proth_test_variant(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", optimized_pocklington_numbers)
def test_optimized_pocklington_test_variant(n):
    result = optimized_pocklington_test_variant(n)
    assert isinstance(result, bool)

@pytest.mark.parametrize("N", generalized_pocklington_numbers)
def test_generalized_pocklington_test(N):
    result = generalized_pocklington_test(N)
    assert isinstance(result, bool)

@pytest.mark.parametrize("N", grau_test_numbers)
def test_grau_test(N):
    result = grau_test(N)
    assert isinstance(result, bool)

@pytest.mark.parametrize("N", grau_probability_numbers)
def test_grau_probability_test(N):
    result = grau_probability_test(N)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", [n for n in valid_primes + valid_composites if not is_real_potency(n)])
def test_miller_rabin(n):
    result = miller_selfridge_rabin_test(n, k=3)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", valid_primes + valid_composites)
def test_solovay_strassen(n):
    result = solovay_strassen_test(n, k=3)
    assert isinstance(result, bool)

@pytest.mark.parametrize("n", valid_primes)
def test_aks(n):
    result = aks_test(n)
    assert isinstance(result, bool)

from src.primality.tests import *
from src.primality.test_protocoll import *
from functools import partial

# Wahrscheinlichkeitsbasierte Tests und Standardwiederholungen
default_repeats = [3, 3, 3]
prob_tests = ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen"]
TEST_ORDER = ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen", "Initial Lucas", "Lucas", "Optimized Lucas", "Wilson", "AKS", "Proth", "Proth Variant", "Pocklington", "Optimized Pocklington", "Optimized Pocklington Variant", "Generalized Pocklington", "Grau", "Grau Probability", "Ramzy", "Rao",  "Pepin", "Lucas-Lehmer"]

# Zuordnung von Tests zu Funktionen und Metadaten
TEST_CONFIG = {
    "Fermat": {
        "runtime_function": fermat_test,
        "protocol_function": fermat_test_protocoll,
        "prob_test": True,
        "number_type": "large_prime",
        "testgroup": "Probabilistische Tests",
        "plotgroup": "Probabilistische Tests"
    },
    "Miller-Selfridge-Rabin": {
        "runtime_function": miller_selfridge_rabin_test,
        "protocol_function": miller_selfridge_rabin_test_protocoll,
        "prob_test": True,
        "number_type": "large_prime",
        "testgroup": "Probabilistische Tests",
        "plotgroup": "Probabilistische Tests"
    },
    "Solovay-Strassen": {
        "runtime_function": solovay_strassen_test,
        "protocol_function": solovay_strassen_test_protocoll,
        "prob_test": True,
        "number_type": "large_prime",
        "testgroup": "Probabilistische Tests",
        "plotgroup": "Probabilistische Tests"
    },
    "Initial Lucas": {
        "runtime_function": initial_lucas_test,
        "protocol_function": initial_lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas",
        "testgroup": "Lucas-Tests",
        "plotgroup": "Lucas-Tests"
    },
    "Lucas": {
        "runtime_function": lucas_test,
        "protocol_function": lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas",
        "testgroup": "Lucas-Tests",
        "plotgroup": "Lucas-Tests"
    },
    "Optimized Lucas": {
        "runtime_function": optimized_lucas_test,
        "protocol_function": optimized_lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas",
        "testgroup": "Lucas-Tests",
        "plotgroup": "Lucas-Tests"
    },
    "Wilson": {
        "runtime_function": wilson_criterion,
        "protocol_function": wilson_criterion_protocoll,
        "prob_test": False,
        "number_type": "small_prime",
        "testgroup": "Langsame Tests",
        "plotgroup": "Langsame Tests"
    },
    "AKS": {
        "runtime_function": aks_test,
        "protocol_function": aks_test_protocoll,
        "prob_test": False,
        "number_type": "small_prime",
        "testgroup": "Langsame Tests",
        "plotgroup": "Langsame Tests"
    },
    "Pepin": {
        "runtime_function": pepin_test,
        "protocol_function": pepin_test_protocoll,
        "prob_test": False,
        "number_type": "fermat",
        "testgroup": "Fermat-Zahlen",
        "plotgroup": "Spezielle Tests"
    },
    "Lucas-Lehmer": {
        "runtime_function": lucas_lehmer_test,
        "protocol_function": lucas_lehmer_test_protocoll,
        "prob_test": False,
        "number_type": "mersenne",
        "testgroup": "Mersenne-Zahlen",
        "plotgroup": "Spezielle Tests"
    },
    "Proth": {
        "runtime_function": proth_test,
        "protocol_function": proth_test_protocoll,
        "prob_test": False,
        "number_type": "proth",
        "testgroup": "Proth-Tests",
        "plotgroup": "Zusammengesetzte"
    },
    "Proth Variant": {
        "runtime_function": proth_test_variant,
        "protocol_function": proth_test_variant_protocoll,
        "prob_test": False,
        "number_type": "proth",
        "testgroup": "Proth-Tests",
        "plotgroup": "Zusammengesetzte"
    },
    "Pocklington": {
        "runtime_function": pocklington_test,
        "protocol_function": pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte"
    },
    "Optimized Pocklington": {
        "runtime_function": optimized_pocklington_test,
        "protocol_function": optimized_pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte"
    },
    "Optimized Pocklington Variant": {
        "runtime_function": optimized_pocklington_test_variant,
        "protocol_function": optimized_pocklington_test_variant_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte"
    },
    "Generalized Pocklington": {
        "runtime_function": generalized_pocklington_test,
        "protocol_function": generalized_pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte"
    },
    "Grau": {
        "runtime_function": grau_test,
        "protocol_function": grau_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte"
    },
    "Grau Probability": {
        "runtime_function": grau_probability_test,
        "protocol_function": grau_probability_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte"
    },
    "Ramzy": {
        "runtime_function": ramzy_test,
        "protocol_function": ramzy_test_protocoll,
        "prob_test": False,
        "number_type": "ramzy",
        "testgroup": "Ramzy",
        "plotgroup": "Zusammengesetzte"
    },
    "Rao": {
        "runtime_function": rao_test,
        "protocol_function": rao_test_protocoll,
        "prob_test": False,
        "number_type": "rao",
        "testgroup": "Rao-Tests",
        "plotgroup": "Zusammengesetzte"
    }
}

# Konfiguration für die Tests erzeugen (inkl. Wiederholungen & Spezial-Zahlentyp)
def get_test_config(include_tests=None, prob_test_repeats=None):
    global TEST_CONFIG  # Optional, aber nicht unbedingt nötig, wenn du nur liest

    if include_tests is None:
        include_tests = list(TEST_CONFIG.keys())

    if prob_test_repeats is None:
        prob_test_repeats = default_repeats

    config = {}  # Lokales Ergebnis-Dict

    for test in include_tests:
        cfg = TEST_CONFIG[test].copy()
        if cfg["prob_test"]:
            idx = prob_tests.index(test)
            k = prob_test_repeats[idx]
            cfg["prob_test_repeats"] = k
            cfg["runtime_function"] = partial(cfg["runtime_function"], k=k)
            cfg["protocol_function"] = partial(cfg["protocol_function"], k=k)
            cfg["label"] = f"{test} (k = {k})"
        else:
            cfg["label"] = test

        config[test] = cfg

    return config
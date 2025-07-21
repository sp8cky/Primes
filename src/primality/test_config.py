from src.primality.tests import *
from src.primality.test_protocoll import *
from functools import partial
from math import log, sqrt


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
        "number_type": "large",
        "testgroup": "Probabilistische Tests",
        "plotgroup": "Probabilistische Tests",
        "runtime_theoretical_fn": lambda n: log(n)**3
    },
    "Miller-Selfridge-Rabin": {
        "runtime_function": miller_selfridge_rabin_test,
        "protocol_function": miller_selfridge_rabin_test_protocoll,
        "prob_test": True,
        "number_type": "large",
        "testgroup": "Probabilistische Tests",
        "plotgroup": "Probabilistische Tests",
        "runtime_theoretical_fn": lambda n: log(n)**4
    },
    "Solovay-Strassen": {
        "runtime_function": solovay_strassen_test,
        "protocol_function": solovay_strassen_test_protocoll,
        "prob_test": True,
        "number_type": "large",
        "testgroup": "Probabilistische Tests",
        "plotgroup": "Probabilistische Tests",
        "runtime_theoretical_fn": lambda n: log(n)**3
    },
    "Initial Lucas": {
        "runtime_function": initial_lucas_test,
        "protocol_function": initial_lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas",
        "testgroup": "Lucas-Tests",
        "plotgroup": "Lucas-Tests",
        "runtime_theoretical_fn": lambda n: n * log(n)**3
    },
    "Lucas": {
        "runtime_function": lucas_test,
        "protocol_function": lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas",
        "testgroup": "Lucas-Tests",
        "plotgroup": "Lucas-Tests",
        "runtime_theoretical_fn": lambda n: sqrt * log(n)**3
    },
    "Optimized Lucas": {
        "runtime_function": optimized_lucas_test,
        "protocol_function": optimized_lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas",
        "testgroup": "Lucas-Tests",
        "plotgroup": "Lucas-Tests",
        "runtime_theoretical_fn": lambda n: test_data["Optimized Lucas"][n]["other_fields"]["num_prime_factors"] * log(n)**3
    },
    "Wilson": {
        "runtime_function": wilson_criterion,
        "protocol_function": wilson_criterion_protocoll,
        "prob_test": False,
        "number_type": "small",
        "testgroup": "Langsame Tests",
        "plotgroup": "Langsame Tests",
        "runtime_theoretical_fn": lambda n: n * log(n)**2
    },
    "AKS04": {
        "runtime_function": aks04_test,
        "protocol_function": aks04_test_protocoll,
        "prob_test": False,
        "number_type": "small",
        "testgroup": "Langsame Tests",
        "plotgroup": "Langsame Tests"
    },
    "AKS10": {
        "runtime_function": aks10_test,
        "protocol_function": aks10_test_protocoll,
        "prob_test": False,
        "number_type": "small",
        "testgroup": "Langsame Tests",
        "plotgroup": "Langsame Tests"
    },
    "Pepin": {
        "runtime_function": pepin_test,
        "protocol_function": pepin_test_protocoll,
        "prob_test": False,
        "number_type": "fermat",
        "testgroup": "Fermat-Zahlen",
        "plotgroup": "Spezielle Tests",
    },
    "Lucas-Lehmer": {
        "runtime_function": lucas_lehmer_test,
        "protocol_function": lucas_lehmer_test_protocoll,
        "prob_test": False,
        "number_type": "mersenne",
        "testgroup": "Mersenne-Zahlen",
        "plotgroup": "Spezielle Tests",
    },
    "Proth": {
        "runtime_function": proth_test,
        "protocol_function": proth_test_protocoll,
        "prob_test": False,
        "number_type": "proth",
        "testgroup": "Proth-Tests",
        "plotgroup": "Zusammengesetzte",
        "runtime_theoretical_fn": lambda n: log(n)**3
    },
    "Proth Variant": {
        "runtime_function": proth_test_variant,
        "protocol_function": proth_test_variant_protocoll,
        "prob_test": False,
        "number_type": "proth",
        "testgroup": "Proth-Tests",
        "plotgroup": "Zusammengesetzte",
        "runtime_theoretical_fn": lambda n: log(n)**3
    },
    "Pocklington": {
        "runtime_function": pocklington_test,
        "protocol_function": pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte",
        "runtime_theoretical_fn": lambda n: log(n)**3
    },
    "Optimized Pocklington": {
        "runtime_function": optimized_pocklington_test,
        "protocol_function": optimized_pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte",
        "runtime_theoretical_fn": lambda n: test_data["Optimized Pocklington"][n]["other_fields"]["num_prime_factors"] * log(n)**3
    },
    "Optimized Pocklington Variant": {
        "runtime_function": optimized_pocklington_test_variant,
        "protocol_function": optimized_pocklington_test_variant_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte",
        "runtime_theoretical_fn": lambda n: test_data["Optimized Pocklington"][n]["other_fields"]["num_prime_factors"] * log(n)**3
    },
    "Generalized Pocklington": {
        "runtime_function": generalized_pocklington_test,
        "protocol_function": generalized_pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington",
        "testgroup": "Pocklington-Tests",
        "plotgroup": "Zusammengesetzte",

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
        "testgroup": "Rao",
        "plotgroup": "Zusammengesetzte"
    }
}



# Konfiguration für die Tests erzeugen (inkl. Wiederholungen & Spezial-Zahlentyp)
def get_test_config(include_tests=None, prob_test_repeats=None, global_seed: int | None = None):
    global TEST_CONFIG

    if include_tests is None:
        include_tests = list(TEST_CONFIG.keys())

    if prob_test_repeats is None:
        prob_test_repeats = default_repeats

    config = {}

    for test in include_tests:
        cfg = TEST_CONFIG[test].copy()
        if cfg["prob_test"]:
            idx = prob_tests.index(test)
            k = prob_test_repeats[idx]
            cfg["prob_test_repeats"] = k
            cfg["runtime_function"] = partial(cfg["runtime_function"], k=k, seed=global_seed)
            cfg["protocol_function"] = partial(cfg["protocol_function"], k=k, seed=global_seed)
            cfg["label"] = f"{test} (k = {k})"
        else:
            cfg["runtime_function"] = partial(cfg["runtime_function"], seed=global_seed)
            cfg["protocol_function"] = partial(cfg["protocol_function"], seed=global_seed)
            cfg["label"] = test

        config[test] = cfg

    return config



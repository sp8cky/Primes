from src.primality.tests import *
from src.primality.test_protocoll import *
from functools import partial

# Wahrscheinlichkeitsbasierte Tests und Standardwiederholungen
default_repeats = [3, 3, 3]
prob_tests = ["Fermat", "Miller-Rabin", "Solovay-Strassen"]

# Zuordnung von Tests zu Funktionen und Metadaten
test_function_mapping = {
    "Fermat": {
        "runtime_function": partial(fermat_test, k=3),
        "protocol_function": partial(fermat_test_protocoll, k=3),
        "prob_test": True,
        "number_type": "general"
    },
    "Miller-Rabin": {
        "runtime_function": partial(miller_selfridge_rabin_test, k=3),
        "protocol_function": partial(miller_selfridge_rabin_test_protocoll, k=3),
        "prob_test": True,
        "number_type": "general"
    },
    "Solovay-Strassen": {
        "runtime_function": partial(solovay_strassen_test, k=3),
        "protocol_function": partial(solovay_strassen_test_protocoll, k=3),
        "prob_test": True,
        "number_type": "general"
    },
    "Initial Lucas": {
        "runtime_function": initial_lucas_test,
        "protocol_function": initial_lucas_test_protocoll,
        "prob_test": False,
        "number_type": "general"
    },
    "Lucas": {
        "runtime_function": lucas_test,
        "protocol_function": lucas_test_protocoll,
        "prob_test": False,
        "number_type": "general"
    },
    "Optimized Lucas": {
        "runtime_function": optimized_lucas_test,
        "protocol_function": optimized_lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"
    },
    "Pepin": {
        "runtime_function": pepin_test,
        "protocol_function": pepin_test_protocoll,
        "prob_test": False,
        "number_type": "fermat"
    },
    "Lucas-Lehmer": {
        "runtime_function": lucas_lehmer_test,
        "protocol_function": lucas_lehmer_test_protocoll,
        "prob_test": False,
        "number_type": "mersenne"
    },
    "Proth": {
        "runtime_function": proth_test,
        "protocol_function": proth_test_protocoll,
        "prob_test": False,
        "number_type": "proth"
    },
    "Proth Variant": {
        "runtime_function": proth_test_variant,
        "protocol_function": proth_test_variant_protocoll,
        "prob_test": False,
        "number_type": "proth"
    },
    "Pocklington": {
        "runtime_function": pocklington_test,
        "protocol_function": pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"
    },
    "Optimized Pocklington": {
        "runtime_function": optimized_pocklington_test,
        "protocol_function": optimized_pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"
    },
    "Optimized Pocklington Variant": {
        "runtime_function": optimized_pocklington_test_variant,
        "protocol_function": optimized_pocklington_test_variant_protocoll,
        "prob_test": False,
        "number_type": "lucas"
    },
    "Generalized Pocklington": {
        "runtime_function": generalized_pocklington_test,
        "protocol_function": generalized_pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"
    },
    "Grau": {
        "runtime_function": grau_test,
        "protocol_function": grau_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"
    },
    "Grau Probability": {
        "runtime_function": grau_probability_test,
        "protocol_function": grau_probability_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"
    },
    "Ramzy": {
        "runtime_function": ramzy_test,
        "protocol_function": ramzy_test_protocoll,
        "prob_test": False,
        "number_type": "ramzy"
    },
    "Rao": {
        "runtime_function": rao_test,
        "protocol_function": rao_test_protocoll,
        "prob_test": False,
        "number_type": "rao"
    },
    "Wilson": {
        "runtime_function": wilson_criterion,
        "protocol_function": wilson_criterion_protocoll,
        "prob_test": False,
        "number_type": "prime"
    },
    "AKS": {
        "runtime_function": aks_test,
        "protocol_function": aks_test_protocoll,
        "prob_test": False,
        "number_type": "general"
    }
}

# Konfiguration für die Tests erzeugen (inkl. Wiederholungen & Spezial-Zahlentyp)
def get_test_config(include_tests=None, prob_test_repeats=None):
    if include_tests is None:
        include_tests = list(test_function_mapping.keys())

    if prob_test_repeats is None:
        prob_test_repeats = default_repeats

    test_config = {}
    for test in include_tests:
        cfg = test_function_mapping[test].copy()
        if cfg["prob_test"]:
            idx = prob_tests.index(test)
            k = prob_test_repeats[idx]
            cfg["prob_test_repeats"] = k
            print(f"Konfiguriere {test} mit k = {k}")
            cfg["label"] = f"{test} (k = {k})"
            print(f"cfg['label'] = {cfg['label']}")
        else:
            cfg["label"] = test

        test_config[test] = cfg

    return test_config
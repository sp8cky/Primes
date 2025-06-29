from src.primality.tests import *
from src.primality.test_protocoll import *
from functools import partial

# Wahrscheinlichkeitsbasierte Tests und Standardwiederholungen
default_repeats = [3, 3, 3]
prob_tests = ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen"]

TEST_GROUPS = {
    "Fermat": "Probabilistische Tests",
    "Miller-Selfridge-Rabin": "Probabilistische Tests",
    "Solovay-Strassen": "Probabilistische Tests",
    
    "Initial Lucas": "Lucas-Tests",
    "Lucas": "Lucas-Tests",
    "Optimized Lucas": "Lucas-Tests",

    "Wilson": "Langsame Tests",
    "AKS": "Langsame Tests",

    "Proth": "Proth-Tests",
    "Proth Variant": "Proth-Tests",

    "Pocklington": "Pocklington-Tests",
    "Optimized Pocklington": "Pocklington-Tests",
    "Optimized Pocklington Variant": "Pocklington-Tests",
    "Generalized Pocklington": "Pocklington-Tests",
    "Grau": "Pocklington-Tests",
    "Grau Probability": "Pocklington-Tests",

    "Rao": "Rao",
    "Ramzy": "Ramzy",
   
    "Pepin": "Fermat-Zahlen",
    "Lucas-Lehmer": "Mersenne-Zahlen",
}

TEST_ORDER = ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen", "Initial Lucas", "Lucas", "Optimized Lucas", "Wilson", "AKS", "Proth", "Proth Variant", "Pocklington", "Optimized Pocklington", "Optimized Pocklington Variant", "Generalized Pocklington", "Grau", "Grau Probability", "Ramzy", "Rao",  "Pepin", "Lucas-Lehmer"]


# Zuordnung von Tests zu Funktionen und Metadaten
test_function_mapping = {
    "Fermat": {
        "runtime_function": fermat_test,
        "protocol_function": fermat_test_protocoll,
        "prob_test": True,
        "number_type": "large_prime"  # große Primzahlen sind sinnvoll
    },
    "Miller-Selfridge-Rabin": {
        "runtime_function": miller_selfridge_rabin_test,
        "protocol_function": miller_selfridge_rabin_test_protocoll,
        "prob_test": True,
        "number_type": "large_prime"  # große Zahlen, da Test für große Zahlen gedacht
    },
    "Solovay-Strassen": {
        "runtime_function": solovay_strassen_test,
        "protocol_function": solovay_strassen_test_protocoll,
        "prob_test": True,
        "number_type": "large_prime"  # ebenfalls große Zahlen sinnvoll
    },
    "Initial Lucas": {
        "runtime_function": initial_lucas_test,
        "protocol_function": initial_lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"  # allgemeine Zahlen, keine spezielle Form nötig
    },
    "Lucas": {
        "runtime_function": lucas_test,
        "protocol_function": lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"  # wie Initial Lucas
    },
    "Optimized Lucas": {
        "runtime_function": optimized_lucas_test,
        "protocol_function": optimized_lucas_test_protocoll,
        "prob_test": False,
        "number_type": "lucas"  # spezieller Lucas-Zahlenbereich (Zerlegung vorhanden)
    },
    "Wilson": {
        "runtime_function": wilson_criterion,
        "protocol_function": wilson_criterion_protocoll,
        "prob_test": False,
        "number_type": "small_prime"  # wegen Laufzeit auf kleine Primzahlen
    },
    "AKS": {
        "runtime_function": aks_test,
        "protocol_function": aks_test_protocoll,
        "prob_test": False,
        "number_type": "small_prime"  # ebenfalls wegen Laufzeit kleine Primzahlen
    },
    "Pepin": {
        "runtime_function": pepin_test,
        "protocol_function": pepin_test_protocoll,
        "prob_test": False,
        "number_type": "fermat"  # genau Fermat-Zahlen
    },
    "Lucas-Lehmer": {
        "runtime_function": lucas_lehmer_test,
        "protocol_function": lucas_lehmer_test_protocoll,
        "prob_test": False,
        "number_type": "mersenne"  # nur Mersenne-Zahlen
    },
    "Proth": {
        "runtime_function": proth_test,
        "protocol_function": proth_test_protocoll,
        "prob_test": False,
        "number_type": "proth"  # Proth-Zahlen
    },
    "Proth Variant": {
        "runtime_function": proth_test_variant,
        "protocol_function": proth_test_variant_protocoll,
        "prob_test": False,
        "number_type": "proth"  # Proth-Zahlen
    },
    "Pocklington": {
        "runtime_function": pocklington_test,
        "protocol_function": pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington"  # Pocklington-Zerlegung, ähnlicher Bereich wie Lucas
    },
    "Optimized Pocklington": {
        "runtime_function": optimized_pocklington_test,
        "protocol_function": optimized_pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington"  # wie Pocklington
    },
    "Optimized Pocklington Variant": {
        "runtime_function": optimized_pocklington_test_variant,
        "protocol_function": optimized_pocklington_test_variant_protocoll,
        "prob_test": False,
        "number_type": "pocklington"  # wie Pocklington
    },
    "Generalized Pocklington": {
        "runtime_function": generalized_pocklington_test,
        "protocol_function": generalized_pocklington_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington"  # wie Pocklington
    },
    "Grau": {
        "runtime_function": grau_test,
        "protocol_function": grau_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington"  # ebenfalls mit Pocklington-Zerlegung
    },
    "Grau Probability": {
        "runtime_function": grau_probability_test,
        "protocol_function": grau_probability_test_protocoll,
        "prob_test": False,
        "number_type": "pocklington"  # wie Grau
    },
    "Ramzy": {
        "runtime_function": ramzy_test,
        "protocol_function": ramzy_test_protocoll,
        "prob_test": False,
        "number_type": "ramzy"  # Ramzy-Zahlen (spezieller Typ)
    },
    "Rao": {
        "runtime_function": rao_test,
        "protocol_function": rao_test_protocoll,
        "prob_test": False,
        "number_type": "rao"  # Rao-Zahlen
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
            cfg["runtime_function"] = partial(cfg["runtime_function"], k=k)
            cfg["protocol_function"] = partial(cfg["protocol_function"], k=k)
            cfg["label"] = f"{test} (k = {k})"
        else:
            cfg["label"] = test

        test_config[test] = cfg

    return test_config
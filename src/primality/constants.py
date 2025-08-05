PRIME = "prime"
COMPOSITE = "composite"
INVALID = "invalid"
NOT_APPLICABLE = "not_applicable"
VALID_RESULTS = {PRIME, COMPOSITE, INVALID, NOT_APPLICABLE}

default_repeats = [3, 3, 3]
prob_tests = ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen"]
TEST_ORDER = ["Fermat", "Miller-Selfridge-Rabin", "Solovay-Strassen", "Initial Lucas", "Lucas", "Optimized Lucas", "Wilson", "AKS", "Pepin", "Lucas-Lehmer", "Proth", "Optimized Proth", "Proth Variant", "Pocklington", "Optimized Pocklington", "Optimized Pocklington Variant", "Generalized Pocklington", "Grau", "Grau Probability", "Rao", "Ramzy"]

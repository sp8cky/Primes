import random
from sympy import primerange

def generate_random_integers(start, end, count):
    return random.sample(range(start, end), count)

def generate_primes(start, end):
    return list(primerange(start, end))

def generate_mixed_dataset(start, end, count):
    """ Erzeugt eine Mischung aus Prim- und Nicht-Primzahlen """
    values = set()
    while len(values) < count:
        n = random.randint(start, end)
        values.add(n)
    return list(values)
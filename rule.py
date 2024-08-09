from sympy import *


class Rule():
    """
    Creation of a single rule consisting of predicates and implication.
    Input:
    - predicate: sympy.Symbol
    - implicant: sympy.Symbol
    """

    def __init__(self, predicate: str, implicant: str):
        self.predicate = sympify(predicate)
        self.implicant = sympify(implicant)
        self.implication = Implies(predicate, implicant)

    def get_predicate(self):
        return self.predicate

    def get_implicant(self):
        return self.implicant

    def get_rule(self):
        return self.implication

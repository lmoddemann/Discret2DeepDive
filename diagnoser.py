from typing import Optional
import time
from sympy import *
import numpy as np


class Diag_solver():
    """
    Solver for diagnosis according its implicant booleans.

    Inputs:
    - rules: List fo rules
    - health_dict: Dictionary of implicants and its booleans whether they are OK (True) or KO (False)
    """

    def __init__(self, rules: Optional[list] = None, health_dict: Optional[dict] = None):
        if rules is None:
            rules = []
        if health_dict is None:
            health_dict = []
        self.rules = rules
        self.health_dict = health_dict
        self.off_components_cols = []
        self.case_idx = 0

    def add_health_states(self, health_states: list):
        # Check whether the dict has enough bools for all rules
        if len(health_states) != len(self.rules):
            print("Mismatch between len health states and len rules")
        else:
            self.health_dict = health_states

    def solve(self):
        """
         Solve the sat-core for inserted rules.
        -------

        # >>> from diagnoser.logic import Rule, Model
        # >>> rules = [Rule(predicates='a & b', implicant='x'), Rule(predicates='b & c', implicant='y')]
        # >>> values_dict = {'x': False, 'y': False}
        # >>> model = Model(rules, values_dict)
        # >>> is_satisfiable, causes_min, causes = model.solve_sym()
        """
        diag = []
        start = time.time()
        # combining rules to one single rule
        self.rules_complete = And(*self.health_dict)

        # check satisfiability
        causes = list(satisfiable(self.rules_complete, all_models=True))
        core_sums = [sum(core.values()) for core in causes]
        # get index where the faults are minimal
        idx_min, = np.where(core_sums == np.max(core_sums))
        min_causes = [causes[idx] for idx in idx_min]
        for cause in min_causes:
            # Append cause to diag
            diag.append([k for k, v in cause.items() if (v == False)])
        end = time.time()
        delta_time = end - start
        return diag, min_causes, causes, delta_time

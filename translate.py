import sympy as sym
from sympy import Symbol
from typing import List
from rule import Rule


SYM_TO_RULE = {
    '&': 'AND',
    '>>': 'IMPLIES',
    }


def rulestr_to_expr(rulestr: str) -> Symbol:
    """ Converts string to sympy expression"""
    # split the implicants
    predicates_str, implicant_str = rulestr.split(SYM_TO_RULE['>>'])
    # build expression
    for op_sym, op_rulestr in SYM_TO_RULE.items():
        implicant_str = implicant_str.replace(op_rulestr, op_sym)
        predicates_str = predicates_str.replace(op_rulestr, op_sym)
    expr = sym.Implies(sym.sympify(predicates_str), sym.sympify(implicant_str))
    return expr


def parse_file(filepath: str, is_sympy: bool = False) -> List[Rule]:
    """Parse File and return list of rules"""
    rules = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if is_sympy:
            expr = sym.sympify(line)
        else:  # expect form
            expr = rulestr_to_expr(line)

        rule = expr_to_rule(expr)
        rules.append(rule)
    return rules


def expr_to_rule(expr: sym.Symbol) -> Rule:
    """Expression to Rule from a sympy expression"""
    if not isinstance(expr, sym.Implies):
        raise TypeError(f'Expect Implies expression, but got {type(expr)}!')
    predicates, implicant = expr.args
    rule = Rule(predicates, implicant)
    return rule

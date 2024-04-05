from sympy.parsing.sympy_parser import parse_expr
from sympy.logic import simplify_logic
from sympy import symbols, Expr
from sympy.logic.boolalg import And, Or, Not

from typing import Tuple, Union
import numpy as np


def is_edge_satisfied(sympy_edge, true_props) -> Tuple[bool, bool]:
    """
    Given the edge and the true propositions,
    returns whether the edge is satisfied.

    returns is_satisfied, is_violated
    """
    if type(sympy_edge) == str:
        sympy_edge = parse_expr(sympy_edge.replace("!", "~"))
    expr = sympy_edge
    for prop in true_props:
        expr = expr.subs(symbols(prop), True)
    if expr == True:
        return True, False
    elif expr == False:
        return False, True
    elif str(expr)[0] == "~":
        # if the remaining symbols are all False, then the edge is satisfied
        return True, False
    else:
        return False, False

def test_edges(self_edge, out_edge, true_props) -> Tuple[bool, bool]:
    """
    Given the self and out edges, and the true proposition,
    Assumes edges start with true props, then false props, and sorted alphabetically.
    returns 
    1) whether the true proposition is in the self edge,
    2) whether the true proposition is in the out edge.
    """
    self_satisfied, self_violated = is_edge_satisfied(self_edge, true_props)
    out_satisfied, out_violated = is_edge_satisfied(out_edge, true_props)
    return self_satisfied, out_satisfied

def sample_edge(
        all_props, 
        max_props=3, 
        rng=np.random.default_rng(), 
        single_true_prop=True,
        sympy_parse=False
    ) -> Union[Tuple[Expr, Expr], Tuple[str, str]]:
    """
    Given all the propositions,
    returns all the possible edges.
    """
    all_props = sorted(list(all_props))

    # randomly select the number of propositions used
    formula_size = rng.integers(1, max_props + 1)

    # randomly select the propositions
    props_selected = rng.choice(all_props, size=formula_size, replace=False)
    out_props = []
    self_props = []

    # randomly select the propositions that are true in the out edge
    if not single_true_prop:
        # 50% chance of any prop being in the out edge
        for i in range(0, max_props):
            for prop in props_selected:
                self_props.append("~{prop}")
                if rng.random() < 0.5: # 50% chance of being in the out edge
                    out_props.append(prop)
                else:
                    out_props.append("~{prop}")
    else:
        # only select one true prop in the out edge
        selected_prop = rng.choice(props_selected, size=1)
        for prop in props_selected:
            self_props.append(f"~{prop}")
            if prop == selected_prop:
                out_props.append(prop)
            else:
                out_props.append(f"~{prop}")
    
    # combine
    out_edge = "&".join(out_props)
    self_edge = "&".join(self_props)
    if sympy_parse:
        out_edge = parse_expr(out_edge)
        self_edge = parse_expr(self_edge)
    return self_edge, out_edge

def expr_edge_to_str(edge: str) -> str:
    return edge.replace('(', '').replace(')', '').replace('~', '!').replace(' ', '')

def expr_edges_to_str(self_edge: str, out_edge: str) -> Tuple[str, str]:
    return expr_edge_to_str(self_edge), expr_edge_to_str(out_edge)

def build_edge_one_hot(edge: str, all_props: str, absence_is_false=False):
    result = np.zeros(len(all_props))
    for i, prop in enumerate(all_props):
        if f"!{prop}" in edge:
            code = -1
        elif prop in edge:
            code = 1
        else:
            # absence of code
            if absence_is_false:
                code = -1
            else:
                code = 0
        result[i] = code
    return result

if __name__ == "__main__":
    assert is_edge_satisfied("a&~b&c", "b") == (False, True)
    assert is_edge_satisfied("a&~b&c", "c") == (False, False)
    assert is_edge_satisfied("a&~b&c", "a") == (False, False)
    assert is_edge_satisfied("a&~b&c", "ac") == (True, False)
    assert is_edge_satisfied("~a&~b&~c", "") == (True, False)
    assert is_edge_satisfied("~a&~b&~c", "a") == (False, True)
    
    rng = np.random.default_rng(seed=42)
    for _ in range(20):
        print(sample_edge("abcedfg", single_true_prop=True, rng=rng, sympy_parse=True))


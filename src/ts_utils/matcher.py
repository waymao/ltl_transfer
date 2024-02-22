from typing import Mapping, Set, Tuple, List

from ltl.dfa import DFA

from copy import deepcopy
from collections import defaultdict
import networkx as nx
import numpy as np
import sympy
from ltl.ltl_utils import LTL, DFAEdge

from ts_utils.ts_policy_bank import TianshouPolicyBank
from typing import Iterable, Tuple


def match_remove_edges(
        test_dfa: nx.DiGraph, 
        train_edges: List[Tuple[str, str]],
        start_state: int, 
        goal_state: int, 
        edge_matcher: str
    ) -> Mapping[Tuple[str, str], Set[Tuple[str, str]]]:
    """
    Remove infeasible edges from DFA graph
    Optimization: construct a mapping 'test2trains' from test_edge_pair to a set of matching train_edge_pairs
    """
    test_dfa_copy = deepcopy(test_dfa)
    test2trains = defaultdict(set)
    test_edges = list(test_dfa.edges.keys())  # make a copy of edges because nx graph dict mutated in-place
    for test_edge in test_edges:
        if test_edge[0] != test_edge[1]:  # ignore self-edge
            # self edge and out edge pair to be tested
            self_edge_label: str = test_dfa.edges[test_edge[0], test_edge[0]]["edge_label"]
            out_edge_label: str = test_dfa.edges[test_edge]["edge_label"]
            test_edge_pair = (self_edge_label, out_edge_label)

            # edge leading to the failure state
            fail_edge = get_fail_edge(test_dfa_copy, test_edge[0])
            is_matched = False

            # match each potential testing outgoing edges with training outgoing edges
            for train_edge in train_edges:
                if edge_matcher == 'rigid':
                    match = match_edges(test_edge_pair, [train_edge])
                elif edge_matcher == 'relaxed':
                    match = match_edges_v2(test_edge_pair, fail_edge, [train_edge])
                if match:
                    test2trains[test_edge_pair].add(train_edge)
                    is_matched = True
            
            # remove test edge if it cannot be matched with any training edge
            if not is_matched:
                test_dfa.remove_edge(test_edge[0], test_edge[1])
                # optimization: return as soon as start and goal are not connected
                feasible_paths_node = list(nx.all_simple_paths(test_dfa, source=start_state, target=goal_state))
                if not feasible_paths_node:
                    # print("short circuit remove_infeasible_edges")
                    return None
    # return the list of training edges that can be matched with each test edge for execution
    return test2trains


def get_fail_edge(dfa, node):
    # check if the direct edge exists from 'node' to fail_state
    if bool(dfa.get_edge_data(node, -1)):
        return dfa.get_edge_data(node, -1)['edge_label']
    else:
        return 'False'


def match_edges_v2(test_edge_pair, fail_edge, train_edges):
    """
    Determine if the test_edge can be matched with any training edge
    match means training edge has an intersecting satisfaction with the target test edge,
    AND is guaranteed to not fail,
    AND does not have an intersecting satisfaction with the test self-edge
    (i.e. test edge can be more constrained than train edge, and vice versa -> more relaxed than match_edges)
    """
    # print('using relaxed match_edge_v2')
    match_bools = [match_single_edge(test_edge_pair, fail_edge, train_edge) for train_edge in train_edges]
    return np.any(match_bools)


def match_single_edge(test_edge_pair, fail_edge, train_edge_pair):
    test_self_edge, test_out_edge = test_edge_pair
    train_self_edge, train_out_edge = train_edge_pair

    # Non empty intersection of test_out_edge and train_out_edge
    c1 = is_model_match(test_out_edge, train_out_edge)
    # Non empty intersection of test_self_edge and train_self_edge
    c2 = is_model_match(test_self_edge, train_self_edge)
    # Empty intersection of train_out_edge with fail_edge
    c3 = not is_model_match(train_out_edge, fail_edge)
    # Empty intersection of train_self_edge with fail_edge
    c4 = not is_model_match(train_self_edge, fail_edge)
    # Empty intersection of train_out_edge with test_self_edge
    c5 = not is_model_match(train_out_edge, test_self_edge)

    # All these conditions must be satisfied
    return np.all([c1, c2, c3, c4, c5])


def is_model_match(formula1, formula2):
    """
    Logic Tools for model counting
    """
    formula1 = sympy.sympify(formula1.replace('!', '~'))
    formula2 = sympy.sympify(formula2.replace('!', '~'))

    sat = sympy.logic.inference.satisfiable(formula1 & formula2)
    if sat:
        return True
    else:
        return False


def match_edges(test_edge_pair: Tuple[str, str], train_edges: List[Tuple[str, str]]):
    """
    Determine if test_edge can be matched with any training_edge
    match means exact match (aka. eq) OR test_edge is less constrained than a training_edge (aka. subset)
    Note: more efficient to convert 'training_edges' before calling this function
    """
    test_self_edge, test_out_edge = test_edge_pair
    test_self_edge = sympy.simplify_logic(test_self_edge.replace('!', '~'), form='dnf')
    test_out_edge = sympy.simplify_logic(test_out_edge.replace('!', '~'), form='dnf')
    train_self_edges = [sympy.simplify_logic(pair[0].replace('!', '~'), form='dnf') for pair in train_edges]
    train_out_edges = [sympy.simplify_logic(pair[1].replace('!', '~'), form='dnf') for pair in train_edges]

    # TODO: works correctly when 'train_edges' contains only 1 train edge
    is_subset_eq_self = np.any([bool(_is_subset_eq(test_self_edge, train_self_edge)) for train_self_edge in train_self_edges])
    is_subset_eq_out = np.any([bool(_is_subset_eq(test_out_edge, train_out_edge)) for train_out_edge in train_out_edges])
    return is_subset_eq_self and is_subset_eq_out


def _is_subset_eq(test_edge, train_edge):
    """
    subset_eq match :=
    every conjunctive term of 'test_edge' can be satisfied by the same 'training_edge'

    Assume edges are in sympy and DNF
    DNF: negation can only precede a propositional variable
    e.g. ~a | b is DNF; ~(a & b) is not DNF

    https://github.com/sympy/sympy/issues/23167
    """
    if test_edge.func == sympy.Or:
        if train_edge.func == sympy.Or:  # train_edge must have equal or more args than test_edge
            return sympy.And(*[sympy.Or(*[_is_subset_eq(test_term, train_term) for test_term in test_edge.args])
                               for train_term in train_edge.args]) and \
                   sympy.And(*[sympy.Or(*[_is_subset_eq(test_term, train_term) for train_term in train_edge.args])
                               for test_term in test_edge.args])
        return sympy.Or(*[_is_subset_eq(term, train_edge) for term in test_edge.args])
    elif test_edge.func == sympy.And:
        return train_edge.func == sympy.And and sympy.And(*[term in train_edge.args for term in test_edge.args])
    else:  # Atom, e.g. a, b, c or Not
        if train_edge.func == sympy.And:
            return test_edge in train_edge.args
        else:
            return test_edge == train_edge


def dfa2graph(dfa: DFA):
    """
    Convert DFA to NetworkX graph
    """
    nodelist: Mapping[int, Mapping[int, dict]] = defaultdict(dict)
    for u, v2label in dfa.nodelist.items():
        for v, label in v2label.items():
            nodelist[u][v] = {"edge_label": label}
    return nx.DiGraph(nodelist)


def get_training_edges(policy_bank: TianshouPolicyBank) -> Tuple[Iterable[DFAEdge], Mapping[DFAEdge, List[Tuple[int, LTL]]]]:
    """
    Pair every outgoing edge that each state-centric policy have achieved during training,
    with the self-edge of the DFA progress state corresponding to the state-centric policy.
    Map each edge pair to corresponding LTLs, possibly one to many.
    """
    edges2ltls: Mapping[Tuple[str, str], Tuple[int, LTL]] = defaultdict(list)
    for i, ltl in enumerate(policy_bank.policy_ltls):
        dfa: DFA = policy_bank.dfas[i]
        dfa_state = dfa.ltl2state[ltl]
        self_edge = dfa.nodelist[dfa_state][dfa_state]
        for out_edge in policy_bank.classifiers[i].possible_edges:
            edges2ltls[(self_edge, out_edge)].append((i, ltl))
    return edges2ltls.keys(), dict(edges2ltls)

if __name__ == '__main__':
    formula = ('and',
               ('until', 'True', 'a'),
                ('and',
                 ('until', 'True', 'c'),
                 ('and',
                  ('until', 'True', 'e'),
                  ('and',
                   ('until', 'True', 'f'),
                   ('and',
                    ('until', 'True', 's'),
                    ('and',
                     ('until', ('not', 'c'), 'f'),
                     ('and',
                      ('until', ('not', 'c'), 's'),
                      ('and',
                       ('until', ('not', 'c'), 'a'),
                       ('and',
                        ('until', ('not', 'c'), 'e'),
                        ('and',
                         ('until', ('not', 'f'), 's'),
                         ('and',
                          ('until', ('not', 'f'), 'a'),
                          ('and',
                           ('until', ('not', 'f'), 'e'),
                           ('and',
                            ('until', ('not', 's'), 'a'),
                            ('and',
                             ('until', ('not', 's'), 'e'),
                             ('until', ('not', 'a'), 'e')))))))))))))))
    train_self_edge = '!a&!g'
    train_out_edge = 'a&!g'
    train_edges = [(train_self_edge, train_out_edge)]
    dfa = DFA(formula)
    dfa_graph = dfa2graph(dfa)
    test2train = match_remove_edges(dfa_graph, train_edges, 0, 5, 'relaxed')
    print(test2train)


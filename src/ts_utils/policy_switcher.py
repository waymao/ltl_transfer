from collections import defaultdict
from typing import Mapping, Optional, Tuple, Union

from ts_utils.matcher import dfa2graph
from .ts_policy_bank import TianshouPolicyBank
from ltl.dfa import DFA
import networkx as nx

class PolicySwitcher:
    def __init__(self, 
                 pb: TianshouPolicyBank, 
                 test2trains: Mapping[Union[tuple, str], list],
                 edges2ltls: Mapping[str, list], 
                 ltl_task
        ):
        self.pb = pb
        self.edge2ltls = edges2ltls
        self.test2trains = test2trains
        self.exclude_list = defaultdict(set)
        self.curr_policy = {}

        self.dfa = DFA(ltl_task)
        self.dfa_graph = dfa2graph(self.dfa)

        self.feasible_paths_node = list(
            nx.all_simple_paths(
                self.dfa_graph, 
                source=self.dfa.state, 
                target=self.dfa.terminal))
        self.feasible_paths_edge = [
            list(path) for path in map(nx.utils.pairwise, self.feasible_paths_node)
        ]
    
    def _compute_option_ranking(self, curr_dfa_state, env_state):
        """
        Given the current ltl state and env state, compute the optimal policy for the next step.
        """
        option2problen = {} # (ltl, edge_pair) -> (prob, len)
        for feasible_path_node, feasible_path_edge in zip(self.feasible_paths_node, self.feasible_paths_edge):
            # for each feasible path, find the current position and the next edge
            if curr_dfa_state in feasible_path_node:
                # gather test edge info for the current path
                curr_pos_in_path = feasible_path_node.index(curr_dfa_state)  # current position on this path
                test_edge = feasible_path_edge[curr_pos_in_path] # next edge
                test_self_edge = self.dfa_graph.edges[test_edge[0], test_edge[0]]["edge_label"]  # self_edge label
                test_out_edge = self.dfa_graph.edges[test_edge]["edge_label"]  # get boolean formula for out edge
                test_edge_pair = (test_self_edge, test_out_edge)

                # for each matched training edge, find the probability
                for train_self_edge, train_out_edge in self.test2trains[test_edge_pair]:
                    for ltl in self.edge2ltls[(train_self_edge, train_out_edge)]:
                        if ltl not in self.exclude_list[(train_self_edge, train_out_edge)]:
                            ltl_id = self.pb.policy2id[ltl]
                            prob, len = self.pb.classifiers[ltl_id].predict(env_state)
                            option2problen[(ltl, train_self_edge, train_out_edge)] = prob, len
        return option2problen

    def get_best_policy(self, curr_dfa_state, env_state):
        """
        Given current env state and dfa state, 
        get the corresponding edge pair, ltl, and the best policy to execute.
        """
        option2problen = self._compute_option_ranking(curr_dfa_state, env_state)
        if option2problen == {}:
            return None, None
        else:
            # sort through the set and return the best policy in ascending order.
            # per python tuple comparison, compare prob first, then len.
            # return the policy with the highest probability and the shortest length.
            (ltl, train_self_edge, train_out_edge), (prob, len) = sorted(option2problen.items(), key=lambda x: (-x[1][0], x[1][1]), reverse=True)[0]
            return (train_self_edge, train_out_edge), ltl, \
                   self.pb.policies[self.pb.policy2id[ltl]]
    
    def exclude_policy(self, edge: Tuple[str, str], ltl: str):
        self.exclude_list[edge].add(ltl)
    
    def reset_excluded_policy(self, edge: Optional[Tuple[str, str]] = None):
        """
        Reset the policy exclusion list. If edge is None, reset all exclusion lists.
        """
        if edge is None:
            self.exclude_list = defaultdict(set)
        else:
            self.exclude_list[edge] = set()

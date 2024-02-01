from typing import List, Mapping
from . import ltl_progression


class DFA:
    def __init__(self, ltl_formula, init_dfa_state=None):
        # Progressing formula
        initial_state, accepting_states, ltl2state, edges = ltl_progression.get_dfa(ltl_formula)
        # Set DFA initial state if given
        if init_dfa_state:
            initial_state = ltl2state[init_dfa_state]
        # setting the DFA
        self.formula: tuple       = ltl_formula
        self.state: int           = initial_state    # initial state id
        self.terminal: List[int]  = accepting_states # list of terminal states
        self.ltl2state: Mapping[tuple, int] = ltl2state        # dictionary from ltl to state
        self.state2ltl = {v: k for k, v in self.ltl2state.items()}
        # Adding the edges
        self.nodelist: Mapping[int, Mapping[int, str]] = {}
        for v1, v2, label in edges:
            if v1 not in self.nodelist:
                self.nodelist[v1] = {}
            self.nodelist[v1][v2] = label

    def progress(self, true_props):
        """
        This method progress the DFA and returns False if we fall from it
        """
        self.state = self._get_next_state(self.state, true_props)

    def _get_next_state(self, v1, true_props):
        for v2 in self.nodelist[v1]:
            if _evaluate_DNF(self.nodelist[v1][v2], true_props):
                return v2
        return -1  # we broke the LTL :/

    def progress_LTL(self, ltl_formula, true_props):
        """
        Returns the progression of 'ltl_formula' given 'true_props'
        Special cases:
            - returns 'None' if 'ltl_formula' is not part of this DFA
            - returns 'False' if the formula is broken by 'true_props'
            - returns 'True' if the terminal state is reached
        """
        if ltl_formula not in self.ltl2state:
            raise NameError('ltl formula ' + ltl_formula + " is not part of this DFA")
        return self.get_LTL(self._get_next_state(self.ltl2state[ltl_formula], true_props))

    def in_terminal_state(self):
        return self.state in self.terminal

    def get_LTL(self, s=None):
        if s is None: s = self.state
        ltl = self.state2ltl[s]
        if type(ltl) == str:
            return ltl
        else:
            return tuple(self.state2ltl[s])

    def is_game_over(self):
        """
        For now, I'm considering gameover if the agent hits a DFA terminal state or if it falls from it
        """
        return self.in_terminal_state() or self.state == -1

    def __str__(self):
        aux = []
        for v1 in self.nodelist:
            aux.extend([str((v1, v2, self.nodelist[v1][v2])) for v2 in self.nodelist[v1]])
        return "\n".join(aux)


"""
Evaluates 'formula' assuming 'true_props' are the only true propositions and the rest are false.
e.g. _evaluate_DNF("a&b|!c&d","d") returns True
"""
def _evaluate_DNF(formula, true_props):
    # ORs
    if "|" in formula:
        for f in formula.split("|"):
            if _evaluate_DNF(f, true_props):
                return True
        return False
    # ANDs
    if "&" in formula:
        for f in formula.split("&"):
            if not _evaluate_DNF(f, true_props):
                return False
        return True
    # NOT
    if formula.startswith("!"):
        return not _evaluate_DNF(formula[1:], true_props)

    # Base cases
    if formula == "True":  return True
    if formula == "False": return False
    return formula in true_props

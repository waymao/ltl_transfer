from gymnasium import Env
from ltl.dfa import DFA
from abc import ABC, abstractmethod

class BaseGame(Env):
    dfa: DFA

    @abstractmethod
    def get_true_propositions() -> str:
        pass

    @abstractmethod
    def get_LTL_goal() -> str:
        pass

# result

from typing import Mapping, Tuple, List
import json
import gzip
import numpy as np


class Classifier:
    def __init__(self):
        self.possible_edges = set()
        pass

    def predict(self, x) -> Tuple[float, float]:
        """
        returns success rate and length.
        """
        pass

    def add_data(self, state, result):
        # To work on later
        pass

    def save(self, path, id):
        pass

    def load(self, path, id):
        pass

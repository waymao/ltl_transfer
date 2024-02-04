# result

from typing import Mapping, Tuple
import json
import gzip


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


class NaiveMatcher(Classifier):
    def __init__(self, ratio=0.75, distance_threshold=0.1):
        self.ratio = ratio
        self.distance_threshold = distance_threshold
        self.data = {}
        self.possible_edges = set()

    def predict(self, point) -> Tuple[float, float]:
        # returns success rate and length.
        x, y, angle = point
        # find the nearest neighbor
        nearest = None
        nearest_distance = float('inf')
        for loc, val in self.data.items():
            distance = (x - loc[0])**2 + (y - loc[1])**2
            if distance < nearest_distance:
                nearest = loc
                nearest_distance = distance
        data = self.data[nearest]
        return int(data["success"]), data["steps"]

    def load(self, path, id):
        file_path = f"{path}/classifier/policy{id}_status.json.gz"
        with gzip.open(file_path, 'rt', encoding='UTF-8') as f:
            data: Mapping[str, dict] = json.load(f)
        # post-process data
        # split "x, y, angle" into a tuple of floats
        self.data = {
            tuple(
                [float(item) for item in loc.split(", ")]
            ): val
            for loc, val in data.items()
        }
        for loc in self.data:
            self.possible_edges.add(self.data[loc]['edge'])

def test_load_knn():
    path = "/home/wyc/data/shared/ltl-transfer-ts/results/miniworld_simp_no_vis_minecraft/mixed_p1.0/lpopl_dsac/map13/0/alpha=0.03/"
    matcher = NaiveMatcher()
    matcher.load(path, 0)

if __name__ == "__main__":
    test_load_knn()
    print("Done!")

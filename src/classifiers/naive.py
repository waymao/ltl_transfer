import cProfile
import os
from typing import Mapping, Tuple, List
import json
import gzip
import numpy as np
from .classifier import Classifier


class RadiusMatcher(Classifier):
    def __init__(self, distance_threshold=.5):
        self.distance_threshold = distance_threshold
        self.data = {}
        self.possible_edges = set()
        self.locs = []


    def group_gather_data(self, locs: List[List]) -> Mapping[str, Tuple[float, float]]:
        """
        With 
        """
        results: Mapping[str, List[float]] = {}
        for loc in locs:
            # print(tuple(loc)) # uncomment to test distance_threshold
            data = self.data[tuple(loc)]
            edges = (data.get('self_edge', None), data['edge'])
            # print(data)  # uncomment to test distance_threshold
            if edges not in results:
                results[edges] = [1, data["steps"]]
            else:
                results[edges][0] += 1
                results[edges][1] += data["steps"]
        final_result: Mapping[str, Tuple[float, float]] = {}
        for key, val in results.items():
            hit_prob = val[0] / len(locs)
            mean_len = val[1] / val[0] if val[0] > 0 else 999999
            final_result[key] = (hit_prob, mean_len)
        return final_result
    

    def predict(self, point) -> Mapping[str, Tuple[float, float]]:
        # returns success rate and length.
        x, y, angle = point

        # gather information
        all_point_loc = np.array(list(self.data.keys()))
        all_point_xy_N2 = all_point_loc[:, :2]
        all_point_angle_N = all_point_loc[:, 2]
        point = np.array([[x, y]])

        # compute euclidean distance
        distance_sq_N = np.sum((all_point_xy_N2 - point)**2, axis=1)
        # compute angle difference https://stackoverflow.com/questions/1878907/
        angle_diff = np.abs((all_point_angle_N - angle + 180) % 360 - 180) 
        
        # ignore data points with angle difference > 60
        distance_sq_N[angle_diff > 60] = float('inf')

        best_items_index = np.where(distance_sq_N + angle_diff / 100 < self.distance_threshold)
        locs = all_point_loc[best_items_index] # N x dim
        return self.group_gather_data(list(locs))

    def load(self, path, id, seed, rollout_method="random"):
        file_path = os.path.join(
            path,
            "classifier", 
            f"{rollout_method}_seed{seed}_det_eval",
            f"policy{id}_rollout.json.gz"
        )
        with gzip.open(file_path, 'rt', encoding='UTF-8') as f:
            data: Mapping[str, dict] = json.load(f)
        # post-process data
        # split "x, y, angle" into a tuple of floats
        self.data = {
            tuple(
                [float(item) for item in loc.split(", ")]
            ): val
            for loc, val in data['results'].items()
        }
        for loc in self.data:
            self.possible_edges.add(self.data[loc]['edge'])
        self.locs = np.array(list(self.data.keys()))

class KNNMatcher(RadiusMatcher):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def predict_n(self, point) -> Tuple[str, float, float]:
        # returns success rate and length.
        x, y, angle = point

        # gather information
        all_point_loc = self.locs
        all_point_xy_N2 = all_point_loc[:, :2]
        all_point_angle_N = all_point_loc[:, 2]
        point = np.array([[x, y]])

        # compute euclidean distance
        distance_sq_N = np.sum((all_point_xy_N2 - point)**2, axis=1)
        # compute angle difference https://stackoverflow.com/questions/1878907/
        angle_diff = np.abs((all_point_angle_N - angle + 180) % 360 - 180) 
        
        # ignore data points with angle difference > 60
        distance_sq_N[angle_diff > 60] = float('inf')

        # find the nearest neighbor
        nearest_idx = np.argpartition(distance_sq_N + angle_diff / 100, self.k)[:self.k]
        return self.group_gather_data(list(all_point_loc[nearest_idx]))


def test_load_knn():
    import os
    path = os.environ['HOME'] + "/data/shared/ltl-transfer-ts/results/miniworld_simp_no_vis_minecraft/mixed_p1.0/lpopl_dsac/map17/0/alpha=0.03/"
    matcher = KNNMatcher()
    matcher.load(path, 0)
    point = np.array([3.5, 3.5, 60])
    for i in range(10000):
        matcher.predict(point)

if __name__ == "__main__":
    # test_load_knn()
    cProfile.run("test_load_knn()", filename="transfer_prof2.prof")
    print("Done!")

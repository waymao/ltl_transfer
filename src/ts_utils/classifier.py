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


class NaiveMatcher(Classifier):
    def __init__(self, distance_threshold=.5):
        self.distance_threshold = distance_threshold
        self.data = {}
        self.possible_edges = set()


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
                results[edges] = [int(data["success"]), data["steps"]]
            else:
                results[edges][0] += int(data["success"])
                results[edges][1] += data["steps"]
        final_result: Mapping[str, Tuple[float, float]] = {}
        for key, val in results.items():
            final_result[key] = tuple([item / len(locs) for item in val])
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


    def predict_one(self, point) -> Tuple[str, float, float]:
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

        # find the nearest neighbor
        nearest_idx = np.argmin(distance_sq_N + angle_diff / 100)

        # find the nearest neighbor, naive approach
        # nearest = None
        # nearest_distance = float('inf')
        # for loc in self.data.keys():
        #     distance = (x - loc[0])**2 + (y - loc[1])**2
        #     if distance < nearest_distance:
        #         nearest = loc
        #         nearest_distance = distance

        data = self.data[tuple(all_point_loc[nearest_idx])]
        return (data.get('self_edge', None), data['edge']), int(data["success"]), data["steps"]

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
    import os
    path = os.environ['HOME'] + "/data/shared/ltl-transfer-ts/results/miniworld_simp_no_vis_minecraft/sequence_p1.0/lpopl_dsac/map13/0/alpha=0.03/"
    matcher = NaiveMatcher()
    matcher.load(path, 0)
    print(matcher.predict([3, 3, 0]))

if __name__ == "__main__":
    test_load_knn()
    print("Done!")

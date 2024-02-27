from .classifier import Classifier
from typing import Mapping, Tuple, List
import json
import gzip
import numpy as np
from .classifier import Classifier
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch


CLASSIFIER_HIDDENS_NEURONS = 512
BATCH_SIZE = 64
SHUFFLE = True


def get_model(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, CLASSIFIER_HIDDENS_NEURONS),
        nn.ReLU(inplace=True),
        # nn.Linear(CLASSIFIER_HIDDENS_NEURONS, CLASSIFIER_HIDDENS_NEURONS),
        # nn.ReLU(inplace=True),
        nn.Linear(CLASSIFIER_HIDDENS_NEURONS, out_features),
        # nn.Softmax(dim=1)
    )


def calc_accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean()


def eval_test_outcome(model: nn.Module, dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        for X, _, y_outcome in dataloader:
            y_hat = model(X)
            total_loss += F.cross_entropy(y_hat, y_outcome)
            total_acc += calc_accuracy(y_hat, y_outcome)
    return total_loss / len(dataloader), total_acc / len(dataloader)

def eval_test_len(model: nn.Module, dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():
        total_loss = 0
        for X, y_len, _ in dataloader:
            y_hat = model(X)
            total_loss += F.mse_loss(y_hat.squeeze(), y_len)
    return total_loss / len(dataloader)

class NNClassifier(Classifier):
    def __init__(self, seed=42):
        self.possible_edges = {} # dicts are ordered asof python3.7
        self.edge_count = 0
        self.seed = 42
        self.outcome_model = None

    def predict(self, x) -> Tuple[float]:
        if self.outcome_model is None:
            raise ValueError("Module not initialized")
        with torch.no_grad():
            x_tensor = torch.tensor([x], dtype=torch.float32)
            x_tensor = (x_tensor - self.x_mean) / self.x_std
            predicted_outcome = torch.softmax(self.outcome_model(x_tensor), dim=1)
            predicted_len = self.len_model(x_tensor)
            return {
                edge: (
                    predicted_outcome[0][id].item(), 
                    predicted_len[0].item()
                )
                for edge, id in self.possible_edges.items()
            }
    
    def save(self, path, id):
        file_path = f"{path}/classifier/policy{id}_nn_classifier.pth.gz"
        with gzip.open(file_path, 'wb') as f:
            state_dict = {
                "state_size": self.x_mean.shape[0],
                "possible_edges": self.possible_edges,
                "edge_count": self.edge_count,
                "mean": self.x_mean,
                "std": self.x_std,
                "outcome_model": self.outcome_model.state_dict(),
                "len_model": self.len_model.state_dict()
            }
            torch.save(state_dict, f)
    
    def load(self, path, id):
        file_path = f"{path}/classifier/policy{id}_nn_classifier.pth.gz"
        with gzip.open(file_path, 'rb') as f:
            state_dict = torch.load(f)
            self.possible_edges = state_dict["possible_edges"]
            self.edge_count = state_dict["edge_count"]
            self.x_mean = state_dict["mean"]
            self.x_std = state_dict["std"]
            in_features = state_dict["state_size"]
            self.outcome_model = get_model(in_features, self.edge_count)
            self.outcome_model.load_state_dict(state_dict["outcome_model"])
            self.len_model = get_model(in_features, 1)
            self.len_model.load_state_dict(state_dict["len_model"])


    def load_raw_data(self, path, id, rollout_method="random"):
        file_path = f"{path}/classifier/policy{id}_{rollout_method}_rollout.json.gz"
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
            data = self.data[loc]
            if (data['self_edge'], data['edge']) not in self.possible_edges:
                self.possible_edges[(data['self_edge'], data['edge'])] = self.edge_count
                self.edge_count += 1
        print("Edge count:", self.edge_count)
        print("Possible Edges:", self.possible_edges)
        
        # pre-processing for ML
        X_raw = []
        len_raw = []
        outcome_raw = []
        for loc, val in self.data.items():
            X_raw.append(np.array(loc))
            len_raw.append(val['steps'])
            outcome_raw.append(self.possible_edges[(val['self_edge'], val['edge'])])
        X = torch.tensor(np.array(X_raw), dtype=torch.float32)
        # normalize
        self.x_mean = X.mean(dim=0)
        self.x_std = X.std(dim=0)
        X = (X - self.x_mean) / self.x_std 
        len_y = torch.tensor(len_raw, dtype=torch.float32)
        outcome_y = torch.tensor(outcome_raw, dtype=torch.int64)
        self.dataset = TensorDataset(X, len_y, outcome_y)
        
        # defining the network
        in_features = X.shape[1]
        out_features = self.edge_count
        self.outcome_model = get_model(in_features, out_features)
        self.len_model = get_model(in_features, 1)

    def train(self, verbose=False):
        ds_train, ds_test = random_split(self.dataset, [.8, .2], torch.Generator().manual_seed(self.seed))
        optim_outcome = Adam(params=self.outcome_model.parameters(), lr=1e-2)
        optim_len = Adam(params=self.len_model.parameters(), lr=1e-2)
        lr_scheduler_outcome = torch.optim.lr_scheduler.StepLR(optim_outcome, step_size=300, gamma=0.2)
        lr_scheduler_len = torch.optim.lr_scheduler.StepLR(optim_len, step_size=300, gamma=0.2)
        dataloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
        num_iter = len(dataloader)
        #training outcome
        acc_max = 0
        for iter in range(1500):
            total_loss_outcome = 0
            total_acc = 0
            for batch_x, batch_len_y, batch_outcome_y in dataloader:
                optim_outcome.zero_grad()
                outcome_y_hat = self.outcome_model.forward(batch_x)
                loss_outcome = F.cross_entropy(outcome_y_hat, batch_outcome_y)
                loss_outcome.backward()
                total_loss_outcome += loss_outcome.item()
                total_acc += calc_accuracy(outcome_y_hat, batch_outcome_y)
                optim_outcome.step()
            loss_test, acc_test = eval_test_outcome(self.outcome_model, ds_test)
            lr_scheduler_outcome.step()
            if (iter + 1) % 100 == 0 or (verbose and iter < 5):
                print(f"    [outcome] iter {iter + 1} loss_outcome_train: {total_loss_outcome/num_iter} acc_train: {total_acc/num_iter} " + \
                    f"loss_test: {loss_test} acc_test: {acc_test} " + \
                    f"lr: {lr_scheduler_outcome.get_last_lr()}")
            if acc_test > 0.95 and acc_max - acc_test > 0.01:
                print("Early stopping.")
                print(f"    [outcome] iter {iter + 1} loss_outcome_train: {total_loss_outcome/num_iter} acc_train: {total_acc/num_iter} " + \
                    f"loss_test: {loss_test} acc_test: {acc_test} " + \
                    f"lr: {lr_scheduler_outcome.get_last_lr()}")
                break
            acc_max = max(acc_test, acc_max)

        # training length
        for iter in range(1000):
            total_loss_len = 0
            for batch_x, batch_len_y, batch_outcome_y in dataloader:
                optim_len.zero_grad()
                len_y_hat = self.len_model.forward(batch_x)
                loss_len = F.mse_loss(len_y_hat.squeeze(), batch_len_y)
                loss_len.backward()
                total_loss_len += loss_len.item()
                optim_len.step()
            lr_scheduler_len.step()
            loss_test = eval_test_len(self.len_model, ds_test)
            if (iter + 1) % 100 == 0 or (verbose and iter < 5):
                print(f"    [length]  iter {iter + 1} loss_len: {total_loss_len/num_iter} loss_len_test: {loss_test} lr: {lr_scheduler_len.get_last_lr()}")

def test_load_classifier():
    import os
    path = os.environ['HOME'] + "/data/shared/ltl-transfer-ts/results/miniworld_simp_no_vis_minecraft/sequence_p1.0/lpopl_dsac/map13/0/alpha=0.03/"
    matcher = NNClassifier()
    matcher.load_raw_data(path, 0)
    print("predicted result before train:", matcher.predict([3, 3, 0]))
    matcher.train()
    print("predicted result:", matcher.predict([3, 3, 0]))

if __name__ == "__main__":
    test_load_classifier()
    print("Done!")


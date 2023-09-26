from typing import List
from torch import nn

def get_MLP(
        num_features: int, 
        num_actions: int, 
        hidden_layers: List[int],
        use_relu=True
    ):
        layers = []
        last_input = num_features
        # first few layers except for the last one
        for layer_neurons in hidden_layers:
            layers.append(nn.Linear(last_input, layer_neurons))
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            last_input = layer_neurons
        layers.append(nn.Linear(last_input, num_actions))
        return nn.Sequential(layers)

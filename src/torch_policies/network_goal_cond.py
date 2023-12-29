from typing import List
from torch import nn
import torch
import numpy as np
from copy import deepcopy

from .network import cnn_init_weights, init_weights, init_weights_fanin

def get_MLP(
        num_features: int, 
        num_actions: int, 
        hidden_layers: List[int],
        use_relu=True,
        final_layer_softmax=False,
        output_init_scale=np.sqrt(2),
        init_method="orthogonal",
        device="cpu"
    ) -> nn.Module:
        layers = []
        last_input = num_features
        # first few layers except for the last one
        for layer_neurons in hidden_layers:
            linear_layer = nn.Linear(last_input, layer_neurons)

            # init
            if init_method == "orthogonal":
                init_weights(linear_layer)
            elif init_method == "fanin":
                init_weights_fanin(linear_layer)
            
            layers.append(linear_layer)
            
            # add relu when needed
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            last_input = layer_neurons

        last_layer = nn.Linear(last_input, num_actions)
        
        # init last layer
        if init_method is not None:
            init_weights(last_layer, output_init_scale)
        layers.append(last_layer)
        if final_layer_softmax:
             layers.append(nn.Softmax(-1))
        network = nn.Sequential(*layers)

        return network.to(device)

def get_CNN_preprocess(in_channels, out_dim=64, device="cpu"):
    # natural CNN
    cnn_preprocess = nn.Sequential(
                    cnn_init_weights(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0)),
                    nn.ReLU(inplace=True),
                    cnn_init_weights(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)),
                    nn.ReLU(inplace=True),
                    cnn_init_weights(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                )
    return cnn_preprocess.to(device)

class GoalCondCNNDense(nn.Module):
    def __init__(self, preprocess_net, fc_in_dim, out_dim, fc_layers=None, num_policies=1):
        super().__init__()
        self.preprocess_net = preprocess_net
        self.fc_in_dim = fc_in_dim
        self.out_dim = out_dim
        self.num_policies = num_policies
        if fc_layers == "default" or fc_layers == "auto":
            self.fc_layers = nn.Sequential(
                cnn_init_weights(nn.Linear(fc_in_dim + num_policies, 256)),
                nn.ReLU(inplace=True),
                cnn_init_weights(nn.Linear(256, 256)),
                nn.ReLU(inplace=True),
                cnn_init_weights(nn.Linear(256, out_dim)),
            )
        elif fc_layers is None:
            self.fc_layers = None
        else:
            self.fc_layers = fc_layers
    
    def forward(self, x, goal: int):
        # CNN preprocess
        x = x / 255.0
        z = self.preprocess_net(x)

        # get goal representation
        goal_arr = torch.zeros((x.shape[0], self.num_policies), device=x.device)
        goal_arr[goal] = 1.0

        # concat
        z = torch.cat([z, goal_arr], dim=-1)

        # final fully connected
        if self.fc_layers is None:
            return z
        else:
            return self.fc_layers(z)

def get_whole_GoalCNN(in_channels, out_dim, fc_layers="auto", device="cpu"):
    cnn_out_dim = 1536
    preprocess_net = get_CNN_preprocess(in_channels, device=device)
    return GoalCondCNNDense(preprocess_net, cnn_out_dim, out_dim, fc_layers=fc_layers).to(device)

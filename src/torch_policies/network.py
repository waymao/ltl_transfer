from typing import List
from torch import nn
import torch
import numpy as np

def init_weights(m, scale=np.sqrt(2), b_init_val=0.0):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            nn.init.orthogonal_(m.weight)
            m.weight.data *= scale
            m.bias.data.fill_(b_init_val)

def init_weights_fanin(m, b_init_val=0.1):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            size = m.weight.size()
            if len(size) == 2:
                fan_in = size[0]
            elif len(size) > 2:
                fan_in = np.prod(size[1:])
            bound = 1. / np.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            m.bias.data.fill_(b_init_val)

def cnn_init_weights(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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

def get_CNN_preprocess(in_channels, embed_dim=64, device="cpu"):
    # https://github.com/ezliu/dream/blob/d52204c94067641df6e6649b19abb359b87ff028/embed.py#L773
    # cnn_preprocess = nn.Sequential(
    #     nn.Conv2d(3, 32, kernel_size=8, stride=4),
    #     nn.ReLU(),
    #     nn.MaxPool2d(3),
    #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
    #     nn.ReLU(),
    #     nn.MaxPool2d(3),
    #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(3),
    #     nn.Flatten(2, -1), # N,C,H, W => N,C, H * W
    #     nn.AvgPool1d(5), # N, C, L => N, C, L'
    #     nn.Flatten(1,-1), # N, C, L' => N, L''
    # )
    # using the same architecture as Tianshou
    # in: 3 * 60 * 80
    # 
    # out: 64
    # cnn_preprocess = nn.Sequential(
    #         cnn_init_weights(nn.Conv2d(in_channels, 24, kernel_size=8, stride=4)), # out: 14 * 19
    #         nn.ReLU(inplace=True),
    #         cnn_init_weights(nn.Conv2d(24, 32, kernel_size=4, stride=2)), # out: 6 * 8
    #         nn.ReLU(inplace=True),
    #         cnn_init_weights(nn.Conv2d(32, 32, kernel_size=3, stride=1)), # 4 * 6
    #         nn.ReLU(inplace=True),
    #         nn.Flatten(),
    # )

    # this is using 
    cnn_preprocess = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=5, stride=2),
        nn.ReLU(),

        nn.Conv2d(32, 32, kernel_size=5, stride=2),
        nn.ReLU(),

        nn.Conv2d(32, 32, kernel_size=4, stride=2),
        nn.ReLU(),

        nn.Flatten(),
        nn.Linear(32 * 7 * 5, embed_dim),
    ) # out: 32 * 7 * 5
    return cnn_preprocess.to(device)

def get_CNN_Dense(preprocess_net, in_dim, out_dim, device="cpu"):
    return nn.Sequential(
        preprocess_net,
        cnn_init_weights(nn.Linear(in_dim, 64)),
        nn.ReLU(inplace=True),
        cnn_init_weights(nn.Linear(64, out_dim)),
    ).to(device)

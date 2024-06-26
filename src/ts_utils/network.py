from typing import List
from torch import nn
import torch
import numpy as np
from copy import deepcopy

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
    torch.nn.init.kaiming_normal_(layer.weight)
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

def get_CNN_preprocess(in_channels, out_dim=64, device="cpu"):
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

    # this is using https://github.com/ezliu/dream/blob/master/embed.py
    # cnn_preprocess = nn.Sequential(
    #     nn.Conv2d(in_channels, 32, kernel_size=5, stride=2),
    #     nn.ReLU(),

    #     nn.Conv2d(32, 32, kernel_size=5, stride=2),
    #     nn.ReLU(),

    #     nn.Conv2d(32, 32, kernel_size=4, stride=2),
    #     nn.ReLU(),
    #     nn.Flatten()
    # ) # out: 32 * 7 * 5

    # cnn_preprocess = nn.Sequential(
    #     nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
        
    #     nn.Conv2d(32, 64, kernel_size=3, stride=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
        
    #     nn.Conv2d(64, 128, kernel_size=3, stride=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
        
    #     nn.Flatten(),
    #     nn.Linear(5120, out_dim),
    #     nn.ReLU()
    # )
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

class CNNDense(nn.Module):
    def __init__(self, preprocess_net, fc_in_dim, out_dim, fc_layers=None):
        super().__init__()
        self.preprocess_net = preprocess_net
        self.fc_in_dim = fc_in_dim
        self.out_dim = out_dim
        if fc_layers == "default" or fc_layers == "auto":
            self.fc_layers = nn.Sequential(
                cnn_init_weights(nn.Linear(fc_in_dim, 256)),
                nn.ReLU(inplace=True),
                cnn_init_weights(nn.Linear(256, 256)),
                nn.ReLU(inplace=True),
                cnn_init_weights(nn.Linear(256, out_dim)),
            )
        elif fc_layers is None:
            self.fc_layers = None
        else:
            self.fc_layers = fc_layers
    
    def forward(self, x):
        # print(x.shape)
        # x = x.permute((0, 3, 1, 2)) # weird torch transpose
        x = x / 255.0
        z = self.preprocess_net(x)
        if self.fc_layers is None:
            return z
        else:
            return self.fc_layers(z)
    
    def deepcopy_w_preprocess(self, new_preprocess_net):
        copied_fc = deepcopy(self.fc_layers)
        copied_fc.eval()
        return CNNDense(new_preprocess_net, self.fc_in_dim, self.out_dim, copied_fc)

def get_CNN_Dense(preprocess_net, in_dim, out_dim, device="cpu"):
    return CNNDense(preprocess_net, in_dim, out_dim, fc_layers="auto").to(device)

def get_whole_CNN(in_channels, out_dim, embed_dim=1536, fc_layers="auto", device="cpu"):
    if fc_layers is None:
        return get_CNN_preprocess(in_channels, out_dim, device)
    else:
        cnn_preprocess = get_CNN_preprocess(in_channels, embed_dim, device)
    return CNNDense(cnn_preprocess, embed_dim, out_dim, fc_layers=fc_layers).to(device)

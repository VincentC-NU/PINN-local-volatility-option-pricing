import torch
import torch.nn as nn

class Baseline_MLP(nn.Module):
    def __init__(self, input_dim=7, width=64, depth=8, activation="tanh"):
        super().__init__()
        if isinstance(activation, str):
            act = nn.Tanh if activation.lower() == "tanh" else nn.ReLU
        else:
            act = activation

        layers = [nn.Linear(input_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
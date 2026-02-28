import torch
from torch import Tensor, nn


class FNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, output: bool = False) -> None:
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU(0.1))
        if output:
            for _ in range(2):
                layers.pop()
        self.fc = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
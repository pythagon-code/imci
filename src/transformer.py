from fnn import FNN
import torch
from torch import Tensor, nn


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(hidden_size, embed_dim)
        self.q_net = FNN(embed_dim, num_layers, output=True)
        self.k_net = FNN(embed_dim, num_layers, output=True)
        self.v_net = FNN(embed_dim, num_layers, output=True)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.output_net = FNN(embed_dim, num_layers)
        self.output_layer = nn.Linear(embed_dim, 1)


    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        q = self.q_net(x)
        k = self.k_net(x)
        v = self.v_net(x)
        x = self.attn(q, k, v)[0]
        x = x.mean(dim = 0)
        x = self.output_net(x)
        x = self.output_layer(x)
        return x
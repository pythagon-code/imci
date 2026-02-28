import modal
from pathlib import Path
import sys
import utils


@utils.app.function(gpu=utils.gpu, image=utils.image)
def run_module(worker_id: int) -> float:
    sys.path.append("/root/src")

    from fnn import FNN
    import torch
    from torch import Tensor, nn
    from transformer import Transformer

    fnn = FNN(hidden_size=100, num_layers=5)
    x = torch.randn(1, 100)

    trans = Transformer(hidden_size=100, num_layers=5, embed_dim=100, num_heads=10)
import sys
sys.path.append("/root/src")

import modal
from pathlib import Path
import utils



@utils.app.function(gpu=utils.gpu, image=utils.image)
def run_module(worker_id: int, config: utils.Config) -> float:
    from fnn import FNN
    import torch
    from torch import Tensor, nn
    from transformer import Transformer

    print(f"Worker {worker_id} started")

    module = FNN(
        hidden_size = config.hidden_size,
        num_layers = config.num_layers,
        output = True,
    )

    trans = Transformer(
        hidden_size = config.hidden_size,
        num_layers = config.num_layers,
        embed_dim = config.embed_dim,
        num_heads = config.num_heads,
    )

    return worker_id
import sys
sys.path.append("/root/src")

import modal
from pathlib import Path
import sys
import utils


@utils.app.function(gpu = utils.gpu, image = utils.image)
def run_reducer(config: utils.Config) -> float:
    sys.path.append("/root/src")

    from fnn import FNN
    import torch
    from torch import Tensor, nn
    from transformer import Transformer

    trans = Transformer(
        hidden_size = config.hidden_size,
        num_layers = config.num_layers,
        embed_dim = config.embed_dim,
        num_heads = config.num_heads,
    )
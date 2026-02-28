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
        num_layers = config.transformer_num_layers,
        embed_dim = config.embed_dim,
        num_heads = config.num_heads,
    )

    if config.transformer_state is not None:
        trans.load_state_dict(config.transformer_state)

    tensor_queues = utils.get_tensor_queues(config.num_workers)
    transformer_queues = utils.get_transformer_queues(config.num_workers)
    my_tensor_queue = tensor_queues[config.worker_id]


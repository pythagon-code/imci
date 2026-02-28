import sys
sys.path.append("/root/src")

import modal
from pathlib import Path
import utils


@utils.app.function(gpu = utils.gpu, image = utils.image)
def run_module(worker_id: int, config: utils.Config) -> float:
    from fnn import FNN
    import torch
    from torch import Tensor, nn
    from transformer import Transformer

    print(f"Worker {worker_id} started")

    module = FNN(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output=True,
    )

    trans = Transformer(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
    )

    if config.module_state is not None:
        module.load_state_dict(config.module_state)
        trans.load_state_dict(config.transformer_state)

    tensor_queues = utils.get_tensor_queues(config.num_modules)
    my_tensor_queue = tensor_queues[worker_id]
    transformer_queue = utils.get_transformer_queue(worker_id)

    return worker_id
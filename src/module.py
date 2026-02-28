import sys
sys.path.append("/root/src")

import modal
from pathlib import Path
import utils


@utils.app.function(gpu = utils.gpu, image = utils.image)
def run_module(config: utils.Config) -> dict:
    from fnn import FNN
    import torch
    from torch import Tensor, nn
    from torch.nn.functional import mse_loss
    from torch import optim
    from transformer import Transformer

    print(f"Worker {config.worker_id} started")

    module = FNN(
        hidden_size = config.hidden_size,
        num_layers = config.num_layers,
        output = True,
    )
    opt = optim.Adam(module.parameters(), lr=0.001)
    module_t = FNN(
        hidden_size = config.hidden_size,
        num_layers = config.num_layers,
        output = True,
    )

    trans = Transformer(
        hidden_size = config.hidden_size,
        num_layers = config.transformer_num_layers,
        embed_dim = config.embed_dim,
        num_heads = config.num_heads,
    )
    for param in trans.parameters():
        param.requires_grad = False

    if config.module_state is not None:
        module.load_state_dict(config.module_state)
        trans.load_state_dict(config.transformer_state)

    queues = utils.get_worker_queues(config.num_workers)
    my_queue = queues[config.worker_id]

    while True:
        opt.zero_grad(set_to_none=True)
        my_input = my_queue.get()
        if my_input is None:
            break
        truth = my_queue.get()
        my_output = module(my_input)
        for q in queues:
            if q != my_queue:
                q.put(my_output.cpu().detach())
        transformer_inputs = [my_output]
        for _ in range(config.num_workers - 2):
            transformer_inputs.append(my_queue.get())

        transformer_inputs = torch.stack(transformer_inputs)
        transformer_output = trans(transformer_inputs)

        loss = mse_loss(transformer_output, truth)
        loss.backward()

        opt.step()

        trans_state_dict = my_queue.get()
        trans.load_state_dict(trans_state_dict)

    return module.state_dict()
import sys

from src.utils import cpu_state_dict

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
    ).cuda()
    opt = optim.Adam(module.parameters(), lr = 0.001)
    module_t = FNN(
        hidden_size = config.hidden_size,
        num_layers = config.num_layers,
        output = True,
    ).cuda()
    module_t.load_state_dict(module.state_dict())
    for param in module_t.parameters():
        param.requires_grad_(False)

    trans = Transformer(
        hidden_size = config.hidden_size,
        num_layers = config.num_embed_layers,
        embed_dim = config.embed_dim,
        num_heads = config.num_heads,
    ).cuda()
    for param in trans.parameters():
        param.requires_grad = False

    if config.module_state is not None:
        module.load_state_dict(config.module_state)
        trans.load_state_dict(config.transformer_state)

    queues = utils.get_worker_queues(config.num_workers)
    my_queue = queues[config.worker_id]

    while True:
        opt.zero_grad(set_to_none=True)

        msg = my_queue.get()
        if msg is None:
            break
        my_input, truth = msg
        my_input = my_input.cuda()
        truth = truth.cuda()
        assert truth.shape[-1] == 1
        print("(", my_input.size(), truth.size(), ")")
        my_output = module(my_input)
        with torch.no_grad():
            my_output_t = module_t(my_input)
        for q in queues:
            if q != my_queue:
                q.put(my_output_t)
        trans_inputs = [my_output]
        for _ in range(config.num_workers - 2):
            trans_inputs.append(my_queue.get())

        trans_inputs = torch.stack(trans_inputs).cuda()
        trans_output = trans(trans_inputs)

        loss = mse_loss(trans_output, truth)
        loss.backward()

        opt.step()

        utils.polyak_update(module_t.state_dict(), module.state_dict(), 0.05)

        for param in trans.state_dict().values():
            new_value = my_queue.get()
            param.copy_(new_value)
        
        assert my_queue.get() == "End of iteration"

    return cpu_state_dict(module_t.state_dict())
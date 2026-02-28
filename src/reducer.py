import sys
sys.path.append("/root/src")

import modal
from pathlib import Path
import utils


@utils.app.function(gpu = utils.gpu, image = utils.image)
def run_reducer(config: utils.Config) -> dict:
    sys.path.append("/root/src")

    from fnn import FNN
    import torch
    from torch import Tensor, nn
    from torch.nn.functional import mse_loss
    from torch import optim
    from transformer import Transformer

    trans = Transformer(
        hidden_size = config.hidden_size,
        num_layers = config.transformer_num_layers,
        embed_dim = config.embed_dim,
        num_heads = config.num_heads,
    ).cuda()
    opt = optim.Adam(trans.parameters(), lr=0.001)
    trans_t = Transformer(
        hidden_size = config.hidden_size,
        num_layers = config.transformer_num_layers,
        embed_dim = config.embed_dim,
        num_heads = config.num_heads,
    ).cuda()
    trans_t.load_state_dict(trans.state_dict())
    for param in trans_t.parameters():
        param.requires_grad_(False)

    if config.transformer_state is not None:
        trans.load_state_dict(config.transformer_state)

    queues = utils.get_worker_queues(config.num_workers)
    my_queue = queues[config.worker_id]

    while True:
        truth = my_queue.get()
        if truth is None:
            break
        truth = truth.cuda()
        trans_inputs = []
        for _ in range(config.num_workers - 1):
            trans_inputs.append(my_queue.get())
        trans_inputs = torch.stack(trans_inputs).cuda()
        trans_output = trans(trans_inputs)
        loss = mse_loss(trans_output, truth)
        loss.backward()

        utils.polyak_update(trans_t.state_dict(), trans.state_dict(), 0.01)
        for q in queues:
            if q != my_queue:
                q.put(trans_t.state_dict())

    return utils.cpu_state_dict(trans_t.state_dict())
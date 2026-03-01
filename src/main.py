from module import run_module
from reducer import run_reducer
import torch
import utils


@utils.app.local_entrypoint()
def main() -> None:
    num_workers = 3

    configs = [utils.Config(
        num_workers = num_workers,
        worker_id = i,
    ) for i in range(num_workers - 1)]

    reducer_config = utils.Config(
        num_workers = num_workers,
        worker_id = num_workers - 1,
    )
    futures = [run_module.spawn(cfg) for cfg in configs] + [run_reducer.spawn(reducer_config)]

    queues = utils.get_worker_queues(num_workers)
    for q in queues:
        q.clear()
    output_queue = utils.get_output_queue()
    output_queue.clear()

    for _ in range(100):
        x = torch.randn((32, reducer_config.hidden_size))
        truth = 2 * x.mean(dim = 1, keepdim = True)
        queues[-1].put(truth)
        for q in queues[: -1]:
            q.put((x, truth))

        output, loss = output_queue.get()

        print(f"Loss at iteration {_}: ", loss)

    for q in queues:
        q.put(None)

    for f in futures:
        print(f.get())
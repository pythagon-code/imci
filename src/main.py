from module import run_module
from reducer import run_reducer
import torch
import utils


@utils.app.local_entrypoint()
def main() -> None:
    num_workers = 5

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
        for q in queues:
            x = torch.randn((64, reducer_config.hidden_size))
            if q != queues[-1]:
                print(_, x.shape)
                q.put(x)
            print(_, (2 * x.mean(dim = 1, keepdim = True)).shape)
            q.put(2 * x.mean(dim = 1, keepdim = True))

        output, loss = output_queue.get()

        print(f"Loss at iteration {_}: ", loss)

        # Wait for all workers to finish consuming params and "End of iteration"
        # before starting the next iteration (avoids queue.clear() racing with workers)
        ack_queue = utils.get_ack_queue()
        for _ack in range(num_workers):
            ack_queue.get()

    for q in queues:
        q.put(None)

    for f in futures:
        print(f.get())
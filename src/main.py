import modal
from module import run_module
from pathlib import Path
from reducer import run_reducer
import sys
from typing import Generator
import utils


@utils.app.local_entrypoint()
def main() -> None:
    num_workers = 10

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
        q.put(None)

    for f in futures:
        print(f.get())
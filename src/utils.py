from dataclasses import dataclass
from pathlib import Path
import modal

@dataclass
class Config:
    num_workers: int = 10
    worker_id: int = 0
    hidden_size: int = 100
    num_layers: int = 12
    embed_dim: int = 128
    num_heads: int = 8
    transformer_num_layers: int = 4
    module_state: dict | None = None
    transformer_state: dict | None = None

app = modal.App("imci")
gpu = "A100"
image = (modal.Image
         .debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "torch")
         .add_local_dir(Path(__file__).parent, remote_path="/root/src"))


def get_worker_queues(num_workers: int) -> list[modal.Queue]:
    queues = []
    for worker_id in range(num_workers):
        queues.append(
            modal.Queue.from_name(f"worker-queue-{worker_id}", create_if_missing=True))
    return queues


def polyak_update(target: dict, source: dict, tau: float) -> None:
    for k, v in source.items():
        target[k] = target[k] * (1 - tau) + v * tau
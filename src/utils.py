from dataclasses import dataclass
from pathlib import Path
import modal

@dataclass
class Config:
    num_workers: int = 2
    worker_id: int = 0
    hidden_size: int = 128
    num_layers: int = 12
    embed_dim: int = 128
    num_heads: int = 4
    num_embed_layers: int = 4
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
            modal.Queue.from_name(f"worker-queue-{worker_id}", create_if_missing = True))
    return queues


def get_output_queue() -> list[modal.Queue]:
    return modal.Queue.from_name("output-queue", create_if_missing = True)


def polyak_update(target: dict, main: dict, tau: float) -> None:
    for target_param, main_param in zip(target.values(), main.values()):
        target_param.copy_(main_param * tau + target_param * (1 - tau))


def cpu_state_dict(state_dict: dict) -> dict:
    return {k: v.cpu() for k, v in state_dict.items()}


def cuda_state_dict(state_dict: dict) -> dict:
    return {k: v.cuda() for k, v in state_dict.items()}
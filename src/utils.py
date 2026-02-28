from dataclasses import dataclass
from pathlib import Path
import modal

@dataclass
class Config:
    num_modules: int = 100
    hidden_size: int = 100
    num_layers: int = 5
    embed_dim: int = 128
    num_heads: int = 8

app = modal.App("imci")
gpu = "A100"
image = (modal.Image
         .debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "torch")
         .add_local_dir(Path(__file__).parent, remote_path="/root/src"))


def get_tensor_queue(worker_id: int):
    return modal.Queue.from_name(f"tensor-queue-{worker_id}", create_if_missing=True)


def get_transformer_queue(worker_id: int):
    return modal.Queue.from_name(f"transformer-queue-{worker_id}", create_if_missing=True)
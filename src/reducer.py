import modal
from pathlib import Path
import sys
import utils


@utils.app.function(gpu=utils.gpu, image=utils.image)
def run_reducer() -> float:
    sys.path.append("/root/src")

    from fnn import FNN
    import torch
    from torch import Tensor, nn
    from transformer import Transformer


import modal
from module import run_module
from pathlib import Path
from reducer import run_reducer
import sys
from typing import Generator
import utils

def config_range(upper: int) -> Generator[tuple[int, utils.Config]]:
    config = utils.Config()
    for i in range(upper):
        yield i, config


@utils.app.local_entrypoint()
def main():
    for x in run_module.starmap(config_range(100)):
        print("", x)
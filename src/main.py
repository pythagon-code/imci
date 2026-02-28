import modal
from pathlib import Path
import sys
import utils


@utils.app.local_entrypoint()
def main():
    for x in run_module.map(range(100)):
        print("", x)
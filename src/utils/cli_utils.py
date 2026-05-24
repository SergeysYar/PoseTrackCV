from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def add_common_args(parser: ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))


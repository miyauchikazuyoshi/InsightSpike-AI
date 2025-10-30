"""Legacy alias that forwards to pipeline.run_experiment for backwards compat."""

from __future__ import annotations

from pathlib import Path

from .config_loader import load_config
from .pipeline import run_experiment


def run_from_config(path: Path) -> None:
    cfg = load_config(path)
    run_experiment(cfg)

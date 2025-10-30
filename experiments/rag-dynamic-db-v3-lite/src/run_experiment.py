"""CLI entry point for RAG v3-lite experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path so that `insightspike` can be imported
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from .config_loader import load_config
from .pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG dynamic DB v3-lite experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    output_path = run_experiment(cfg)
    logging.info("Experiment finished. Results saved to %s", output_path)


if __name__ == "__main__":
    main()

"""CLI entry point for Exp II–III (self-contained)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE
REPO_ROOT = Path(__file__).resolve().parents[3]

for p in (REPO_ROOT, REPO_ROOT / "src", PKG_ROOT):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from .config_loader import load_config
from .pipeline import run_experiment


def _env_cloud_safe() -> None:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("INSIGHTSPIKE_LITE_MODE", "1")
    os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")
    os.environ.setdefault("INSIGHTSPIKE_LOG_DIR", str(REPO_ROOT / "results" / "logs"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "results" / "mpl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiments II–III (self-contained lite)")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    _env_cloud_safe()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    out = run_experiment(cfg)
    logging.info("Experiment finished. Results saved to %s", out)


if __name__ == "__main__":
    main()


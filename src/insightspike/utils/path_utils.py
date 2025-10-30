"""
Path utilities for resolving project-relative paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def _detect_project_root(start: Optional[Path] = None) -> Path:
    """Best-effort detection of project root by searching for markers.

    Markers: pyproject.toml, README.md, .git
    Falls back to the first parent containing a "src" directory.
    Otherwise returns the given start dir or current working directory.
    """
    here = start or Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "README.md").exists() or (
            parent / ".git"
        ).exists():
            return parent
        # Common layout: <root>/src/insightspike/...
        if (parent / "src").is_dir() and (parent / "src" / "insightspike").is_dir():
            # parent is likely the repo root
            return parent
    return Path.cwd()


_PROJECT_ROOT = _detect_project_root()


def resolve_project_relative(path_like: Union[str, Path]) -> str:
    """Resolve a path relative to the project root if it's not absolute.

    Args:
        path_like: Input path (str or Path)

    Returns:
        Absolute path string
    """
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    # Treat environment variables and user home markers
    expanded = Path(os.path.expanduser(os.path.expandvars(str(path_like))))
    if expanded.is_absolute():
        return str(expanded)
    return str((_PROJECT_ROOT / expanded).resolve())


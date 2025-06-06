"""CLI package initialization"""
# This file makes the cli directory a Python package

from .dependency_commands import deps
from .deps_typer import deps_app
from .main import app

__all__ = ["app", "deps_app", "deps"]

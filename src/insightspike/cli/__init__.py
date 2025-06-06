"""CLI package initialization"""
# This file makes the cli directory a Python package

from .deps_typer import deps_app
from .dependency_commands import deps

__all__ = ['deps_app', 'deps']

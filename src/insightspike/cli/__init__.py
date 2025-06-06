"""CLI package initialization"""
# This file makes the cli directory a Python package

from .main import app
from .deps_typer import deps_app
from .dependency_commands import deps

__all__ = ['app', 'deps_app', 'deps']

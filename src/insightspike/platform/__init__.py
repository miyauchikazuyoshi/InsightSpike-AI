"""
Platform detection and dependency management.
"""

from .dependency_resolver import *
from .detector import *
from .poetry_integration import *

__all__ = ["get_platform_info", "DependencyResolver", "PoetryDependencyManager"]

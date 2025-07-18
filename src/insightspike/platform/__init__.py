"""
Platform detection and dependency management.
"""

from .detector import *
from .dependency_resolver import *
from .poetry_integration import *

__all__ = ["get_platform_info", "DependencyResolver", "PoetryDependencyManager"]
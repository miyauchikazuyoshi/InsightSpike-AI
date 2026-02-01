"""Information Gain (IG) package.

This package provides modular components for Information Gain calculation.

Modules:
    types: EntropyMethod enum and IGResult dataclass
    methods: ImprovedEntropyMethods for advanced entropy calculation
"""

from .types import EntropyMethod, IGResult
from .methods import ImprovedEntropyMethods

__all__ = [
    "EntropyMethod",
    "IGResult",
    "ImprovedEntropyMethods",
]

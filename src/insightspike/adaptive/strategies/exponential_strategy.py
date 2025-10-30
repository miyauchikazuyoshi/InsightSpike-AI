"""Placeholder ExponentialStrategy for backward compatibility tests."""
from .base import BaseStrategy

class ExponentialStrategy(BaseStrategy):
    """Simple exponential decay variant (delegates to BaseStrategy)."""
    def __init__(self, decay: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.extra_decay = decay

    def _calculate_temperature(self, attempt: int) -> float:  # override
        base = super()._calculate_temperature(attempt)
        return base * (self.extra_decay ** attempt)

__all__ = ["ExponentialStrategy"]
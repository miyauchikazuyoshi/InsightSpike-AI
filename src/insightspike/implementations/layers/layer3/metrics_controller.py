"""Metrics selector wrapper for Layer3."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MetricsController:
    """Wrap MetricsSelector to expose delta_ged / delta_ig and info."""

    def __init__(self, config: Optional[Any] = None):
        try:
            from ...algorithms.metrics_selector import MetricsSelector  # type: ignore

            self._selector = MetricsSelector(config)
            self.delta_ged = self._selector.delta_ged
            self.delta_ig = self._selector.delta_ig
            self.info = self._selector.get_algorithm_info()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("MetricsController fallback stub: %s", exc)

            class _StubSelector:
                def delta_ged(self, *a, **k):
                    return 0.0

                def delta_ig(self, *a, **k):
                    return 0.0

                def get_algorithm_info(self):
                    return {"ged_algorithm": "stub", "ig_algorithm": "stub"}

            self._selector = _StubSelector()
            self.delta_ged = self._selector.delta_ged
            self.delta_ig = self._selector.delta_ig
            self.info = self._selector.get_algorithm_info()


__all__ = ["MetricsController"]

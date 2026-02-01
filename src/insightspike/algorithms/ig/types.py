"""Type definitions for Information Gain module.

This module contains enums and dataclasses used across the IG package.
"""

from dataclasses import dataclass
from enum import Enum


class EntropyMethod(Enum):
    """Methods for calculating entropy."""

    SHANNON = "shannon"  # Classic Shannon entropy
    CLUSTERING = "clustering"  # Clustering-based entropy using silhouette score
    MUTUAL_INFO = "mutual_info"  # Mutual information-based
    FEATURE_ENTROPY = "feature_entropy"  # Feature distribution entropy
    STRUCTURAL = "structural"  # Graph structural entropy
    DEGREE_DISTRIBUTION = "degree_distribution"  # Degree distribution entropy
    VON_NEUMANN = "von_neumann"  # Von Neumann spectral entropy


@dataclass
class IGResult:
    """Result of Information Gain calculation with metadata."""

    ig_value: float
    entropy_before: float
    entropy_after: float
    computation_time: float
    method: EntropyMethod
    sample_count: int
    feature_count: int
    approximation_used: bool = False

    @property
    def information_gain_rate(self) -> float:
        """Relative information gain rate."""
        if self.entropy_before == 0:
            return 0.0
        return self.ig_value / self.entropy_before

    @property
    def is_significant(self) -> bool:
        """Check if information gain is statistically significant."""
        return self.ig_value > 0.1 and self.sample_count >= 10


__all__ = ["EntropyMethod", "IGResult"]

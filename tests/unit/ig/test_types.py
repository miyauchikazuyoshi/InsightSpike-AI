"""Unit tests for ig.types module."""

import pytest

from insightspike.algorithms.ig.types import EntropyMethod, IGResult


class TestEntropyMethod:
    """Tests for EntropyMethod enum."""

    def test_entropy_method_values(self):
        assert EntropyMethod.SHANNON.value == "shannon"
        assert EntropyMethod.CLUSTERING.value == "clustering"
        assert EntropyMethod.MUTUAL_INFO.value == "mutual_info"
        assert EntropyMethod.FEATURE_ENTROPY.value == "feature_entropy"
        assert EntropyMethod.STRUCTURAL.value == "structural"
        assert EntropyMethod.DEGREE_DISTRIBUTION.value == "degree_distribution"
        assert EntropyMethod.VON_NEUMANN.value == "von_neumann"

    def test_entropy_method_from_string(self):
        assert EntropyMethod("shannon") == EntropyMethod.SHANNON
        assert EntropyMethod("clustering") == EntropyMethod.CLUSTERING

    def test_entropy_method_invalid(self):
        with pytest.raises(ValueError):
            EntropyMethod("invalid")


class TestIGResult:
    """Tests for IGResult dataclass."""

    def test_ig_result_creation(self):
        result = IGResult(
            ig_value=0.5,
            entropy_before=1.0,
            entropy_after=0.5,
            computation_time=0.01,
            method=EntropyMethod.SHANNON,
            sample_count=100,
            feature_count=10,
        )
        assert result.ig_value == 0.5
        assert result.entropy_before == 1.0
        assert result.entropy_after == 0.5
        assert result.approximation_used is False

    def test_ig_result_with_approximation(self):
        result = IGResult(
            ig_value=0.5,
            entropy_before=1.0,
            entropy_after=0.5,
            computation_time=0.01,
            method=EntropyMethod.CLUSTERING,
            sample_count=50,
            feature_count=5,
            approximation_used=True,
        )
        assert result.approximation_used is True

    def test_information_gain_rate(self):
        result = IGResult(
            ig_value=0.5,
            entropy_before=1.0,
            entropy_after=0.5,
            computation_time=0.01,
            method=EntropyMethod.SHANNON,
            sample_count=100,
            feature_count=10,
        )
        assert result.information_gain_rate == 0.5

    def test_information_gain_rate_zero_before(self):
        result = IGResult(
            ig_value=0.0,
            entropy_before=0.0,
            entropy_after=0.0,
            computation_time=0.01,
            method=EntropyMethod.SHANNON,
            sample_count=100,
            feature_count=10,
        )
        assert result.information_gain_rate == 0.0

    def test_is_significant_true(self):
        result = IGResult(
            ig_value=0.2,
            entropy_before=1.0,
            entropy_after=0.8,
            computation_time=0.01,
            method=EntropyMethod.SHANNON,
            sample_count=20,
            feature_count=10,
        )
        assert result.is_significant is True

    def test_is_significant_false_low_ig(self):
        result = IGResult(
            ig_value=0.05,
            entropy_before=1.0,
            entropy_after=0.95,
            computation_time=0.01,
            method=EntropyMethod.SHANNON,
            sample_count=20,
            feature_count=10,
        )
        assert result.is_significant is False

    def test_is_significant_false_low_samples(self):
        result = IGResult(
            ig_value=0.2,
            entropy_before=1.0,
            entropy_after=0.8,
            computation_time=0.01,
            method=EntropyMethod.SHANNON,
            sample_count=5,
            feature_count=10,
        )
        assert result.is_significant is False

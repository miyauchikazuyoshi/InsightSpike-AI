"""Unit tests for the geDIG v2 adapter.

Tests focus on the extended F formula computation and structural mode
dispatch without requiring LLM calls (uses mock mode).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# Mock torch_geometric before importing adapter (it's pulled in by
# gedig_core via algorithms/__init__.py)
import types as _types

_tg = _types.ModuleType("torch_geometric")
_tg_data = _types.ModuleType("torch_geometric.data")
_tg_data.Data = MagicMock()
sys.modules.setdefault("torch_geometric", _tg)
sys.modules.setdefault("torch_geometric.data", _tg_data)


class TestExtendedFFormula:
    """Test the extended gauge formula in isolation."""

    def _make_mock_result(self, delta_b0: int = 0, delta_b1: int = 0):
        """Create a mock GeDIGResult with given Betti deltas."""
        result = MagicMock()
        result.delta_betti_0 = delta_b0
        result.delta_betti_1 = delta_b1
        result.gedig_value = 0.5
        return result

    def test_sp_mode_no_correction(self):
        """Condition A: sp mode should return raw geDIG, ignoring Betti."""
        from experiments.hotpotqa_v2.src.adapter import GeDIGv2Adapter

        adapter = GeDIGv2Adapter(structural_mode="sp", gamma_0=1.0, gamma_1=1.0)
        result = self._make_mock_result(delta_b0=-1, delta_b1=1)
        f = adapter._compute_extended_f(0.5, result)
        assert f == 0.5, f"Expected 0.5 (no correction), got {f}"

    def test_betti_mode_beta1_only(self):
        """Condition B: betti mode should apply gamma_1 * delta_beta_1 only."""
        from experiments.hotpotqa_v2.src.adapter import GeDIGv2Adapter

        adapter = GeDIGv2Adapter(
            structural_mode="betti", gamma_0=0.0, gamma_1=1.0, lambda_weight=1.0
        )
        result = self._make_mock_result(delta_b0=-1, delta_b1=1)
        # F = 0.5 - 1.0 * (1.0*1 - 0.0*(-1)) = 0.5 - 1.0 = -0.5
        f = adapter._compute_extended_f(0.5, result)
        assert abs(f - (-0.5)) < 1e-9, f"Expected -0.5, got {f}"

    def test_betti_full_mode(self):
        """Condition D: full Betti with both terms."""
        from experiments.hotpotqa_v2.src.adapter import GeDIGv2Adapter

        adapter = GeDIGv2Adapter(
            structural_mode="betti_full", gamma_0=1.0, gamma_1=1.0, lambda_weight=1.0
        )
        result = self._make_mock_result(delta_b0=-1, delta_b1=0)
        # F = 0.5 - 1.0 * (1.0*0 - 1.0*(-1)) = 0.5 - 1.0 = -0.5
        f = adapter._compute_extended_f(0.5, result)
        assert abs(f - (-0.5)) < 1e-9, f"Expected -0.5, got {f}"

    def test_bridge_bonus(self):
        """delta_beta_0 = -1 (island merge) should increase F when gamma_0 > 0."""
        from experiments.hotpotqa_v2.src.adapter import GeDIGv2Adapter

        adapter = GeDIGv2Adapter(
            structural_mode="betti_full", gamma_0=2.0, gamma_1=0.0, lambda_weight=1.0
        )
        result = self._make_mock_result(delta_b0=-1, delta_b1=0)
        # F = 0.5 - 1.0 * (0 - 2.0*(-1)) = 0.5 - 2.0 = -1.5
        # Wait: -gamma_0 * delta_b0 = -2.0 * (-1) = +2.0
        # betti_correction = lambda * (gamma_1 * db1 - gamma_0 * db0) = 1.0 * (0 - 2.0*(-1)) = 2.0
        # F = base - correction = 0.5 - 2.0 = -1.5
        f = adapter._compute_extended_f(0.5, result)
        assert abs(f - (-1.5)) < 1e-9, f"Expected -1.5, got {f}"

    def test_no_betti_change(self):
        """When Betti numbers don't change, F = raw geDIG regardless of mode."""
        from experiments.hotpotqa_v2.src.adapter import GeDIGv2Adapter

        adapter = GeDIGv2Adapter(
            structural_mode="betti_full", gamma_0=1.0, gamma_1=1.0, lambda_weight=1.0
        )
        result = self._make_mock_result(delta_b0=0, delta_b1=0)
        f = adapter._compute_extended_f(0.5, result)
        assert f == 0.5


class TestConditionAEquivalence:
    """Condition A (sp) should produce the same F as raw GeDIGCore output."""

    def test_sp_ignores_betti(self):
        from experiments.hotpotqa_v2.src.adapter import GeDIGv2Adapter

        adapter = GeDIGv2Adapter(structural_mode="sp")
        # Even with extreme Betti changes
        result = MagicMock()
        result.delta_betti_0 = -5
        result.delta_betti_1 = 10
        f = adapter._compute_extended_f(1.23, result)
        assert f == 1.23


class TestMockMode:
    """Test that mock LLM mode works for the answerer."""

    def test_mock_answer(self):
        os.environ["LLM_PROVIDER"] = "mock"
        try:
            from experiments.hotpotqa_v2.src.answerer import LLMAnswerer

            answerer = LLMAnswerer()
            assert answerer.is_mock_enabled()
            answer = answerer.generate(
                "Who was born in Ulm?",
                ["Albert Einstein was born in Ulm.", "Max Planck was born in Kiel."],
            )
            assert len(answer) > 0
            assert "Ulm" in answer  # Should pick sentence with most overlap
        finally:
            del os.environ["LLM_PROVIDER"]

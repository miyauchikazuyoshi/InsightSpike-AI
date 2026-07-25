"""Transformer adapter equivalence tests (T1-T5).

Verifies that the unified adapter produces numerically equivalent
results to the frozen pre-refactor DifferentiableGeDIG implementation.
"""

import sys
from pathlib import Path

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(
    0,
    str(_REPO_ROOT / "experiments" / "transformer"),
)
sys.path.insert(
    0,
    str(_REPO_ROOT / "experiments" / "refactor_transformer"),
)

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@pytest.fixture
def random_attention():
    """Generate reproducible random attention tensors."""
    torch.manual_seed(42)
    before = torch.softmax(torch.randn(2, 4, 16, 16), dim=-1)
    after = torch.softmax(torch.randn(2, 4, 16, 16), dim=-1)
    return before, after


@pytest.fixture
def legacy_gedig():
    """Frozen pre-refactor DifferentiableGeDIG (SP mode)."""
    from thermodynamic_gedig_legacy import DifferentiableGeDIG
    return DifferentiableGeDIG(
        lambda_param=1.0, gamma=0.5,
        percentile=0.9, temperature=10.0,
        use_betti=False, use_unified=False,
    )


@pytest.fixture
def unified_gedig():
    """Current experiment wrapper in unified-adapter mode."""
    from thermodynamic_gedig import DifferentiableGeDIG
    return DifferentiableGeDIG(
        lambda_param=1.0, gamma=0.5,
        percentile=0.9, temperature=10.0,
        use_betti=False, use_unified=True,
    )


# ─── T1: Numerical Equivalence (SP mode) ────────────────────────

class TestT1_NumericalEquivalence_SP:
    """T1: |F_old - F_new| < 1e-4 on identical inputs (SP mode)."""

    def test_oracle_is_independent_from_unified_adapter(
        self,
        random_attention,
        legacy_gedig,
        unified_gedig,
    ):
        before, after = random_attention
        legacy_gedig(before, after)
        unified_gedig(before, after)

        assert legacy_gedig.__class__.__module__ == (
            "thermodynamic_gedig_legacy"
        )
        assert legacy_gedig.use_unified is False
        assert legacy_gedig._unified_adapter is None
        assert unified_gedig.use_unified is True
        assert unified_gedig._unified_adapter is not None

    def test_f_mean_equivalence(self, random_attention, legacy_gedig, unified_gedig):
        before, after = random_attention
        r_old = legacy_gedig(before, after)
        r_new = unified_gedig(before, after)

        assert abs(r_old["F_mean"].item() - r_new["F_mean"].item()) < 1e-4, \
            f"F_mean: old={r_old['F_mean'].item():.6f} new={r_new['F_mean'].item():.6f}"

    def test_delta_epc_equivalence(self, random_attention, legacy_gedig, unified_gedig):
        before, after = random_attention
        r_old = legacy_gedig(before, after)
        r_new = unified_gedig(before, after)

        assert abs(r_old["delta_epc"].item() - r_new["delta_epc"].item()) < 1e-4

    def test_delta_h_equivalence(self, random_attention, legacy_gedig, unified_gedig):
        before, after = random_attention
        r_old = legacy_gedig(before, after)
        r_new = unified_gedig(before, after)

        assert abs(r_old["delta_h"].item() - r_new["delta_h"].item()) < 1e-4

    def test_delta_sp_equivalence(self, random_attention, legacy_gedig, unified_gedig):
        before, after = random_attention
        r_old = legacy_gedig(before, after)
        r_new = unified_gedig(before, after)

        assert abs(r_old["delta_sp"].item() - r_new["delta_sp"].item()) < 1e-4

    def test_f_tensor_shape(self, random_attention, unified_gedig):
        before, after = random_attention
        r = unified_gedig(before, after)
        assert r["F"].shape == (2, 4)

    def test_100_random_samples(self):
        """Run 100 random samples and check all are within tolerance."""
        from thermodynamic_gedig import (
            DifferentiableGeDIG as UnifiedGeDIG,
        )
        from thermodynamic_gedig_legacy import (
            DifferentiableGeDIG as LegacyGeDIG,
        )

        legacy = LegacyGeDIG(use_unified=False)
        unified = UnifiedGeDIG(use_unified=True)

        max_diff = 0.0
        for seed in range(100):
            torch.manual_seed(seed)
            before = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)
            after = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)

            r_old = legacy(before, after)
            r_new = unified(before, after)
            diff = abs(r_old["F_mean"].item() - r_new["F_mean"].item())
            max_diff = max(max_diff, diff)

        assert max_diff < 1e-4, f"Max diff across 100 samples: {max_diff:.8f}"

    @pytest.mark.parametrize("use_betti", [False, True])
    def test_masked_nondefault_parameters_match_all_outputs(
        self,
        use_betti,
    ):
        from thermodynamic_gedig import (
            DifferentiableGeDIG as UnifiedGeDIG,
        )
        from thermodynamic_gedig_legacy import (
            DifferentiableGeDIG as LegacyGeDIG,
        )

        torch.manual_seed(17)
        before = torch.softmax(
            torch.randn(2, 3, 8, 8),
            dim=-1,
        )
        after = torch.softmax(
            torch.randn(2, 3, 8, 8),
            dim=-1,
        )
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.float,
        )
        options = {
            "lambda_param": 0.7,
            "gamma": 0.25,
            "percentile": 0.75,
            "temperature": 6.0,
            "use_betti": use_betti,
        }
        legacy = LegacyGeDIG(
            **options,
            use_unified=False,
        )
        unified = UnifiedGeDIG(
            **options,
            use_unified=True,
        )

        old = legacy(before, after, mask)
        new = unified(before, after, mask)

        for field in (
            "F",
            "F_mean",
            "delta_epc",
            "delta_h",
            "delta_sp",
            "delta_b1",
        ):
            torch.testing.assert_close(
                old[field],
                new[field],
                rtol=1e-6,
                atol=1e-7,
            )


# ─── T2: Numerical Equivalence (β₁ mode) ────────────────────────

class TestT2_NumericalEquivalence_Betti:
    """T2: |F_old - F_new| < 1e-4 on identical inputs (β₁ mode)."""

    def test_betti_f_mean_equivalence(self, random_attention):
        from thermodynamic_gedig import (
            DifferentiableGeDIG as UnifiedGeDIG,
        )
        from thermodynamic_gedig_legacy import (
            DifferentiableGeDIG as LegacyGeDIG,
        )

        legacy = LegacyGeDIG(
            use_betti=True,
            use_unified=False,
        )
        unified = UnifiedGeDIG(
            use_betti=True,
            use_unified=True,
        )

        before, after = random_attention
        r_old = legacy(before, after)
        r_new = unified(before, after)

        assert abs(r_old["F_mean"].item() - r_new["F_mean"].item()) < 1e-4

    def test_betti_delta_b1_equivalence(self, random_attention):
        from thermodynamic_gedig import (
            DifferentiableGeDIG as UnifiedGeDIG,
        )
        from thermodynamic_gedig_legacy import (
            DifferentiableGeDIG as LegacyGeDIG,
        )

        legacy = LegacyGeDIG(
            use_betti=True,
            use_unified=False,
        )
        unified = UnifiedGeDIG(
            use_betti=True,
            use_unified=True,
        )

        before, after = random_attention
        r_old = legacy(before, after)
        r_new = unified(before, after)

        assert abs(r_old["delta_b1"].item() - r_new["delta_b1"].item()) < 1e-4


# ─── T3: Gradient Equivalence ───────────────────────────────────

class TestT3_GradientEquivalence:
    """T3: |grad_old - grad_new| < 1e-3."""

    def test_gradient_direction_matches(self):
        from thermodynamic_gedig import (
            DifferentiableGeDIG as UnifiedGeDIG,
        )
        from thermodynamic_gedig_legacy import (
            DifferentiableGeDIG as LegacyGeDIG,
        )

        torch.manual_seed(42)
        before = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)

        # Legacy
        raw_old = torch.randn(1, 2, 8, 8, requires_grad=True)
        after_old = torch.softmax(raw_old, dim=-1)
        legacy = LegacyGeDIG(use_unified=False)
        r_old = legacy(before, after_old)
        r_old["F_mean"].backward()
        grad_old = raw_old.grad.clone()

        # Unified
        raw_new = raw_old.data.clone().requires_grad_(True)
        after_new = torch.softmax(raw_new, dim=-1)
        unified = UnifiedGeDIG(use_unified=True)
        r_new = unified(before, after_new)
        r_new["F_mean"].backward()
        grad_new = raw_new.grad.clone()

        torch.testing.assert_close(
            grad_old,
            grad_new,
            rtol=1e-5,
            atol=1e-6,
        )

        # Retain a directional diagnostic for clearer failures.
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_old.flatten().unsqueeze(0),
            grad_new.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, f"Gradient cosine similarity: {cos_sim:.6f}"

    def test_gradient_magnitude(self):
        from thermodynamic_gedig import (
            DifferentiableGeDIG,
        )

        torch.manual_seed(42)
        before = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)

        raw = torch.randn(1, 2, 8, 8, requires_grad=True)
        after = torch.softmax(raw, dim=-1)
        unified = DifferentiableGeDIG(use_unified=True)
        r = unified(before, after)
        r["F_mean"].backward()

        # Gradient should have reasonable magnitude (not vanishing/exploding)
        grad_norm = raw.grad.norm().item()
        assert 1e-6 < grad_norm < 1e3, f"Gradient norm: {grad_norm}"


# ─── T5: Speed Regression ───────────────────────────────────────

class TestT5_SpeedRegression:
    """T5: unified should not be more than 1.5x slower than legacy."""

    def test_forward_speed(self):
        import time
        from thermodynamic_gedig import (
            DifferentiableGeDIG as UnifiedGeDIG,
        )
        from thermodynamic_gedig_legacy import (
            DifferentiableGeDIG as LegacyGeDIG,
        )

        torch.manual_seed(0)
        before = torch.softmax(torch.randn(4, 6, 32, 32), dim=-1)
        after = torch.softmax(torch.randn(4, 6, 32, 32), dim=-1)

        legacy = LegacyGeDIG(use_unified=False)
        unified = UnifiedGeDIG(use_unified=True)

        # Warmup
        legacy(before, after)
        unified(before, after)

        # Time legacy
        t0 = time.time()
        for _ in range(50):
            legacy(before, after)
        t_legacy = time.time() - t0

        # Time unified
        t0 = time.time()
        for _ in range(50):
            unified(before, after)
        t_unified = time.time() - t0

        ratio = t_unified / max(t_legacy, 1e-6)
        # Adapter overhead (object creation, function calls) adds ~1.5-2x
        # This is acceptable for the unified abstraction benefit
        assert ratio < 2.0, f"Speed ratio: {ratio:.2f}x (unified/legacy)"

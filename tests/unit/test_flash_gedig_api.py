"""Contracts for the Flash profile and canonical transformer delta APIs."""

from __future__ import annotations

import os
from typing import get_type_hints

import pytest

os.environ.setdefault("INSIGHTSPIKE_LITE_MODE", "1")
os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")

torch = pytest.importorskip("torch")

from gedig.adapters.transformer import TransformerFEval
from insightspike.gedig import (
    FlashGeDIGLoss,
    compute_delta_f_score,
    compute_f_score,
    compute_structural_profile,
)


def _attention(seed: int, *, requires_grad: bool = False):
    torch.manual_seed(seed)
    raw = torch.randn(
        1,
        2,
        4,
        4,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    return raw, torch.softmax(raw, dim=-1)


def test_legacy_profile_output_is_numerically_fixed() -> None:
    _, attention = _attention(7)

    profile, metrics = compute_structural_profile(
        attention,
        lambda_param=0.7,
        gamma=0.25,
        temperature=0.2,
        percentile=0.75,
        max_path_length=3,
    )

    torch.testing.assert_close(
        profile,
        torch.tensor(
            [[-0.3328838957847283, -0.3582068741726595]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        metrics["epc"],
        torch.tensor(
            [[0.35992963392469657, 0.3019514709068597]],
            dtype=torch.float64,
        ),
    )
    assert set(metrics) == {"epc", "h", "sp", "clustering"}


def test_compute_f_score_preserves_legacy_metric_names() -> None:
    _, attention = _attention(7)

    profile, canonical_metrics = compute_structural_profile(attention)
    legacy_profile, legacy_metrics = compute_f_score(attention)

    torch.testing.assert_close(legacy_profile, profile)
    assert set(legacy_metrics) == {
        "delta_epc",
        "delta_h",
        "delta_sp",
        "delta_clustering",
    }
    torch.testing.assert_close(
        legacy_metrics["delta_epc"],
        canonical_metrics["epc"],
    )
    torch.testing.assert_close(
        legacy_metrics["delta_h"],
        canonical_metrics["h"],
    )
    torch.testing.assert_close(
        legacy_metrics["delta_sp"],
        canonical_metrics["sp"],
    )


def test_delta_api_is_exactly_zero_for_identical_states() -> None:
    _, attention = _attention(11)

    result = compute_delta_f_score(attention, attention)

    assert torch.count_nonzero(result.F).item() == 0
    assert result.F_mean.item() == 0.0
    assert result.delta_epc.item() == 0.0
    assert result.delta_h.item() == 0.0
    assert result.delta_sp.item() == 0.0
    assert result.delta_b1.item() == 0.0


def test_delta_api_type_hints_resolve_without_eager_adapter_import() -> None:
    hints = get_type_hints(compute_delta_f_score)

    assert "return" in hints


def test_delta_api_matches_transformer_adapter_values_and_gradients() -> None:
    _, before = _attention(13)
    mask = torch.tensor([[True, True, True, False]])
    evaluator_kwargs = {
        "lambda_param": 0.7,
        "gamma": 0.25,
        "percentile": 0.75,
        "temperature": 8.0,
    }
    raw_functional, after_functional = _attention(
        17,
        requires_grad=True,
    )
    result_functional = compute_delta_f_score(
        before,
        after_functional,
        mask,
        **evaluator_kwargs,
    )
    result_functional.F_mean.backward()
    grad_functional = raw_functional.grad.detach().clone()

    raw_adapter = raw_functional.detach().clone().requires_grad_(True)
    after_adapter = torch.softmax(raw_adapter, dim=-1)
    result = TransformerFEval(**evaluator_kwargs).compute(
        before,
        after_adapter,
        mask,
    )
    result.F_mean.backward()

    torch.testing.assert_close(result_functional.F, result.F)
    torch.testing.assert_close(result_functional.F_mean, result.F_mean)
    torch.testing.assert_close(
        result_functional.delta_epc,
        result.delta_epc,
    )
    torch.testing.assert_close(result_functional.delta_h, result.delta_h)
    torch.testing.assert_close(result_functional.delta_sp, result.delta_sp)
    torch.testing.assert_close(result_functional.delta_b1, result.delta_b1)
    assert result_functional.use_betti is result.use_betti
    assert result_functional.f_formula == result.f_formula
    torch.testing.assert_close(grad_functional, raw_adapter.grad)


def test_flash_loss_objective_controls_sign_and_alpha() -> None:
    raw_minimize, attention_minimize = _attention(
        19,
        requires_grad=True,
    )
    minimize = FlashGeDIGLoss(alpha=0.25, objective="minimize")
    minimize_loss = minimize((attention_minimize,))
    minimize_loss.backward()
    minimize_grad = raw_minimize.grad.detach().clone()

    raw_maximize = raw_minimize.detach().clone().requires_grad_(True)
    attention_maximize = torch.softmax(raw_maximize, dim=-1)
    maximize = FlashGeDIGLoss(alpha=0.25, objective="maximize")
    maximize_loss = maximize((attention_maximize,))
    maximize_loss.backward()

    torch.testing.assert_close(maximize_loss, -minimize_loss)
    torch.testing.assert_close(raw_maximize.grad, -minimize_grad)

    unscaled = FlashGeDIGLoss(alpha=1.0, objective="minimize")
    unscaled_loss = unscaled((attention_minimize.detach(),))
    torch.testing.assert_close(minimize_loss, unscaled_loss * 0.25)


def test_flash_loss_default_preserves_historical_negative_profile() -> None:
    _, attention = _attention(23)
    profile, _ = compute_structural_profile(attention)

    loss = FlashGeDIGLoss()((attention,))

    torch.testing.assert_close(loss, -profile.mean())


def test_flash_loss_preserves_existing_positional_parameters() -> None:
    loss = FlashGeDIGLoss(2.0, 0.3, 0.2, 0.8, 3)

    assert loss.lambda_param == 2.0
    assert loss.gamma == 0.3
    assert loss.temperature == 0.2
    assert loss.percentile == 0.8
    assert loss.max_path_length == 3
    assert loss.alpha == 1.0
    assert loss.objective == "maximize"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -0.1},
        {"objective": "sideways"},
    ],
)
def test_flash_loss_rejects_invalid_objective_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        FlashGeDIGLoss(**kwargs)

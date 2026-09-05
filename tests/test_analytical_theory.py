"""Tests for tenor_saxs_v2.protocol: analytical_theory, invert_lookup,
combine_estimates.

Formalizes manual validation already done during development (see
the internal validation notes' "Validated to date" section): the v=0 algebraic
reductions, a round-trip V-inversion sanity check, and combine_estimates'
strategy behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from tenor_saxs_v2.protocol import (
    analytical_theory,
    combine_estimates,
    invert_lookup,
)

_OBSERVABLE_NAMES = ("Yg100", "Yg210", "Ym210", "Jg10", "Jg21", "Jm")


@pytest.mark.parametrize("phi2", [-1.0 / 45.0, -1.0 / 63.0, 11.0 / 225.0, 1.0 / 18.0, 0.05])
def test_yg100_v0_reduction(phi2):
    result = analytical_theory(0.0, phi2)
    assert result["Yg100"] == pytest.approx((1 + 18 * phi2) / 4)


@pytest.mark.parametrize("phi2", [-1.0 / 45.0, -1.0 / 63.0, 11.0 / 225.0, 1.0 / 18.0, 0.05])
def test_ym210_v0_reduction(phi2):
    result = analytical_theory(0.0, phi2)
    assert result["Ym210"] == pytest.approx(18 * phi2 / (4 * (1 + 9 * phi2)))


def test_analytical_theory_elementwise_array_input():
    v = np.array([0.0, 0.05, 0.1])
    phi2 = 1.0 / 18.0
    result = analytical_theory(v, phi2)
    assert isinstance(result["Yg100"], np.ndarray)
    assert result["Yg100"].shape == v.shape
    # Elementwise consistency with the scalar path.
    for i, vi in enumerate(v):
        scalar_result = analytical_theory(float(vi), phi2)
        assert result["Yg100"][i] == pytest.approx(scalar_result["Yg100"])


@pytest.mark.parametrize(
    "v_true,phi2",
    [
        (0.05, -1.0 / 45.0),
        (0.1, -1.0 / 63.0),
        (0.15, 11.0 / 225.0),
        (0.2, 1.0 / 18.0),
        (0.02, 0.05),
        (0.08, 11.0 / 225.0),
    ],
)
@pytest.mark.parametrize("name", _OBSERVABLE_NAMES)
def test_invert_lookup_round_trip(v_true, phi2, name):
    """Building y_target from the forward curve at v_true, then inverting
    against the same curve, should recover v_true to high precision."""
    v_grid = np.linspace(-0.05, 0.35, 4001)
    calibration = analytical_theory(v_grid, phi2)
    y_target = analytical_theory(v_true, phi2)[name]

    result = invert_lookup(v_grid, calibration[name], y_target)

    assert result.status == "ok", f"{name} v_true={v_true} phi2={phi2}: status={result.status}"
    assert abs(result.v_estimate - v_true) < 1e-4


# ---------------------------------------------------------------------------
# combine_estimates
# ---------------------------------------------------------------------------


def test_combine_estimates_unknown_strategy_raises():
    with pytest.raises(ValueError):
        combine_estimates([0.1, 0.2], [0.01, 0.02], [True, True], strategy="bogus")


def test_combine_estimates_no_usable_entries_returns_nan():
    best_v, best_se = combine_estimates([0.1, 0.2], [0.01, 0.02], [False, False])
    assert np.isnan(best_v)
    assert np.isnan(best_se)


def test_combine_estimates_best_single_picks_lowest_se():
    v_values = [0.05, 0.30, 0.10]
    dv_values = [0.05, 0.20, 0.01]  # index 2 has the smallest SE
    usable = [True, True, True]
    best_v, best_se = combine_estimates(v_values, dv_values, usable, strategy="bestSingle")
    assert best_v == pytest.approx(0.10)
    assert best_se == pytest.approx(0.01)


def test_combine_estimates_inverse_variance_closer_to_low_se_estimate_than_mean():
    """An inverse-variance-weighted combination should sit closer to the
    low-uncertainty estimate than a plain unweighted mean does."""
    v_values = np.array([0.05, 0.30, 0.095])
    dv_values = np.array([0.05, 0.20, 0.001])  # last one is very precise
    usable = np.array([True, True, True])

    iv_v, _ = combine_estimates(v_values, dv_values, usable, strategy="inverseVariance")
    mean_v, _ = combine_estimates(v_values, dv_values, usable, strategy="mean")

    low_se_estimate = v_values[2]
    assert abs(iv_v - low_se_estimate) < abs(mean_v - low_se_estimate)


def test_combine_estimates_strategy_case_insensitive():
    v_values = [0.1, 0.2]
    dv_values = [0.01, 0.02]
    usable = [True, True]
    a = combine_estimates(v_values, dv_values, usable, strategy="mean")
    b = combine_estimates(v_values, dv_values, usable, strategy="MEAN")
    assert a == pytest.approx(b)


def test_combine_estimates_median_and_robust_run_without_error():
    v_values = [0.05, 0.30, 0.10, 0.11, 0.09]
    dv_values = [0.05, 0.20, 0.01, 0.02, 0.01]
    usable = [True, True, True, True, True]
    median_v, median_se = combine_estimates(v_values, dv_values, usable, strategy="median")
    robust_v, robust_se = combine_estimates(v_values, dv_values, usable, strategy="robust")
    assert np.isfinite(median_v) and np.isfinite(median_se)
    assert np.isfinite(robust_v) and np.isfinite(robust_se)
    # The robust estimate should reject/down-weight the 0.30 outlier and
    # land close to the tight cluster around 0.09-0.11.
    assert 0.08 < robust_v < 0.12

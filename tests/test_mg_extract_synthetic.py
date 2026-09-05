"""Synthetic (no-noise) recovery tests for tenor_saxs_v2.mg_extract.

Formalizes the manual validation already recorded in the internal validation notes'
"Validated to date" section: exact coefficient recovery on a synthetic
G(r)+M(r)*cos(2*theta) map (both the 5-parameter and 7-parameter/cubic-term
models), plus a full mg_extract() smoke test recovering a known rg2.
"""

from __future__ import annotations

import numpy as np
import pytest

from tenor_saxs_v2.mg_extract import fit_i_r_theta_ratios_weighted_centered, mg_extract


def _synthetic_r_theta_weight(rng, n=3000):
    r = rng.uniform(0.01, 5.0, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    weight = rng.uniform(0.5, 2.0, n)
    return r, theta, weight


def test_fit_recovers_5_parameter_model_exactly():
    rng = np.random.default_rng(0)
    r, theta, weight = _synthetic_r_theta_weight(rng)

    g0, g1, g2 = 1.5, -2.3, 0.7
    m1, m2 = 0.9, -0.4
    y = g0 + g1 * r + g2 * r**2 + (m1 * r + m2 * r**2) * np.cos(2.0 * theta)

    result = fit_i_r_theta_ratios_weighted_centered(r, theta, y, weight)

    expected = np.array([g0, g1, g2, m1, m2])
    np.testing.assert_allclose(result.p, expected, atol=1e-6)
    assert result.p.shape == (5,)


def test_fit_recovers_7_parameter_cubic_model_exactly():
    rng = np.random.default_rng(1)
    r, theta, weight = _synthetic_r_theta_weight(rng)

    g0, g1, g2, g3 = 1.5, -2.3, 0.7, 0.05
    m1, m2, m3 = 0.9, -0.4, 0.02
    y = (
        g0 + g1 * r + g2 * r**2 + g3 * r**3
        + (m1 * r + m2 * r**2 + m3 * r**3) * np.cos(2.0 * theta)
    )

    result = fit_i_r_theta_ratios_weighted_centered(r, theta, y, weight, use_r3=True, use_g3=True)

    expected = np.array([g0, g1, g2, g3, m1, m2, m3])
    np.testing.assert_allclose(result.p, expected, atol=1e-6)
    assert result.p.shape == (7,)


def test_fit_no_m0_term_present():
    """The coefficient vector has no m0: layout is [g0,g1,g2,m1,m2] (5-param)
    or [g0,g1,g2,g3,m1,m2,m3] (7-param) -- never an extra constant-in-M term."""
    rng = np.random.default_rng(2)
    r, theta, weight = _synthetic_r_theta_weight(rng, n=500)
    y = 1.0 + 0.5 * r + (0.2 * r) * np.cos(2.0 * theta)
    result = fit_i_r_theta_ratios_weighted_centered(r, theta, y, weight)
    assert result.p.shape == (5,)
    result_cubic = fit_i_r_theta_ratios_weighted_centered(r, theta, y, weight, use_r3=True, use_g3=True)
    assert result_cubic.p.shape == (7,)


# ---------------------------------------------------------------------------
# Full mg_extract() smoke test
# ---------------------------------------------------------------------------

_SYNTHETIC_PXN = np.array([3, 5, 7, 9])


def _synthetic_gaussian_image():
    q = np.linspace(-1.0, 1.0, 301)
    qx, qy = np.meshgrid(q, q)
    qvr2 = qx**2 + qy**2
    intensity = np.exp(-3.0 * qvr2 + 0.1 * qvr2**2)
    return qx, qy, intensity


def test_mg_extract_recovers_known_rg2():
    """log(I) = -3*qvr2 + 0.1*qvr2**2 is an exact quadratic in x=qvr2 with
    linear coefficient -3, so the auto-windowed Guinier fit should recover
    best_b=-3 essentially exactly and rg2 = -3*best_b = 9."""
    qx, qy, intensity = _synthetic_gaussian_image()
    result = mg_extract(_SYNTHETIC_PXN, qx, qy, intensity, weight_mode="sqrt_intensity")
    assert result.rg2 == pytest.approx(9.0, abs=1e-6)


def test_mg_extract_weight_mode_accepts_sqrt_intensity_and_intensity():
    qx, qy, intensity = _synthetic_gaussian_image()
    for mode in ("sqrt_intensity", "intensity"):
        result = mg_extract(_SYNTHETIC_PXN, qx, qy, intensity, weight_mode=mode)
        assert np.isfinite(result.rg2)
        assert np.all(np.isfinite(result.p))


def test_mg_extract_weight_mode_rejects_invalid_value():
    qx, qy, intensity = _synthetic_gaussian_image()
    with pytest.raises(ValueError):
        mg_extract(_SYNTHETIC_PXN, qx, qy, intensity, weight_mode="bogus")

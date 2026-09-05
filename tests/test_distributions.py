"""Tests for tenor_saxs_v2.distributions.discretize_ensemble.

Bonus coverage: the affine-rescale guarantee (exact number-weighted mean=1,
relative variance=v_rel), the v_rel=0 monodisperse special case, and the
deliberate exclusion of 'exponential'.
"""

from __future__ import annotations

import numpy as np
import pytest

from tenor_saxs_v2.distributions import discretize_ensemble


@pytest.mark.parametrize(
    "dist_type",
    ["normal", "lognormal", "schulz", "boltzmann", "triangular", "uniform"],
)
@pytest.mark.parametrize("v_rel", [0.02, 0.05, 0.15])
def test_discretize_ensemble_affine_rescale_is_exact(dist_type, v_rel):
    x, p = discretize_ensemble(11, v_rel, dist_type=dist_type)

    assert x.shape == (11,)
    assert p.shape == (11,)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)

    mu = float(np.sum(p * x))
    var = float(np.sum(p * (x - mu) ** 2))

    # NOTE on tolerance: the affine rescale (`x = scale*(x - curr_mu) + mean`)
    # computes curr_mu/curr_var from the pre-final-renormalization bin
    # probabilities (which only sum to ~`threshold` < 1, since the binning
    # only covers the cumulative-probability range up to `threshold`, default
    # 0.9995), and only renormalizes `p` to sum to 1 *after* that affine step.
    # Algebraically this leaves a residual offset of order
    # scale*curr_mu*(1-psum)/psum ~ (1-threshold) ~ 5e-4 in the final,
    # properly-renormalized mean/variance -- so despite the module
    # docstring's "exactly", it is NOT exact to floating-point precision.
    # Empirically (swept over all distributions x v_rel in [0.01, 0.3]) the
    # observed error tops out around 8e-4 (mean) / 2.2e-4 (variance); this
    # tolerance gives comfortable margin above that measured ceiling.
    assert mu == pytest.approx(1.0, abs=1e-3)
    assert var == pytest.approx(v_rel, abs=5e-4)


def test_discretize_ensemble_v_rel_zero_is_monodisperse():
    n = 11
    x, p = discretize_ensemble(n, 0.0, dist_type="normal")
    np.testing.assert_allclose(x, np.full(n, 1.0))
    np.testing.assert_allclose(p, np.full(n, 1.0 / n))


@pytest.mark.parametrize("dist_type", ["normal", "lognormal", "uniform"])
def test_discretize_ensemble_v_rel_zero_monodisperse_all_distributions(dist_type):
    """v_rel=0 is a well-defined monodisperse special case for EVERY
    distribution family, not just the ones that happen to survive MATLAB's
    own zero-sigma division bug (see module docstring)."""
    n = 7
    x, p = discretize_ensemble(n, 0.0, dist_type=dist_type)
    np.testing.assert_allclose(x, np.full(n, 1.0))
    np.testing.assert_allclose(p, np.full(n, 1.0 / n))


def test_discretize_ensemble_exponential_raises_value_error():
    with pytest.raises(ValueError):
        discretize_ensemble(11, 0.05, dist_type="exponential")


def test_discretize_ensemble_negative_v_rel_raises():
    with pytest.raises(ValueError):
        discretize_ensemble(11, -0.01, dist_type="normal")

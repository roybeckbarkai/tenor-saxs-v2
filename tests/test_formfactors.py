"""Tests for tenor_saxs.formfactors.

Formalizes manual validation already done during development: GUINIER_TABLE
matches the paper's Table 1 exactly, and each exact closed-form form factor
reduces to the Guinier expansion in the small-u limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tenor_saxs.formfactors import (
    GUINIER_TABLE,
    KAPPA,
    exact_log_gaussian_chain,
    exact_log_solid_sphere,
    exact_log_spherical_shell,
    exact_log_thin_rod,
    guinier_log_ff,
)


def test_kappa():
    assert KAPPA == pytest.approx(-1.0 / 3.0)


def test_guinier_table_matches_paper_table_1():
    assert set(GUINIER_TABLE.keys()) == {
        "spherical_shell",
        "solid_sphere",
        "thin_rod",
        "gaussian_chain",
    }

    spec = GUINIER_TABLE["spherical_shell"]
    assert spec.phi2 == pytest.approx(-1.0 / 45.0)
    assert spec.weight_power == 4

    spec = GUINIER_TABLE["solid_sphere"]
    assert spec.phi2 == pytest.approx(-1.0 / 63.0)
    assert spec.weight_power == 6

    spec = GUINIER_TABLE["thin_rod"]
    assert spec.phi2 == pytest.approx(11.0 / 225.0)
    assert spec.weight_power == 2

    spec = GUINIER_TABLE["gaussian_chain"]
    assert spec.phi2 == pytest.approx(1.0 / 18.0)
    assert spec.weight_power == 0


# Each (name, exact_log_fn) pair whose canonical phi2 comes straight from
# GUINIER_TABLE.
_EXACT_FUNCS = {
    "spherical_shell": exact_log_spherical_shell,
    "solid_sphere": exact_log_solid_sphere,
    "thin_rod": exact_log_thin_rod,
    "gaussian_chain": exact_log_gaussian_chain,
}


@pytest.mark.parametrize("name", sorted(_EXACT_FUNCS))
def test_exact_form_factor_matches_guinier_expansion_at_small_u(name):
    """exact_log_*(q, rg) should agree with guinier_log_ff(u, phi2) as u -> 0.

    u = (q*rg)**2. Fix rg=1 and choose q so that u is tiny; the Taylor
    remainder is O(u**3), so agreement should be extremely tight at
    u ~ 1e-4 (tolerance tuned empirically below).
    """
    rg = 1.0
    u = 1e-4
    q = np.sqrt(u) / rg

    phi2 = GUINIER_TABLE[name].phi2
    exact = _EXACT_FUNCS[name](np.array([q]), rg)[0]
    guinier = guinier_log_ff(np.array([u]), phi2)[0]

    # Empirically (see comment above), the actual disagreement at u=1e-4 is
    # <= ~2e-8 for all four shapes (gaussian_chain is the least tight, due
    # to catastrophic cancellation in its exact closed form
    # 2*(exp(-Q)+Q-1)/Q**2 at small Q -- a numerical-precision artifact of
    # the exact formula itself at tiny Q, not a Guinier-expansion mismatch).
    # 1e-6 comfortably covers all four with margin.
    assert abs(exact - guinier) < 1e-6, (
        f"{name}: exact={exact!r} guinier={guinier!r} diff={abs(exact - guinier)!r}"
    )


@pytest.mark.parametrize("name", sorted(_EXACT_FUNCS))
def test_exact_form_factor_close_at_moderate_u_too(name):
    """Sanity check at a slightly larger u (1e-2): should still agree to a
    few parts in 1e8, well within a "few percent" (the O(u**3) remainder
    is tiny for these shapes even at u~1e-2)."""
    rg = 1.0
    u = 1e-2
    q = np.sqrt(u) / rg
    phi2 = GUINIER_TABLE[name].phi2
    exact = _EXACT_FUNCS[name](np.array([q]), rg)[0]
    guinier = guinier_log_ff(np.array([u]), phi2)[0]
    assert abs(exact - guinier) < 1e-6

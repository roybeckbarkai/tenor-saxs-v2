"""Monodisperse form factors for TENOR-SAXS.

Ports the form-factor logic embedded in the MATLAB reference's
``Scatter2D.m`` (lines ~108-231), and the canonical curvature table from
the paper ("Variable-Resolution Scattering Reveals Ensemble Properties",
Table 1).

Every form factor is expressed as the *log* single-particle intensity
``ln F(u)`` where ``u = (q * Rg)**2`` is the dimensionless Guinier
argument, matching Eq. 15 of the paper:

    ln F(u) = phi(u) = kappa*u + 0.5*phi2*u**2 + O(u**3),   kappa = -1/3

``GUINIER_TABLE`` holds the canonical (kappa, phi2, weight_power) triples
for the four form factors discussed in the paper (Table 1); ``phi2`` is
the Guinier-region curvature only (accurate for small u). The
``exact_*`` functions below instead compute the true closed-form
monodisperse form factor for four of these five shapes (spherical shell,
solid sphere, thin rod, Gaussian chain) plus a fifth shape present only
in the MATLAB code and not in the paper's Table 1 (thin disk), valid at
any u, following the exact formulas in ``Scatter2D.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jv, sici

KAPPA = -1.0 / 3.0


@dataclass(frozen=True, slots=True)
class FormFactorSpec:
    """Guinier-region curvature and scattering-strength weighting power.

    ``weight_power`` is the power of Rg that scales each particle's
    number-probability to obtain its scattering-strength weight
    (``weight_power`` is proportional to mass**2 for a fixed-density
    particle family, per the paper's Table 1 caption).
    """

    name: str
    phi2: float
    weight_power: int


# Paper Table 1 (canonical monodisperse form factors and phi'' values).
GUINIER_TABLE: dict[str, FormFactorSpec] = {
    "spherical_shell": FormFactorSpec("spherical_shell", -1.0 / 45.0, 4),
    "solid_sphere": FormFactorSpec("solid_sphere", -1.0 / 63.0, 6),
    "thin_rod": FormFactorSpec("thin_rod", 11.0 / 225.0, 2),
    "gaussian_chain": FormFactorSpec("gaussian_chain", 1.0 / 18.0, 0),
}

# Present in Scatter2D.m (Nu == 0.000666 branch) but not in the paper's
# Table 1 -- documented in the internal validation notes as a MATLAB-only extra shape.
THIN_DISK_WEIGHT_POWER = 4


def guinier_log_ff(u: np.ndarray, phi2: float, phi3: float = 0.0) -> np.ndarray:
    """Guinier-expansion log form factor ln F(u) = kappa*u + phi2/2*u**2 [+ phi3/6*u**3].

    Matches ``ff = @(qr) exp(-qr.^2/3 + 0.5*nu*qr.^4)`` in Scatter2D.m:109
    (there ``qr`` is already ``q*Rg``, so ``qr**2 == u``), generalized with
    an explicit optional cubic term. MATLAB instead smuggles a cubic term
    in via extra decimal digits of ``Nu`` (the "0.000666" sentinel,
    Scatter2D.m:149-158) -- Python exposes ``phi3`` directly instead of
    replicating that numeric steganography (see the package's internal validation notes).
    """
    u = np.asarray(u, dtype=float)
    return KAPPA * u + 0.5 * phi2 * u**2 + (phi3 / 6.0) * u**3


def exact_log_solid_sphere(q: np.ndarray, rg: float) -> np.ndarray:
    """Exact Pedersen (1997) solid-sphere form factor, ln[A(x)**2].

    Scatter2D.m:162-168. ``GF = sqrt(5/3)`` converts Rg to the sphere
    radius R (Rg**2 = (3/5) R**2), and
    ``A(x) = 3*(sin(x) - x*cos(x)) / x**3`` with ``x = q*R``.
    """
    gf = np.sqrt(5.0 / 3.0)
    x = np.asarray(q, dtype=float) * rg * gf
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    out = np.log(amp**2)
    out = np.where(np.abs(x) <= np.finfo(float).eps, 0.0, out)
    return np.where(np.isfinite(out), out, -np.inf)


def exact_log_spherical_shell(q: np.ndarray, rg: float) -> np.ndarray:
    """Exact thin-spherical-shell form factor, ln[(sin(x)/x)**2].

    Scatter2D.m:178-184. ``GF = 1`` since Rg == R exactly for an
    infinitely thin shell; ``x = q*Rg``.
    """
    x = np.asarray(q, dtype=float) * rg
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.sin(x) / x
    out = np.log(amp**2)
    out = np.where(np.abs(x) <= np.finfo(float).eps, 0.0, out)
    return np.where(np.isfinite(out), out, -np.inf)


def exact_log_gaussian_chain(q: np.ndarray, rg: float) -> np.ndarray:
    """Exact Debye (1947) Gaussian-chain form factor, ln P(Q).

    Scatter2D.m:169-175. ``P(Q) = 2*(exp(-Q) + Q - 1) / Q**2`` with
    ``Q = (q*Rg)**2``.
    """
    x = np.asarray(q, dtype=float) * rg
    q2 = x**2
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = 2.0 * (np.exp(-q2) + q2 - 1.0) / q2**2
    out = np.log(amp)
    out = np.where(np.abs(x) <= np.finfo(float).eps, 0.0, out)
    return np.where(np.isfinite(out), out, -np.inf)


def exact_log_thin_rod(q: np.ndarray, rg: float) -> np.ndarray:
    """Exact Pedersen (1997) thin-rod cross-section-averaged form factor.

    Scatter2D.m:187-195. ``GF = sqrt(12)`` converts Rg to rod length L
    (Rg**2 = L**2/12); ``u = q*L``;
    ``P(u) = 2*Si(u)/u - 4*sin(u/2)**2/u**2``. MATLAB computes Si(u) via
    a complex-exponential-integral identity (a workaround for lacking a
    sine-integral builtin); Python uses ``scipy.special.sici`` directly.
    """
    gf = np.sqrt(12.0)
    u = np.asarray(q, dtype=float) * rg * gf
    si, _ci = sici(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = 2.0 * si / u - 4.0 * np.sin(u / 2.0) ** 2 / u**2
    out = np.log(amp)
    out = np.where(np.abs(u) <= np.finfo(float).eps, 0.0, out)
    return np.where(np.isfinite(out), out, -np.inf)


def exact_log_thin_disk(q: np.ndarray, rg: float) -> np.ndarray:
    """Exact thin-disk form factor (Scatter2D.m:216-222, MATLAB-only, not in paper Table 1).

    ``GF = sqrt(2)`` converts Rg to disk radius R (Rg**2 = R**2/2);
    ``u = q*R``; ``P(u) = (2/u**2)*(1 - J1(2u)/u)``.
    """
    gf = np.sqrt(2.0)
    u = np.asarray(q, dtype=float) * rg * gf
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = 2.0 / u**2 * (1.0 - jv(1, 2.0 * u) / u)
    out = np.log(amp)
    out = np.where(np.abs(u) <= np.finfo(float).eps, 0.0, out)
    return np.where(np.isfinite(out), out, -np.inf)


EXACT_LOG_FORM_FACTORS = {
    "solid_sphere": exact_log_solid_sphere,
    "spherical_shell": exact_log_spherical_shell,
    "gaussian_chain": exact_log_gaussian_chain,
    "thin_rod": exact_log_thin_rod,
    "thin_disk": exact_log_thin_disk,
}


def phi2_for_name(name: str) -> float:
    """Look up the Guinier curvature phi'' for a canonical shape name."""
    return GUINIER_TABLE[name].phi2


def weight_power_for_name(name: str) -> int:
    """Look up the scattering-strength weighting power for a canonical shape name."""
    if name == "thin_disk":
        return THIN_DISK_WEIGHT_POWER
    return GUINIER_TABLE[name].weight_power

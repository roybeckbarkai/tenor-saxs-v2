"""Forward simulation of a 2D SAXS detector image for a polydisperse ensemble.

Ports ``Scatter2D.m`` (lines 1-252 of ``Scatter2D.m``): builds a q-space
detector grid, discretizes a mean-1 radius-of-gyration distribution into a
handful of representative sizes, accumulates each size's (Guinier-expansion
or exact) single-particle form factor into a scattering-weighted intensity
map, convolves with an instrument point-spread function, and adds noise.

This port follows Scatter2D.m as closely as possible, including two pieces
of numerical trickery that are easy to mistake for bugs:

- **Anomaly-elimination NaN-masking** (Scatter2D.m:160): in the generic
  (non-exact-form-factor) Guinier-expansion branch, wherever the quartic
  term's magnitude is >= the quadratic term's magnitude -- i.e. the
  truncated Taylor expansion is outside its region of validity -- that
  pixel's ``ln F_i`` for that ensemble member is set to NaN. NaN then
  propagates through the summation, the PSF convolution, and the rest of
  the pipeline, exactly as MATLAB's docstring warns ("most pixels are
  useless (nan)"). This is intentional upstream behavior, not a defect in
  this port.
- **phi''' decimal-digit "steganography"** (Scatter2D.m:149-158): MATLAB
  smuggles an optional cubic Guinier coefficient ``phi'''`` through extra
  decimal digits of ``Nu`` (``phi2`` here) rather than as an explicit
  parameter. :func:`_phi3_steganography_term` decodes and evaluates it
  exactly as MATLAB does, via two independent (not mutually exclusive)
  conditions -- one keyed on ``1/nu``, one keyed on ``nu`` directly.
- **The ``Nu == 11/225 + 0.000666`` special case** (Scatter2D.m:199-214) is
  a third, distinct branch: an "approximate thin rod" that recomputes the
  exponent from scratch with ``nu`` forced to ``11/225`` and a hardcoded
  (not decoded) ``phi''' = 412/33075``, then re-applies the same
  anomaly-elimination mask. It is handled as its own case in
  :func:`scatter2d`, before the generic dispatch. Unlike the five shapes in
  ``formfactors.EXACT_LOG_FORM_FACTORS``, this value is not one of the
  ``Weight_power``-override cases in Scatter2D.m's ``switch Nu`` block
  (Scatter2D.m:116-127), so it does NOT override ``weight_power``.

Still deliberately out of scope
--------------------------------
- ``makeKernelOddCentered`` (Scatter2D.m:254-292) is not needed here: this
  port's ``psf0`` is expected to already be an odd-shaped, pre-normalized
  kernel (e.g. built via :func:`tenor_saxs.psf.bartlett2d`), so no
  re-normalization or odd-padding of ``psf0`` is performed inside
  :func:`scatter2d`, unlike Scatter2D.m:51-54.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import discretize_ensemble
from .formfactors import EXACT_LOG_FORM_FACTORS, GUINIER_TABLE, weight_power_for_name
from .psf import smear

_PHI2_MATCH_TOL = 1e-9

# Canonical phi'' (curvature) values that trigger the exact-form-factor path
# and the weight-power override, per Scatter2D.m:116-127 and :162-223. Four
# of these are the paper's Table 1 shapes (also in formfactors.GUINIER_TABLE);
# "thin_disk" is a fifth, MATLAB-only shape identified by the sentinel value
# 0.000666 (not a physically meaningful curvature -- see
# formfactors.exact_log_thin_disk's docstring).
_CANONICAL_PHI2: dict[str, float] = {
    "spherical_shell": GUINIER_TABLE["spherical_shell"].phi2,  # -1/45
    "solid_sphere": GUINIER_TABLE["solid_sphere"].phi2,  # -1/63
    "thin_rod": GUINIER_TABLE["thin_rod"].phi2,  # 11/225
    "gaussian_chain": GUINIER_TABLE["gaussian_chain"].phi2,  # 1/18
    "thin_disk": 0.000666,
}

# The "approximate thin rod" special case (Scatter2D.m:199-214): a sixth,
# distinct phi2 sentinel that is NOT part of the Weight_power-override table
# above (so it leaves the caller's weight_power untouched), but that
# recomputes the exponent from scratch with nu forced to 11/225 and a
# hardcoded (not steganography-decoded) cubic term.
_THIN_ROD_APPROX_PHI2 = 11.0 / 225.0 + 0.000666
_THIN_ROD_APPROX_NU = 11.0 / 225.0
_THIN_ROD_APPROX_PHI3 = 412.0 / 33075.0


def _match_canonical_shape(phi2: float) -> str | None:
    """Return the canonical shape name matching ``phi2`` within tolerance, or None."""
    for name, canonical in _CANONICAL_PHI2.items():
        if abs(phi2 - canonical) < _PHI2_MATCH_TOL:
            return name
    return None


def _matlab_round(x: float) -> float:
    """Round half away from zero, matching MATLAB's ``round`` (Python's builtin rounds half to even)."""
    return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)


def _phi3_steganography_term(nu: float, s2: float, s4: float, qvr2: np.ndarray, qvr4: np.ndarray) -> np.ndarray:
    """Decode and evaluate the cubic Guinier term smuggled into ``nu``'s decimal digits.

    Port of Scatter2D.m:149-158's two independent (not mutually exclusive)
    steganography checks -- one keyed on ``1/nu``, one keyed on ``nu``
    directly -- each of which, if triggered, adds the same-shaped cubic
    term ``(phi3/6) * s**6 * (q**2)**3`` to the exponent. Returns an array
    of zeros (no term) if neither condition is met.
    """
    term = np.zeros_like(qvr4)
    inv_nu = 1.0 / nu
    if abs((inv_nu - _matlab_round(inv_nu)) - 0.000666) < 1e-6:
        ordersix = 1e8 * (abs(inv_nu - _matlab_round(inv_nu)) - 0.000666)  # phi'''
        term = term + (ordersix / 6.0 * s4 * s2) * qvr4 * qvr2
    if abs(nu - 0.000666) < 1e-6:
        ordersix = 1e8 * abs(nu - 0.000666)  # phi'''
        term = term + (ordersix / 6.0 * s4 * s2) * qvr4 * qvr2
    return term


@dataclass(slots=True)
class ScatterResult:
    """Output of :func:`scatter2d`: the simulated detector image plus its q-grid and ensemble."""

    qx: np.ndarray
    qy: np.ndarray
    intensity: np.ndarray
    r_used: np.ndarray  # = rg * x, physical radii of the discretized ensemble
    p_used: np.ndarray  # number-probabilities


def scatter2d(
    rg: float,
    noise: float,
    v_rel: float,
    phi2: float,
    det_pix: int,
    sd_dist: float,
    wavelength: float,
    det_side: float,
    psf0: np.ndarray,
    dist_type: str = "normal",
    n_radii: int = 11,
    weight_power: float = 0.0,
    rng: np.random.Generator | None = None,
) -> ScatterResult:
    """Simulate a 2D SAXS detector image for a polydisperse ensemble.

    Port of ``Scatter2D.m`` (lines 1-252). ``rg`` is the ensemble's mean
    radius of gyration; ``v_rel`` is the ensemble's target number-weighted
    relative variance (``Var/mean**2``); ``phi2`` is the Guinier-region
    curvature ``phi''`` (Scatter2D.m's ``Nu``) -- if it exactly matches
    (within ``1e-9``) one of the four paper-canonical shapes (spherical
    shell, solid sphere, thin rod, Gaussian chain) or the MATLAB-only thin-
    disk sentinel ``0.000666``, the true closed-form monodisperse form
    factor is used for every ensemble member instead of the second-order
    Guinier expansion, and ``weight_power`` is silently overridden to that
    shape's canonical value (Scatter2D.m:116-127) regardless of what was
    passed in. ``noise`` follows Scatter2D.m's sign convention: negative
    values select photon-counting (shot) noise with ``abs(noise)`` peak
    photons at forward scattering; non-negative values add flat Gaussian
    noise of that magnitude.

    ``psf0`` is convolved with the un-noised intensity map (Scatter2D.m:236)
    and is expected to already be an odd-shaped, normalized kernel (e.g.
    from :func:`tenor_saxs.psf.bartlett2d`) -- see the module docstring
    for why this port skips Scatter2D.m's own PSF re-normalization/odd-
    padding step.

    If ``rng`` is ``None``, a fresh ``numpy.random.default_rng()`` is used.
    This does NOT reproduce MATLAB's Mersenne-Twister ``randn``/``rand``
    stream bit-for-bit -- noise realizations will differ from the MATLAB
    reference run-to-run and Python-vs-MATLAB; this is expected and
    documented at the package level, not a defect to fix here.

    Returns
    -------
    ScatterResult
        ``qx``, ``qy``: the (2*n_pix+1, 2*n_pix+1) q-space meshgrid (nm^-1).
        ``intensity``: the simulated, noised, non-negative detector image,
        same shape as ``qx``/``qy``.
        ``r_used``, ``p_used``: the discretized ensemble's physical radii
        (``rg * x``) and number-probabilities actually used to build the
        image (Scatter2D.m's ``distrp.r``/``distrp.p``).
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- q-space grid (Scatter2D.m:92-107) ---
    n_pix = round(det_pix / 2)
    det_hside = det_side / 2.0
    max_q = 4.0 * np.pi / wavelength * det_hside / sd_dist
    qv = np.linspace(-max_q, max_q, 2 * n_pix + 1)
    qx, qy = np.meshgrid(qv, qv)  # default indexing='xy', matches MATLAB meshgrid(qv)
    qvr2 = qx**2 + qy**2
    qvr = np.sqrt(qvr2)

    # --- weight-power override for exact-form-factor shapes (Scatter2D.m:116-127) ---
    matched_name = _match_canonical_shape(phi2)
    effective_weight_power = weight_power_for_name(matched_name) if matched_name is not None else weight_power

    # --- discretize the ensemble once, with the effective weight power (Scatter2D.m:128) ---
    x, p = discretize_ensemble(n_radii, v_rel, dist_type, threshold=0.995, weight_power=effective_weight_power)

    # --- accumulate the scattering-weighted intensity over ensemble members (Scatter2D.m:137-231) ---
    qvr4 = qvr2**2
    is_thin_rod_approx = abs(phi2 - _THIN_ROD_APPROX_PHI2) < _PHI2_MATCH_TOL
    intensity = np.zeros_like(qvr2)
    for i in range(x.size):
        s = rg * x[i]  # physical size of this ensemble member
        s2 = s * s
        s4 = s2 * s2
        if matched_name is not None:
            log_f = EXACT_LOG_FORM_FACTORS[matched_name](qvr, s)
        elif is_thin_rod_approx:
            # Scatter2D.m:199-214: "approximate thin rod" -- recomputes the
            # exponent from scratch with nu forced to 11/225 (discarding the
            # +0.000666 sentinel offset) plus a hardcoded (not
            # steganography-decoded) cubic term, then re-applies the same
            # anomaly-elimination mask below. This completely REPLACES what
            # the generic branch would have computed, it does not build on it.
            quad_term = -(s2 / 3.0) * qvr2
            quart_term = 0.5 * _THIN_ROD_APPROX_NU * s4 * qvr4
            log_f = quad_term + quart_term + (_THIN_ROD_APPROX_PHI3 / 6.0 * s4 * s2) * qvr4 * qvr2
            log_f = np.where(np.abs(quad_term) < np.abs(quart_term), np.nan, log_f)
        else:
            # Generic second-order Guinier expansion (Scatter2D.m:144-160),
            # plus the optional phi3 steganography term and the
            # anomaly-elimination mask, both computed from the quad/quart
            # terms BEFORE the phi3 term is folded in (matching the exact
            # order of operations in Scatter2D.m -- the mask compares only
            # the quadratic and quartic magnitudes, never the cubic term).
            quad_term = -(s2 / 3.0) * qvr2
            quart_term = 0.5 * phi2 * s4 * qvr4
            log_f = quad_term + quart_term
            log_f = log_f + _phi3_steganography_term(phi2, s2, s4, qvr2, qvr4)
            log_f = np.where(np.abs(quad_term) < np.abs(quart_term), np.nan, log_f)
        # Weight-power exponent applies to the dimensionless x[i] (Scatter2D.m's
        # r_vect(ii)), not the physical size s -- Scatter2D.m:227.
        intensity += (x[i] ** effective_weight_power) * p[i] * np.exp(log_f)

    # --- instrument PSF convolution (Scatter2D.m:236) ---
    intensity = smear(intensity, psf0)

    # --- noise (Scatter2D.m:238-247) ---
    if noise < 0:
        intensity = intensity * abs(noise)  # abs(noise) = peak photon count
        intensity = intensity + np.sqrt(np.clip(intensity, 0.0, None)) * rng.standard_normal(intensity.shape)
        intensity = intensity / abs(noise)
    else:
        intensity = intensity + noise * rng.standard_normal(intensity.shape)

    # --- clip negatives (Scatter2D.m:250) ---
    intensity[intensity < 0] = 0.0

    return ScatterResult(qx=qx, qy=qy, intensity=intensity, r_used=rg * x, p_used=p)

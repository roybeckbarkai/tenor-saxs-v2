"""Ensemble radius-of-gyration discretization for TENOR-SAXS.

Ports ``size_distribution_discrete_no_max`` (``Scatter2D.m:781-894``), the
discretizer actually used by the forward-simulation pipeline, plus the
scattering-weighted-variance targeting search from
``run_TENOR_benchmark.m``'s ``target_effective_distribution``/``trial``
(lines 502-529 there).

Distribution family note
-------------------------
MATLAB's own codebase is internally inconsistent about ``'exponential'``:
``Scatter2D.m``'s own copy of the discretizer has no ``'exponential'`` case
(and no ``otherwise`` clause) so it crashes; ``run_TENOR_benchmark.m``'s
separate copy of the same function silently aliases ``'exponential'`` to
the ``'boltzmann'`` (Laplace) shape. Neither is what the name promises,
and a *true* one-sided exponential has a relative variance fixed at
exactly 1 (``Var = Mean**2``) with no free shape parameter -- it cannot
be tuned to an arbitrary target ``V`` at all, which makes it useless for
this package's V-sweeps regardless of which MATLAB behavior it might
otherwise match. Per an explicit decision, ``'exponential'`` is therefore
**not supported** here -- ``discretize_ensemble`` raises ``ValueError``
for it rather than picking one of MATLAB's inconsistent behaviors.

V=0 special case
-----------------
At ``v_rel=0`` several of MATLAB's PDF branches divide by a zero
``sigma`` (``normal``, ``lognormal``, ``boltzmann``, ``triangular``),
producing an all-NaN kernel that crashes downstream ``interp1`` calls
with ``v(0): subscripts must be either integers...`` (confirmed by
running the vendored ``.m`` files under Octave). Only ``uniform``
happens to survive by accident. This port instead treats ``v_rel == 0``
as an explicit, well-defined monodisperse special case for every
distribution family (see the package's internal validation notes).
"""

from __future__ import annotations

import numpy as np

_EPS_NUDGE = 1e-11  # matches MATLAB's `linspace(0, 10e-12, ...)` nudge (same value)


def _fine_pdf(u_fine: np.ndarray, mean: float, sigma: float, v_rel: float, dist_type: str) -> np.ndarray:
    """Unnormalized PDF on the fine grid, per Scatter2D.m:804-833."""
    dt = dist_type.lower()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if dt == "normal":
            w = np.exp(-((u_fine - mean) ** 2) / (2.0 * sigma**2))
        elif dt == "lognormal":
            s = np.sqrt(np.log1p(v_rel))
            m = np.log(mean) - 0.5 * s**2
            w = (1.0 / u_fine) * np.exp(-((np.log(u_fine) - m) ** 2) / (2.0 * s**2))
        elif dt == "schulz":
            z = 1.0 / v_rel - 1.0
            ln_w = z * np.log(u_fine) - (z + 1.0) * u_fine / mean
            ln_w = np.where(np.isnan(ln_w), -np.inf, ln_w)
            ln_w = ln_w - np.nanmax(ln_w[np.isfinite(ln_w)]) if np.any(np.isfinite(ln_w)) else ln_w
            w = np.exp(ln_w)
        elif dt == "boltzmann":
            w = np.exp(-np.sqrt(2.0) * np.abs(u_fine - mean) / sigma)
        elif dt == "triangular":
            length = sigma * np.sqrt(6.0)
            w = np.maximum(0.0, 1.0 - np.abs(u_fine - mean) / length)
        elif dt == "uniform":
            length = sigma * np.sqrt(3.0)
            w = ((u_fine >= mean - length) & (u_fine <= mean + length)).astype(float)
        elif dt == "exponential":
            # Deliberately unsupported -- see module docstring: a true
            # exponential's relative variance is fixed at 1 (not tunable),
            # and MATLAB's own two copies of this distribution disagree on
            # what 'exponential' even means (crash vs. Laplace alias).
            raise ValueError(
                "dist_type='exponential' is not supported: a true exponential's "
                "relative variance is fixed at 1 and cannot be tuned to a target V "
                "(see module docstring for why this differs from MATLAB)."
            )
        else:
            raise ValueError(f"Unknown distribution type: {dist_type!r}")
    w = np.where(np.isnan(w) | np.isinf(w), 0.0, w)
    return w


def discretize_ensemble(
    n: int,
    v_rel: float,
    dist_type: str = "normal",
    threshold: float = 0.9995,
    xmin: float = 0.0,
    weight_power: float = 0.0,
    n_fine: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a mean-1 radius-of-gyration distribution into N representative sizes.

    Port of ``size_distribution_discrete_no_max`` (Scatter2D.m:781-894):
    importance-sampled binning on the scattering-strength-weighted CDF
    (``weight_power``), with each bin's representative radius taken as its
    number-weighted center of mass, followed by an affine rescale that
    forces the realized NUMBER-weighted mean to exactly 1 and the
    NUMBER-weighted relative variance to exactly ``v_rel``.

    Returns
    -------
    x : (n,) representative radii (number-weighted mean=1, number-weighted
        relative variance=v_rel)
    p : (n,) number-probabilities (sum to 1)

    Note: the returned ``p`` is NOT scattering-weighted -- multiply by
    ``x**weight_power`` (and renormalize) to get the scattering-weighted
    ensemble, exactly as ``Scatter2D`` does when building the intensity map.
    """
    mean = 1.0
    if v_rel < 0:
        raise ValueError(f"v_rel must be >= 0, got {v_rel}")
    if v_rel == 0.0:
        # True monodisperse special case -- MATLAB's discretizer divides by a
        # zero sigma here and crashes for most distribution families (see
        # module docstring); this is the well-defined limit instead.
        x = np.full(n, mean, dtype=float)
        p = np.full(n, 1.0 / n, dtype=float)
        return x, p

    sigma = np.sqrt(v_rel) * mean

    if dist_type.lower() in ("lognormal", "schulz", "boltzmann"):
        s_tmp = np.sqrt(np.log1p(v_rel))
        m_weighted = np.log(mean) - 0.5 * s_tmp**2 + weight_power * s_tmp**2
        xmax = np.exp(m_weighted + 5.0 * s_tmp)
    else:
        xmax = mean + 20.0 * sigma

    u_fine = np.linspace(xmin, xmax, n_fine)
    w_fine = _fine_pdf(u_fine, mean, sigma, v_rel, dist_type)
    total = w_fine.sum()
    if total <= 0:
        raise RuntimeError(f"Degenerate PDF for dist_type={dist_type!r}, v_rel={v_rel}")
    w_fine = w_fine / total

    w_weighted = w_fine * (u_fine**weight_power)
    w_weighted_total = w_weighted.sum()
    c_weight = np.cumsum(w_weighted) / w_weighted_total
    c_weight = c_weight + np.linspace(0.0, _EPS_NUDGE, u_fine.size)
    c_number = np.cumsum(w_fine)

    edges_prob = np.linspace(0.0, threshold, n + 1)
    r_edges = np.interp(edges_prob, c_weight, u_fine)

    c_moment1 = np.cumsum(w_fine * u_fine)
    p_edges_number = np.interp(r_edges, u_fine, c_number)
    m1_edges = np.interp(r_edges, u_fine, c_moment1)

    p = np.diff(p_edges_number)
    m1 = np.diff(m1_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(p > 0, m1 / p, 0.5 * (r_edges[:-1] + r_edges[1:]))

    p = np.clip(p, 0.0, None)
    if p.sum() <= 0:
        raise RuntimeError(f"Degenerate binning for dist_type={dist_type!r}, v_rel={v_rel}")

    curr_mu = float(np.sum(p * x))
    curr_var = float(np.sum(p * (x - curr_mu) ** 2))
    if curr_var <= 0:
        raise RuntimeError(f"Zero realized variance before affine fix (dist_type={dist_type!r}, v_rel={v_rel})")
    scale_factor = np.sqrt(v_rel / curr_var)
    x = scale_factor * (x - curr_mu) + mean
    x = np.maximum(x, xmin)
    p = p / p.sum()
    return x, p


def scattering_weighted_stats(x: np.ndarray, p: np.ndarray, weight_power: float) -> tuple[float, float, float]:
    """Scattering-weighted (mean, relative variance, RMS) of a discretized ensemble.

    Reweights the number-probabilities ``p`` by ``x**weight_power``
    (matching ``trial()``'s ``p = p.*(r/max(r)).^w; p=p/sum(p)`` in
    ``run_TENOR_benchmark.m:522-526`` -- the ``/max(r)**w`` there is a
    numerical-stability constant that cancels under the final
    renormalization, so it is omitted here) and computes the resulting
    weighted mean, relative variance ``V = Var/mean**2``, and RMS.
    """
    w = p * np.asarray(x, dtype=float) ** weight_power
    w_sum = w.sum()
    if w_sum <= 0:
        raise RuntimeError("Degenerate scattering weights (all zero)")
    w = w / w_sum
    mu = float(np.sum(w * x))
    var = float(np.sum(w * (x - mu) ** 2))
    rms = float(np.sqrt(np.sum(w * x**2)))
    return mu, var / mu**2, rms


class VarianceTargetUnreachable(RuntimeError):
    """Raised when target_effective_distribution cannot bracket the target V."""


def target_effective_distribution(
    target_v: float,
    target_observed_rg: float,
    dist_name: str,
    n: int,
    weight_power: float,
    p_minimum: float = 1e-5,
    p_maximum: float = 5.0,
    variance_tolerance: float = 2e-4,
    max_iterations: int = 50,
    generator_coverage: float = 0.995,
    apply_weight_again: bool = True,
    expansion_factor: float = 1.6,
    n_scan: int = 40,
) -> dict:
    """Find the discretizer inputs that realize a target SCATTERING-weighted ensemble.

    Port of ``target_effective_distribution``/``trial``/``safe``/``lastvalid``
    (``run_TENOR_benchmark.m:502-529``). ``discretize_ensemble`` only
    guarantees the NUMBER-weighted relative variance equals its input
    exactly; this function bisects over the discretizer's input relative
    std-dev ``pn`` (``v_rel_input = pn**2``) until the realized
    SCATTERING-weighted relative variance matches ``target_v``, then
    back-solves the physical size scale ``requested_rg`` so the realized
    scattering-weighted RMS radius equals ``target_observed_rg``.

    Non-monotonicity safeguard (added after the original MATLAB-parity port):
    the MATLAB original's bisection implicitly ASSUMES ``V(pn)`` (the
    realized scattering-weighted variance as a function of the number-input
    relative std ``pn``) rises monotonically with ``pn``. This is true for
    heavy/semi-infinite-tailed families (normal, lognormal, schulz), but
    empirically FALSE for compact-support families under a high
    ``weight_power`` -- ``uniform`` and ``triangular`` at ``weight_power=6``
    (solid sphere) both RISE then DECLINE as ``pn`` grows past a peak
    (confirmed directly: uniform's scattering-weighted V peaks around
    ``V~=0.016`` near ``pn~=0.22-0.32`` then falls to ``~0.010`` by
    ``pn~=0.55``). Blindly bisecting past that peak can converge to a
    spurious ``pn`` that nominally satisfies ``V(pn)=target_v`` on the WRONG
    (declining) branch, producing a physically confusing result (this was
    the direct cause of `fig3_reproduction.png`'s declining right-panel
    curve for uniform/triangular). Fixed by first SCANNING ``V(pn)`` over
    ``n_scan`` log-spaced points, locating its peak, and bisecting only
    within the verified-monotonic RISING branch up to that peak; if
    ``target_v`` exceeds the achievable peak, this raises
    :class:`VarianceTargetUnreachable` (i.e. the target is genuinely
    unreachable for this distribution family/weight_power/n combination --
    the caller should skip that case, not plot a spurious answer).

    Returns a dict with keys: target_v, realized_v, p_numerical,
    input_variance, requested_rg, target_observed_rg, predicted_observed_rg,
    weighted_rms_factor, x, p (the reweighted, scattering-weighted p).
    """

    def trial(pn: float) -> dict:
        x, p = discretize_ensemble(n, pn**2, dist_name, threshold=generator_coverage, weight_power=weight_power)
        if apply_weight_again:
            w = p * (x / x.max()) ** weight_power
            p = w / w.sum()
        mu = float(np.sum(p * x))
        v = float(np.sum(p * (x - mu) ** 2)) / mu**2
        rms = float(np.sqrt(np.sum(p * x**2)))
        return {"pn": pn, "x": x, "p": p, "mu": mu, "V": v, "rms": rms}

    def safe(pn: float):
        try:
            return True, trial(pn)
        except Exception:
            return False, None

    scan_pns = np.geomspace(p_minimum, p_maximum, n_scan)
    scan_results = [stats for ok, stats in (safe(pn) for pn in scan_pns) if ok]
    if not scan_results:
        raise VarianceTargetUnreachable(
            f"No valid discretization found at all for dist_name={dist_name!r}, weight_power={weight_power}, n={n}"
        )

    v_scan = np.array([r["V"] for r in scan_results])
    peak_idx = int(np.argmax(v_scan))
    peak_v = float(v_scan[peak_idx])
    peak_pn = scan_results[peak_idx]["pn"]

    if target_v > peak_v * (1.0 + 1e-6):
        raise VarianceTargetUnreachable(
            f"target_v={target_v} exceeds the achievable peak scattering-weighted V "
            f"({peak_v:.5f} at pn={peak_pn:.4f}) for dist_name={dist_name!r}, "
            f"weight_power={weight_power}, n={n} -- V(pn) is non-monotonic for this "
            "distribution family/weight_power combination (rises then declines with "
            "pn), so no larger pn can reach a higher V either."
        )

    lo_pn = scan_results[0]["pn"]
    if scan_results[0]["V"] >= target_v:
        best = scan_results[0]
    else:
        lo, hi = lo_pn, peak_pn
        best = scan_results[peak_idx]
        for _ in range(max_iterations):
            mid = 0.5 * (lo + hi)
            ok_mid, mid_stats = safe(mid)
            if not ok_mid:
                hi = mid
                continue
            best = mid_stats
            rel_err = abs(mid_stats["V"] - target_v) / max(target_v, 1e-8)
            if rel_err <= variance_tolerance:
                break
            if mid_stats["V"] < target_v:
                lo = mid
            else:
                hi = mid

    requested_rg = target_observed_rg / best["rms"]
    return {
        "target_v": target_v,
        "realized_v": best["V"],
        "p_numerical": best["pn"],
        "input_variance": best["pn"] ** 2,
        "requested_rg": requested_rg,
        "target_observed_rg": target_observed_rg,
        "predicted_observed_rg": requested_rg * best["rms"],
        "weighted_rms_factor": best["rms"],
        "x": best["x"],
        "p": best["p"],
    }

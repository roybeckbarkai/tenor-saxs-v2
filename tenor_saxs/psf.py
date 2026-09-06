"""Point-spread functions for TENOR-SAXS: the instrument PSF used at
simulation time, and the digital-smearing kernel pair used at analysis
time inside MG_extract.

Convolution convention
-----------------------
MATLAB's ``filter2(h, x, 'same')`` performs 2D *correlation* (not
convolution) with zero-padded boundaries. :func:`smear` reproduces this
via :func:`scipy.ndimage.correlate`. This distinction is moot for our own
kernel families (the Bartlett instrument PSF and the Gaussian smearing
kernels are built from an odd, symmetric 1D window, so correlation and
convolution coincide for them), but it is NOT moot for NaN handling:
:func:`scipy.signal.fftconvolve` was tried first and rejected, because it
computes one global FFT over the whole array, so a single NaN anywhere in
the input silently turns the *entire* output NaN. ``scipy.ndimage.correlate``
is a direct (spatial-domain) operation, so a NaN source pixel only
contaminates output pixels within the kernel's own footprint of it --
matching MATLAB's ``filter2`` exactly. This mattered in practice:
``simulation.scatter2d`` legitimately produces NaN pixels (the
anomaly-elimination masking in Scatter2D.m:160), and the fftconvolve
version silently corrupted every downstream case that hit any NaN at all
(caught by comparing against the Octave reference; see the package's internal validation notes).

``filter2_ungridded`` (below) is a full port of MG_extract.m:589-2106: it
additionally handles curved/non-uniform q-grids (large detectors where
the small-angle q-is-linear-in-pixel-index approximation breaks down) via
center-equivalent angular coordinates, a per-pixel relative solid-angle
correction, and a first-order Taylor expansion of the kernel around each
source/destination pixel's true coordinate displacement -- all with the
same NaN-safe normalized-convolution treatment as :func:`smear`. For a
genuinely regular grid (as ``Scatter2D`` always produces) it reduces to
the same fast path as :func:`smear`; its value is for irregular/curved
grids, e.g. real detector geometries or very wide-angle simulations.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import correlate as _ndimage_correlate
from scipy.ndimage import correlate1d as _correlate1d

_FAST_CORRELATE_SEPARABLE_TOL = 1e-9


def _fast_correlate_same(intensity: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``scipy.ndimage.correlate(intensity, kernel, mode='constant', cval=0.0)``, sped up.

    A direct 2D correlation costs O(rows*cols*kernel_rows*kernel_cols); for
    the smearing kernels this package builds (:func:`gaussian_smearing_kernel`,
    :func:`bartlett2d`), which are ALWAYS exact rank-1 outer products by
    construction and can be as large as ~125 pixels on a side, that direct
    cost is the dominant runtime cost of the whole analysis pipeline (a
    single such call was measured at several seconds). If ``kernel`` is
    (numerically) rank-1, this instead does two successive 1D correlations
    -- mathematically identical for a separable kernel, but
    O(rows*cols*(kernel_rows+kernel_cols)), ~100x fewer operations for a
    125x125 kernel. Falls back to the direct 2D correlation for any
    kernel that isn't (numerically) separable.
    """
    U, S, Vh = np.linalg.svd(kernel, full_matrices=False)
    total_energy = np.linalg.norm(S)
    residual_energy = np.linalg.norm(S[1:]) if S.size > 1 else 0.0
    if total_energy <= 0 or residual_energy / max(total_energy, np.finfo(float).eps) > _FAST_CORRELATE_SEPARABLE_TOL:
        return _ndimage_correlate(intensity, kernel, mode="constant", cval=0.0)
    gy = U[:, 0] * S[0]  # outer(gy, gx) == kernel exactly (kernel is rank-1 to within tolerance)
    gx = Vh[0, :]
    temp = _correlate1d(intensity, gy, axis=0, mode="constant", cval=0.0)
    return _correlate1d(temp, gx, axis=1, mode="constant", cval=0.0)


def tri1d(n: int, mode: str = "raised") -> np.ndarray:
    """1D triangular window (init_TENOR_params.m:56-77)."""
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.array([1.0])
    idx = np.arange(n, dtype=float)
    c = (n - 1) / 2.0
    if mode == "zero":
        if n == 2:
            return np.array([1.0, 1.0])
        return 1.0 - np.abs(idx - c) / c
    d = (n + 1) / 2.0
    w = 1.0 - np.abs(idx - c) / d
    return np.clip(w, 0.0, None)


def bartlett2d(n: int, m: int | None = None, mode: str = "raised") -> np.ndarray:
    """Normalized 2D triangular (Bartlett-like) kernel (init_TENOR_params.m:40-54).

    ``mode='raised'`` (default) keeps nonzero edge weights; ``mode='zero'``
    is the classic Bartlett window with zero endpoints.
    """
    if m is None:
        m = n
    wy = tri1d(n, mode)
    wx = tri1d(m, mode)
    kernel = np.outer(wy, wx)
    total = kernel.sum()
    if total > 0:
        kernel = kernel / total
    return kernel


def gaussian_smearing_kernel(pxx: int, pxy: int, signum: float = 4.0) -> np.ndarray:
    """Separable anisotropic-Gaussian digital-smearing kernel.

    Port of the kernel construction inlined twice in ``MG_extract.m``
    (lines 75-97 for the wide kernel, 118-135 for the narrow one):
    ``H = exp(-linspace(-signum,signum,pxy).^2/2)' * exp(-linspace(-signum,signum,pxx).^2/2)``,
    normalized to sum 1. ``pxx``/``pxy`` are the odd pixel counts along the
    detector's x (columns) and y (rows) axes.
    """
    if pxx < 1 or pxy < 1 or pxx % 2 == 0 or pxy % 2 == 0:
        raise ValueError(f"pxx, pxy must be positive odd integers, got pxx={pxx}, pxy={pxy}")
    gx = np.exp(-(np.linspace(-signum, signum, pxx) ** 2) / 2.0)
    gy = np.exp(-(np.linspace(-signum, signum, pxy) ** 2) / 2.0)
    kernel = np.outer(gy, gx)
    return kernel / kernel.sum()


def validate_pxn(pxn: np.ndarray) -> np.ndarray:
    """Validate a 4-element smearing pixel quartet [nx1, ny1, nx2, ny2].

    Port of the Pxn validation in ``MG_extract.m:55-66``: all four entries
    must be positive odd integers, the pair (nx1,ny1) must differ as a set
    from (nx2,ny2), and neither pair may be internally symmetric unless
    both entries equal 1. Raises ``ValueError`` on violation (MATLAB
    instead silently falls back to hardcoded defaults on this failure --
    this port makes the failure explicit; see the package's internal validation notes).
    """
    pxn = np.asarray(pxn, dtype=int)
    if pxn.shape != (4,):
        raise ValueError(f"Pxn must have 4 elements, got shape {pxn.shape}")
    if np.any(pxn < 1) or np.any(pxn % 2 == 0):
        raise ValueError(f"All Pxn entries must be positive odd integers, got {pxn}")
    pair1, pair2 = set(pxn[:2].tolist()), set(pxn[2:].tolist())
    if pair1 == pair2:
        raise ValueError(f"Pxn pairs must differ as sets: {pxn[:2]} vs {pxn[2:]}")
    if pxn[0] == pxn[1] and pxn[0] * pxn[1] != 1:
        raise ValueError(f"Narrow-kernel pair must not be internally symmetric (unless [1,1]): {pxn[:2]}")
    if pxn[2] == pxn[3] and pxn[2] * pxn[3] != 1:
        raise ValueError(f"Wide-kernel pair must not be internally symmetric (unless [1,1]): {pxn[2:]}")
    return pxn


def fit_kernel_sigma_pixels(kernel: np.ndarray, axis: int) -> float:
    """Recover a Gaussian kernel's sigma (in pixels) along one axis from its own analytic profile.

    Port of the lightweight (non-robust) sigma recovery applied directly
    to the kernel array itself in ``MG_extract.m:94-97,132-135``: take the
    kernel's central row (``axis=1``, x-direction) or column (``axis=0``,
    y-direction), fit ``log(profile)`` to a degree-2 polynomial, and
    convert the quadratic coefficient to a sigma. This is deliberately the
    simple non-robust fit (matching what MATLAB actually calls at these
    two sites), not the more elaborate ``fit_log_gaussian``/
    ``gaussian_sigma_from_fit`` machinery used elsewhere in MG_extract.m.
    """
    if axis == 1:
        n = kernel.shape[1]
        profile = kernel[kernel.shape[0] // 2, :]
    else:
        n = kernel.shape[0]
        profile = kernel[:, kernel.shape[1] // 2]
    idx = np.arange(1, n + 1, dtype=float)
    with np.errstate(divide="ignore"):
        log_profile = np.log(profile)
    coeffs = np.polyfit(idx, log_profile, 2)
    a = coeffs[0]
    if a >= 0:
        return np.inf
    return np.sqrt(-1.0 / (2.0 * a))


def smear(intensity: np.ndarray, kernel: np.ndarray, use_single: bool = False) -> np.ndarray:
    """Convolve ``intensity`` with ``kernel`` (regular-grid fast path, see module docstring).

    Uses :func:`scipy.ndimage.correlate` (a direct, spatially-local
    correlation with zero-padded boundaries), matching MATLAB's
    ``filter2(kernel, intensity, 'same')`` exactly -- both by construction
    (correlation, not convolution; our kernels are symmetric so the two
    coincide anyway) and, critically, in how NaN pixels propagate: a NaN
    only contaminates output pixels within the kernel's own footprint of
    that source pixel. This matters because ``simulation.scatter2d`` can
    legitimately produce NaN pixels (the anomaly-elimination masking in
    Scatter2D.m:160). ``scipy.signal.fftconvolve`` was tried first and
    rejected: it computes a single global FFT over the whole array, so a
    lone NaN anywhere in the input silently turns the ENTIRE output NaN
    -- a real bug caught by comparing against the Octave reference (see
    the internal validation notes).

    ``use_single``: MATLAB's ``filter2_ungridded`` (which ``MG_extract.m``
    always calls, even on the exact-shortcut path this function otherwise
    reproduces) casts ``H``/``I``/``X``/``Y`` to ``single`` (float32) before
    filtering -- and, through MATLAB's single-contaminates-double type
    propagation, every computation downstream of that point (the log-ratio,
    the polynomial fit, ...) stays in float32 for the rest of that MATLAB
    call. Pass ``use_single=True`` to replicate the convolution step's own
    rounding for MATLAB-parity comparisons; the result is still returned as
    float64 (this port does not propagate float32 through the rest of the
    pipeline the way MATLAB's type system does -- see the package's internal validation notes).
    """
    if use_single:
        intensity = intensity.astype(np.float32)
        kernel = kernel.astype(np.float32)
    result = _fast_correlate_same(intensity, kernel)
    return result.astype(np.float64)


# =============================================================================
# filter2_ungridded: PSF application on a possibly curved/non-uniform q-grid
# =============================================================================
#
# Full port of MG_extract.m:589-2106 (plus its local helpers 2109-2451).
# Physical background (condensed from that file's extensive header comment,
# MG_extract.m:591-1018):
#
# H is a bi-Gaussian PSF sampled in "center-equivalent" q-pixel units at the
# detector center. The physical PSF is fixed in scattering SOLID ANGLE, not
# in q-space, and q is nonlinear in scattering angle away from the beam
# center (q = (4*pi/lambda)*sin(alpha/2)). For a detector where this
# curvature matters (e.g. a wide-angle or otherwise large detector where the
# small-angle q-is-linear-in-pixel-index approximation breaks down), this
# function:
#   1. maps (qx,qy) to "center-equivalent angular coordinates" (QX,QY) that
#      equal q to first order near the beam center but track the true
#      angular PSF away from it;
#   2. computes each pixel's relative solid angle and folds it into a
#      NaN-safe NORMALIZED convolution (numerator/denominator quadrature,
#      MG_extract.m:695-992) so that missing/invalid samples get zero
#      statistical weight rather than being treated as measured zeros;
#   3. for a genuinely regular, correctly-scaled grid, short-circuits to the
#      same fast path as :func:`smear`;
#   4. otherwise, applies a first-order Taylor expansion of the (necessarily
#      separable) kernel around each pixel's coordinate displacement from a
#      nominal regular grid.
_DEFAULT_ALLOW_SHORTCUT = True
_DEFAULT_REL_CUTOFF = 1e-3
_GRID_REGULAR_TOL = 1e-3
_GRID_SCALE_TOL = 1e-2
_KERNEL_SEPARABLE_TOL = 1e-5
_GAUSSIAN_FIT_TOL = 1e-4
_WEIGHT_FLOOR = 1e-12
_DENOMINATOR_FLOOR = 1e-12
_SOLID_ANGLE_FLOOR = 1e-6
_GRADIENT_SAFETY_FACTOR = 1.25
_MAX_FIRST_ORDER_SHIFT = 0.25
_CENTER_HALF_WIDTH = 2


def _index_derivatives(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of a 2D field w.r.t. detector column (``dA_dc``) and row (``dA_dr``) index.

    Port of ``index_derivatives`` (MG_extract.m:2128-2170): central
    differences in the interior, one-sided at the first/last row or column.
    """
    rows, cols = a.shape
    da_dc = np.zeros((rows, cols))
    da_dr = np.zeros((rows, cols))
    if cols > 1:
        da_dc[:, 0] = a[:, 1] - a[:, 0]
        da_dc[:, -1] = a[:, -1] - a[:, -2]
    if cols > 2:
        da_dc[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
    if rows > 1:
        da_dr[0, :] = a[1, :] - a[0, :]
        da_dr[-1, :] = a[-1, :] - a[-2, :]
    if rows > 2:
        da_dr[1:-1, :] = 0.5 * (a[2:, :] - a[:-2, :])
    return da_dc, da_dr


def _fit_log_gaussian(
    coordinate: np.ndarray, profile: np.ndarray, weight_floor: float, fit_tolerance: float
) -> tuple[np.ndarray, bool]:
    """Fit ``log(profile) = p0*q**2 + p1*q + p2``; accept only if the reconstructed,
    normalized profile matches the input to within ``fit_tolerance`` (relative, max-norm).

    Port of ``fit_log_gaussian`` (MG_extract.m:2173-2263).
    """
    coordinate = np.asarray(coordinate, dtype=float).ravel()
    profile = np.asarray(profile, dtype=float).ravel()

    if profile.size < 3 or np.unique(coordinate[np.isfinite(coordinate)]).size < 3:
        return np.zeros(3), False

    profile_scale = np.max(np.abs(profile))
    valid = (
        np.isfinite(coordinate)
        & np.isfinite(profile)
        & (profile > 0)
        & (profile >= weight_floor * max(profile_scale, np.finfo(float).eps))
    )
    if np.count_nonzero(valid) < 3:
        return np.zeros(3), False

    p = np.polyfit(coordinate[valid], np.log(profile[valid]), 2)  # [p0, p1, p2]
    fitted_log_profile = (p[0] * coordinate + p[1]) * coordinate + p[2]
    fitted_profile = np.exp(fitted_log_profile)

    fitted_sum = np.sum(fitted_profile)
    reference_sum = np.sum(profile)
    if not np.isfinite(fitted_sum) or fitted_sum <= 0 or not np.isfinite(reference_sum) or reference_sum == 0:
        return np.zeros(3), False

    fitted_profile = fitted_profile / fitted_sum
    reference_profile = profile / reference_sum
    relative_fit_error = np.max(np.abs(fitted_profile - reference_profile)) / max(
        np.max(np.abs(reference_profile)), np.finfo(float).eps
    )
    accepted = bool(np.isfinite(relative_fit_error) and relative_fit_error <= fit_tolerance)
    return p, accepted


def _gaussian_sigma_from_fit(p: np.ndarray) -> float:
    """``sigma = sqrt(-1/(2*p[0]))`` if ``p[0] < 0`` else ``inf`` (MG_extract.m:2264-2281)."""
    return float(np.sqrt(-1.0 / (2.0 * p[0]))) if p[0] < 0 else float(np.inf)


def _positive_median_step(offsets: np.ndarray) -> float:
    """Median of finite, positive, nonzero coordinate differences (MG_extract.m:2433-2451)."""
    d = np.abs(np.diff(np.asarray(offsets, dtype=float).ravel()))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 1.0


def _gradient_support_1d(
    weights: np.ndarray,
    offsets: np.ndarray,
    intensity: np.ndarray,
    valid_mask: np.ndarray,
    dimension: int,
    rel_cutoff: float,
    safety_factor: float,
) -> tuple[int, int]:
    """Image-gradient-aware adaptive kernel-support truncation.

    Port of ``gradient_support_1d`` (MG_extract.m:2284-2430). ``dimension``
    is ``0`` for the row/vertical axis, ``1`` for the column/horizontal axis
    (MATLAB's ``1``/``2`` respectively -- shifted by one for Python's 0-based
    axis convention). Returns 0-based, INCLUSIVE ``(first_index, last_index)``
    bounds into ``weights``/``offsets``.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    offsets = np.asarray(offsets, dtype=float).ravel()
    n = weights.size
    center_index = (n - 1) // 2  # 0-based; matches MATLAB's ceil(n/2) 1-based center, shifted by 1

    first_index, last_index = 0, n - 1

    valid_values = np.abs(intensity[valid_mask])
    if valid_values.size == 0:
        return first_index, last_index
    signal_scale = np.max(valid_values) if valid_values.size else np.nan
    if not np.isfinite(signal_scale):
        return first_index, last_index
    if signal_scale <= np.finfo(float).eps:
        return center_index, center_index

    coordinate_step = _positive_median_step(offsets)

    if dimension == 0:
        intensity_difference = np.diff(intensity, axis=0)
        valid_pairs = valid_mask[:-1, :] & valid_mask[1:, :]
    elif dimension == 1:
        intensity_difference = np.diff(intensity, axis=1)
        valid_pairs = valid_mask[:, :-1] & valid_mask[:, 1:]
    else:
        raise ValueError("dimension must be 0 or 1")

    valid_differences = np.abs(intensity_difference[valid_pairs])
    if valid_differences.size == 0:
        gradient_bound = 0.0
    else:
        gradient_bound = float(np.max(valid_differences)) / max(coordinate_step, np.finfo(float).eps)
    gradient_bound *= safety_factor

    estimated_change = np.minimum(2.0 * signal_scale, gradient_bound * np.abs(offsets))
    estimated_contribution = np.abs(weights) * estimated_change / signal_scale
    omitted_contribution = float(np.sum(estimated_contribution) - estimated_contribution[center_index])

    left_index, right_index = center_index, center_index
    while omitted_contribution > rel_cutoff and (left_index > 0 or right_index < n - 1):
        left_contribution = estimated_contribution[left_index - 1] if left_index > 0 else -np.inf
        right_contribution = estimated_contribution[right_index + 1] if right_index < n - 1 else -np.inf
        if left_contribution >= right_contribution:
            left_index -= 1
            omitted_contribution -= estimated_contribution[left_index]
        else:
            right_index += 1
            omitted_contribution -= estimated_contribution[right_index]

    half_width = min(center_index - first_index, last_index - center_index)
    return center_index - half_width, center_index + half_width


def _separable_filter2(gy: np.ndarray, gx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Equivalent to MATLAB's ``filter2(gy*gx.', x, 'same')`` for a rank-1 kernel.

    Port of ``separable_filter2`` (MG_extract.m:2109-2125): two successive 1D
    CORRELATIONS (not convolutions -- ``scipy.ndimage.correlate1d`` already
    matches ``filter2``'s orientation, so unlike the MATLAB original this
    needs no kernel-flip/``conv2`` workaround), zero-padded, matching
    ``filter2``'s ``'same'`` boundary convention.
    """
    temp = _correlate1d(x, np.asarray(gy, dtype=float), axis=0, mode="constant", cval=0.0)
    return _correlate1d(temp, np.asarray(gx, dtype=float), axis=1, mode="constant", cval=0.0)


def filter2_ungridded(
    kernel: np.ndarray,
    intensity: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    wavelength: float,
    allow_shortcut: bool = _DEFAULT_ALLOW_SHORTCUT,
    dx_h: float | None = None,
    dy_h: float | None = None,
    rel_cutoff: float = _DEFAULT_REL_CUTOFF,
    use_single: bool = False,
) -> np.ndarray:
    """Filter detector data with a center-q Gaussian PSF on a possibly curved/non-uniform q-grid.

    Full port of ``filter2_ungridded`` (MG_extract.m:589-2106). See the
    module-level comment above this function for a summary of the physics
    (center-equivalent angular coordinates, relative solid-angle
    normalization, first-order Taylor-corrected separable filtering) and
    :mod:`tenor_saxs.psf`'s module docstring for why the underlying
    kernel application (both here and in :func:`smear`) uses a direct
    spatial correlation rather than an FFT-based one (NaN locality).

    Parameters
    ----------
    kernel:
        The PSF, sampled in center-equivalent q-pixel units (``dx_h``/
        ``dy_h`` apart). Must be exactly (within ``1e-5`` relative rank-1
        residual energy) separable if the curved-grid general path is
        needed -- true for every kernel this package builds
        (:func:`gaussian_smearing_kernel`, :func:`bartlett2d`), since both
        are exact outer products by construction.
    intensity:
        Detector intensity/counts, one equal-area physical pixel each.
    qx, qy:
        q-coordinates of each detector pixel, same shape as ``intensity``.
    wavelength:
        In the units reciprocal to ``qx``/``qy`` (e.g. nm if q is in
        1/nm).
    allow_shortcut:
        If ``True`` (default) and the grid is regular with matching kernel
        pitch, use the fast (plain or NaN-safe normalized) path directly.
    dx_h, dy_h:
        Center-equivalent q-pixel pitch of ``kernel``. Defaults to the
        detector's own estimated center pitch (one kernel sample per
        detector pixel at q=0).
    rel_cutoff:
        Relative error budget for adaptively truncating the kernel's
        support based on the image's own local gradient
        (:func:`_gradient_support_1d`); ``0`` disables truncation
        (MG_extract's own call sites always pass ``0``). MATLAB's function
        default is ``1e-3``, kept here for standalone-call fidelity.
    use_single:
        MATLAB casts ``H``/``I``/``X``/``Y`` to ``single`` (float32) at the
        very start of this function (MG_extract.m:1110-1113), and its
        single-contaminates-double type propagation keeps everything
        downstream in that call in float32 too. Pass ``True`` to replicate
        that rounding for MATLAB-parity comparisons; the internal
        computation still promotes back to float64 partway through (NumPy
        does not have MATLAB's automatic type-contamination), and the
        return value is always float64. See :func:`smear`'s docstring and
        the internal validation notes.

    Returns
    -------
    np.ndarray
        Filtered intensity, same shape as ``intensity``, with ``NaN`` where
        the normalized denominator is too small to trust.
    """
    if not np.isscalar(rel_cutoff) or not np.isfinite(rel_cutoff) or rel_cutoff < 0:
        raise ValueError("rel_cutoff must be a finite nonnegative scalar.")
    if not np.isscalar(wavelength) or not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be a finite positive scalar.")

    _dtype = np.float32 if use_single else np.float64
    H = np.asarray(kernel, dtype=_dtype).astype(np.float64)
    I = np.asarray(intensity, dtype=_dtype).astype(np.float64)
    X = np.asarray(qx, dtype=_dtype).astype(np.float64)
    Y = np.asarray(qy, dtype=_dtype).astype(np.float64)
    rows, cols = I.shape
    if X.shape != I.shape or Y.shape != I.shape:
        raise ValueError("I, X, and Y must have identical shapes.")
    if H.size == 0:
        raise ValueError("H must be nonempty.")

    eps = np.finfo(float).eps

    # -------- normalize the PSF (MG_extract.m:1128-1139) ---------------------
    h_sum = np.sum(H)
    if not np.isfinite(h_sum) or abs(h_sum) <= _WEIGHT_FLOOR:
        raise ValueError("H must have a finite nonzero sum.")
    H = H / h_sum

    # -------- q -> center-equivalent angular coordinates (MG_extract.m:1141-1218) --
    coordinate_valid = np.isfinite(X) & np.isfinite(Y)
    q_magnitude = np.hypot(X, Y)
    asin_argument = wavelength * q_magnitude / (4.0 * np.pi)
    physical_q = coordinate_valid & (asin_argument >= 0) & (asin_argument < 1)

    if np.any(coordinate_valid) and np.all(physical_q[coordinate_valid]):
        angular_mode = True
        alpha = 2.0 * np.arcsin(asin_argument)
        center_q_per_radian = 2.0 * np.pi / wavelength
        direction_x = np.zeros((rows, cols))
        direction_y = np.zeros((rows, cols))
        nonzero_q = coordinate_valid & (q_magnitude > eps)
        direction_x[nonzero_q] = X[nonzero_q] / q_magnitude[nonzero_q]
        direction_y[nonzero_q] = Y[nonzero_q] / q_magnitude[nonzero_q]
        QX = center_q_per_radian * alpha * direction_x
        QY = center_q_per_radian * alpha * direction_y
        center_pixels = coordinate_valid & ~nonzero_q
        QX[center_pixels] = 0.0
        QY[center_pixels] = 0.0
        QX[~coordinate_valid] = np.nan
        QY[~coordinate_valid] = np.nan
    else:
        angular_mode = False
        QX = X.copy()
        QY = Y.copy()
        alpha = np.zeros((rows, cols))
        warnings.warn(
            "filter2_ungridded: some q values do not satisfy lambda*q/(4*pi) < 1. "
            "X and Y are treated directly as filter coordinates and solid-angle "
            "conversion is disabled.",
            stacklevel=2,
        )

    # -------- estimate center-reference detector spacing (MG_extract.m:1220-1414) --
    q_squared = X**2 + Y**2
    q_squared = np.where(coordinate_valid, q_squared, np.inf)
    center_flat = int(np.argmin(q_squared))
    if not np.isfinite(q_squared.flat[center_flat]):
        raise ValueError("No finite X,Y coordinate pair is available.")
    center_row, center_col = np.unravel_index(center_flat, (rows, cols))

    r0 = max(0, center_row - _CENTER_HALF_WIDTH)
    r1 = min(rows, center_row + _CENTER_HALF_WIDTH + 1)
    c0 = max(0, center_col - _CENTER_HALF_WIDTH)
    c1 = min(cols, center_col + _CENTER_HALF_WIDTH + 1)

    dqx_matrix = np.diff(QX, axis=1)
    dqy_matrix = np.diff(QY, axis=0)
    dqy_dcol_matrix = np.diff(QY, axis=1)
    dqx_drow_matrix = np.diff(QX, axis=0)

    qx_region = QX[r0:r1, c0:c1]
    qy_region = QY[r0:r1, c0:c1]
    dqx_center = np.diff(qx_region, axis=1)
    dqy_center = np.diff(qy_region, axis=0)
    dqy_dcol_center = np.diff(qy_region, axis=1)
    dqx_drow_center = np.diff(qx_region, axis=0)

    valid_dqx_center = dqx_center[np.isfinite(dqx_center) & (dqx_center != 0)]
    valid_dqy_center = dqy_center[np.isfinite(dqy_center) & (dqy_center != 0)]
    valid_dqy_dcol_center = dqy_dcol_center[np.isfinite(dqy_dcol_center)]
    valid_dqx_drow_center = dqx_drow_center[np.isfinite(dqx_drow_center)]

    if valid_dqx_center.size:
        dqx_grid = float(np.median(valid_dqx_center))
    else:
        valid_dqx_all = dqx_matrix[np.isfinite(dqx_matrix) & (dqx_matrix != 0)]
        if valid_dqx_all.size == 0:
            raise ValueError("Could not determine a finite nonzero horizontal center-reference coordinate spacing.")
        dqx_grid = float(np.median(valid_dqx_all))
        warnings.warn(
            "filter2_ungridded: no valid horizontal spacing in the central region; "
            "using the median spacing over the complete detector.",
            stacklevel=2,
        )

    if valid_dqy_center.size:
        dqy_grid = float(np.median(valid_dqy_center))
    else:
        valid_dqy_all = dqy_matrix[np.isfinite(dqy_matrix) & (dqy_matrix != 0)]
        if valid_dqy_all.size == 0:
            raise ValueError("Could not determine a finite nonzero vertical center-reference coordinate spacing.")
        dqy_grid = float(np.median(valid_dqy_all))
        warnings.warn(
            "filter2_ungridded: no valid vertical spacing in the central region; "
            "using the median spacing over the complete detector.",
            stacklevel=2,
        )

    if not np.isfinite(dqx_grid) or dqx_grid == 0:
        raise ValueError("Could not determine a finite nonzero horizontal center-reference coordinate spacing.")
    if not np.isfinite(dqy_grid) or dqy_grid == 0:
        raise ValueError("Could not determine a finite nonzero vertical center-reference coordinate spacing.")

    dqy_dcol_grid = float(np.median(valid_dqy_dcol_center)) if valid_dqy_dcol_center.size else 0.0
    dqx_drow_grid = float(np.median(valid_dqx_drow_center)) if valid_dqx_drow_center.size else 0.0

    dx_h_used = dqx_grid if dx_h is None else float(dx_h)
    dy_h_used = dqy_grid if dy_h is None else float(dy_h)
    if not np.isfinite(dx_h_used) or dx_h_used == 0:
        raise ValueError("dx_h must be a finite nonzero scalar.")
    if not np.isfinite(dy_h_used) or dy_h_used == 0:
        raise ValueError("dy_h must be a finite nonzero scalar.")

    # -------- grid regularity / scale-match classification (MG_extract.m:1482-1566) --
    horizontal_grid_error = np.nanmax(np.abs(dqx_matrix - dqx_grid)) / max(abs(dqx_grid), eps)
    vertical_grid_error = np.nanmax(np.abs(dqy_matrix - dqy_grid)) / max(abs(dqy_grid), eps)
    horizontal_cross_error = np.nanmax(np.abs(dqy_dcol_matrix - dqy_dcol_grid)) / max(abs(dqy_grid), eps)
    vertical_cross_error = np.nanmax(np.abs(dqx_drow_matrix - dqx_drow_grid)) / max(abs(dqx_grid), eps)
    central_cross_error = max(abs(dqy_dcol_grid) / max(abs(dqy_grid), eps), abs(dqx_drow_grid) / max(abs(dqx_grid), eps))

    is_regular = (
        horizontal_grid_error <= _GRID_REGULAR_TOL
        and vertical_grid_error <= _GRID_REGULAR_TOL
        and horizontal_cross_error <= _GRID_REGULAR_TOL
        and vertical_cross_error <= _GRID_REGULAR_TOL
        and central_cross_error <= _GRID_REGULAR_TOL
    )

    # NOTE: MATLAB defines scale_match TWICE (MG_extract.m:1544-1565); the second
    # definition (without the extra abs() around the pitch difference itself,
    # only around each pitch individually via the outer abs()) overwrites and is
    # the one actually used. Ported as the active (second) definition.
    scale_match = (
        abs(dqx_grid - dx_h_used) / max(abs(dx_h_used), eps) <= _GRID_SCALE_TOL
        and abs(dqy_grid - dy_h_used) / max(abs(dy_h_used), eps) <= _GRID_SCALE_TOL
    )

    # -------- relative solid angle per detector pixel (MG_extract.m:1567-1630) --
    apply_solid_angle = True  # MATLAB's APPLY_SOLID_ANGLE is hardcoded True
    if apply_solid_angle and angular_mode:
        dqx_dc, dqx_dr = _index_derivatives(QX)
        dqy_dc, dqy_dr = _index_derivatives(QY)
        coordinate_area = np.abs(dqx_dc * dqy_dr - dqx_dr * dqy_dc)
        sphere_factor = np.ones((rows, cols))
        nonzero_alpha = np.isfinite(alpha) & (np.abs(alpha) > np.sqrt(eps))
        sphere_factor[nonzero_alpha] = np.sin(alpha[nonzero_alpha]) / alpha[nonzero_alpha]
        solid_angle = sphere_factor * coordinate_area
        solid_angle = np.where(coordinate_valid, solid_angle, np.nan)

        candidate = solid_angle[coordinate_valid & np.isfinite(solid_angle) & (solid_angle > 0)]
        if candidate.size == 0:
            raise ValueError("Could not calculate a valid solid-angle map.")
        nominal_solid_angle = float(np.median(candidate))
        if not np.isfinite(nominal_solid_angle) or nominal_solid_angle <= 0:
            raise ValueError("Could not determine a nominal solid angle.")

        solid_angle_ratio = solid_angle / nominal_solid_angle
        bad = ~np.isfinite(solid_angle_ratio) | (solid_angle_ratio < _SOLID_ANGLE_FLOOR)
        solid_angle_ratio = np.where(bad, np.nan, solid_angle_ratio)
    else:
        solid_angle_ratio = np.ones((rows, cols))

    # -------- validity mask (MG_extract.m:1632-1651) --------------------------
    intensity_valid = np.isfinite(I)
    geometry_valid = np.isfinite(QX) & np.isfinite(QY) & np.isfinite(solid_angle_ratio) & (solid_angle_ratio > 0)
    valid_mask = intensity_valid & geometry_valid
    has_invalid = not bool(np.all(valid_mask))

    # -------- regular-grid shortcut (MG_extract.m:1653-1730) -------------------
    if allow_shortcut and is_regular and scale_match:
        solid_angle_values = solid_angle_ratio[valid_mask]
        if solid_angle_values.size == 0:
            return np.full((rows, cols), np.nan)

        solid_angle_is_uniform = np.max(np.abs(solid_angle_values - 1.0)) <= _GRID_REGULAR_TOL

        if not has_invalid and solid_angle_is_uniform:
            return _fast_correlate_same(I, H)

        numerator_source = np.where(valid_mask, I, 0.0)
        denominator_source = np.where(valid_mask, solid_angle_ratio, 0.0)

        numerator = _fast_correlate_same(numerator_source, H)
        denominator = _fast_correlate_same(denominator_source, H)

        # NORMALIZE_EDGES is hardcoded False in MATLAB -> this branch always runs.
        inside_kernel_weight = _fast_correlate_same(np.ones((rows, cols)), H)
        denominator = denominator + 1.0 - inside_kernel_weight

        with np.errstate(invalid="ignore", divide="ignore"):
            filtered = numerator / denominator
        invalid_output = ~np.isfinite(denominator) | (denominator <= _DENOMINATOR_FLOOR)
        filtered = np.where(invalid_output, np.nan, filtered)
        return filtered

    # -------- extract separable rank-1 factors from H (MG_extract.m:1732-1791) --
    kernel_rows, kernel_cols = H.shape
    row_center = np.ceil(kernel_rows / 2.0)
    col_center = np.ceil(kernel_cols / 2.0)

    U, S, Vh = np.linalg.svd(H.astype(np.float64), full_matrices=False)
    total_singular_energy = np.linalg.norm(S)
    residual_singular_energy = np.linalg.norm(S[1:]) if S.size > 1 else 0.0
    separability_error = residual_singular_energy / max(total_singular_energy, eps)
    if separability_error > _KERNEL_SEPARABLE_TOL:
        raise ValueError(f"H is not sufficiently rank-1 separable. Relative rank-1 residual: {separability_error:.3e}.")

    sqrt_s1 = np.sqrt(S[0])
    gy = U[:, 0] * sqrt_s1
    gx = Vh[0, :] * sqrt_s1

    if np.sum(gy) < 0:
        gy, gx = -gy, -gx
    gy_sum, gx_sum = np.sum(gy), np.sum(gx)
    if abs(gy_sum) <= _WEIGHT_FLOOR or abs(gx_sum) <= _WEIGHT_FLOOR:
        raise ValueError("Separable PSF factors have a near-zero sum.")
    gy = gy / gy_sum
    gx = gx / gx_sum

    qy_kernel = (np.arange(1, kernel_rows + 1) - row_center) * dy_h_used
    qx_kernel = (np.arange(1, kernel_cols + 1) - col_center) * dx_h_used

    # -------- select I-dependent Gaussian support (MG_extract.m:1804-1837) -----
    if rel_cutoff > 0:
        pass_cutoff = rel_cutoff / 2.0
        row_first, row_last = _gradient_support_1d(
            gy, qy_kernel, I, valid_mask, 0, pass_cutoff, _GRADIENT_SAFETY_FACTOR
        )
        col_first, col_last = _gradient_support_1d(
            gx, qx_kernel, I, valid_mask, 1, pass_cutoff, _GRADIENT_SAFETY_FACTOR
        )
    else:
        row_first, row_last = 0, kernel_rows - 1
        col_first, col_last = 0, kernel_cols - 1

    gy = gy[row_first : row_last + 1]
    gx = gx[col_first : col_last + 1]
    qy_kernel = qy_kernel[row_first : row_last + 1]
    qx_kernel = qx_kernel[col_first : col_last + 1]
    gy = gy / np.sum(gy)
    gx = gx / np.sum(gx)

    # -------- derivative filters for the retained support (MG_extract.m:1839-1919) --
    if gx.size == 1:
        px, accepted_x = np.zeros(3), True
        gx, dgx, sigma_x = np.array([1.0]), np.array([0.0]), np.inf
    else:
        px, accepted_x = _fit_log_gaussian(qx_kernel, gx, _WEIGHT_FLOOR, _GAUSSIAN_FIT_TOL)
        if accepted_x:
            dgx = (2.0 * px[0] * qx_kernel + px[1]) * gx
            dgx = dgx - np.sum(dgx) * gx
            sigma_x = _gaussian_sigma_from_fit(px)
        else:
            sigma_x = np.inf
            dgx = np.zeros_like(gx)

    if gy.size == 1:
        py, accepted_y = np.zeros(3), True
        gy, dgy, sigma_y = np.array([1.0]), np.array([0.0]), np.inf
    else:
        py, accepted_y = _fit_log_gaussian(qy_kernel, gy, _WEIGHT_FLOOR, _GAUSSIAN_FIT_TOL)
        if accepted_y:
            dgy = (2.0 * py[0] * qy_kernel + py[1]) * gy
            dgy = dgy - np.sum(dgy) * gy
            sigma_y = _gaussian_sigma_from_fit(py)
        else:
            sigma_y = np.inf
            dgy = np.zeros_like(gy)

    if not accepted_x or not accepted_y:
        raise ValueError(
            "The retained PSF support is not sufficiently Gaussian. "
            f"Horizontal support: {gx.size} samples. Vertical support: {gy.size} samples."
        )

    # -------- nominal regular coordinate grid & displacement (MG_extract.m:1920-1977) --
    column_index = np.arange(cols, dtype=float)
    row_index = np.arange(rows, dtype=float)[:, None]

    qx_origin_samples = QX - column_index[None, :] * dqx_grid
    qy_origin_samples = QY - row_index * dqy_grid
    qx_origin = float(np.nanmedian(qx_origin_samples))
    qy_origin = float(np.nanmedian(qy_origin_samples))

    QX_nominal = qx_origin + column_index[None, :] * dqx_grid
    QY_nominal = qy_origin + row_index * dqy_grid

    epsilon_x = QX - QX_nominal
    epsilon_y = QY - QY_nominal
    epsilon_x = np.where(geometry_valid, epsilon_x, 0.0)
    epsilon_y = np.where(geometry_valid, epsilon_y, 0.0)

    max_shift_x = np.nanmax(np.abs(epsilon_x)) / max(sigma_x, eps)
    max_shift_y = np.nanmax(np.abs(epsilon_y)) / max(sigma_y, eps)
    if max(max_shift_x, max_shift_y) > _MAX_FIRST_ORDER_SHIFT:
        warnings.warn(
            f"filter2_ungridded: maximum coordinate displacement exceeds "
            f"{_MAX_FIRST_ORDER_SHIFT:.2f} of the Gaussian sigma. First-order accuracy "
            f"may be insufficient. Normalized shifts: X={max_shift_x:.3f}, Y={max_shift_y:.3f}.",
            stacklevel=2,
        )

    # -------- NaN-safe numerator/denominator sources (MG_extract.m:1979-2007) --
    numerator_source = np.where(valid_mask, I, 0.0)
    denominator_source = np.where(valid_mask, solid_angle_ratio if apply_solid_angle else 1.0, 0.0)

    # -------- first-order derivative-expanded numerator (MG_extract.m:2009-2044) --
    numerator_0 = _separable_filter2(gy, gx, numerator_source)
    numerator_x_source = _separable_filter2(gy, dgx, numerator_source * epsilon_x)
    numerator_x_destination = epsilon_x * _separable_filter2(gy, dgx, numerator_source)
    numerator_y_source = _separable_filter2(dgy, gx, numerator_source * epsilon_y)
    numerator_y_destination = epsilon_y * _separable_filter2(dgy, gx, numerator_source)
    numerator = numerator_0 + numerator_x_source - numerator_x_destination + numerator_y_source - numerator_y_destination

    # -------- first-order derivative-expanded denominator (MG_extract.m:2046-2074) --
    denominator_0 = _separable_filter2(gy, gx, denominator_source)
    denominator_x_source = _separable_filter2(gy, dgx, denominator_source * epsilon_x)
    denominator_x_destination = epsilon_x * _separable_filter2(gy, dgx, denominator_source)
    denominator_y_source = _separable_filter2(dgy, gx, denominator_source * epsilon_y)
    denominator_y_destination = epsilon_y * _separable_filter2(dgy, gx, denominator_source)
    denominator = (
        denominator_0 + denominator_x_source - denominator_x_destination + denominator_y_source - denominator_y_destination
    )

    # -------- boundary convention (MG_extract.m:2076-2093): NORMALIZE_EDGES is False --
    inside_kernel_weight = _separable_filter2(gy, gx, np.ones((rows, cols)))
    denominator = denominator + 1.0 - inside_kernel_weight

    # -------- final normalized result (MG_extract.m:2095-2106) -----------------
    with np.errstate(invalid="ignore", divide="ignore"):
        filtered = numerator / denominator
    invalid_output = ~np.isfinite(denominator) | (denominator <= _DENOMINATOR_FLOOR)
    filtered = np.where(invalid_output, np.nan, filtered)
    return filtered

"""Reproduce paper Fig. 2: theory-vs-simulation disparity for Y_G, Y_G2, Y_M
across canonical form factors and V.

Paper: "Variable-Resolution Scattering Reveals Ensemble Properties"
(Steinitz & Beck), Fig. 2 caption: "Observables obtained from 2D simulation
results (dashed lines) and the theoretical first-order ratio from Tab. 2
(solid lines), plotted for a range of variances (V) and different
form-factors ... normally distributed ensemble had an average R0=5nm in a
typical Diamond B21 beamline setup, without noise."

IMPORTANT (fixed per an internal design review): V is the
SCATTERING-WEIGHTED relative variance throughout this package (and the
paper) -- but ``distributions.discretize_ensemble`` only directly controls
the NUMBER-weighted relative variance of its input. An earlier version of
this script fed the swept ``V_GRID`` value directly as that NUMBER input,
which only coincides with the scattering-weighted V for `weight_power=0`
shapes (gaussian_chain here). For `weight_power>0` shapes (solid_sphere,
spherical_shell, thin_rod, thin_disk), that made the x-axis silently wrong
-- and for high weight_power in particular, the discrepancy is severe: the
achievable SCATTERING-weighted V from a normal number-distribution
saturates well below 1 as the number-input variance grows (confirmed
directly: solid_sphere, weight_power=6, N=11 saturates around V~=0.06 no
matter how broad the number-input distribution gets), which is exactly
why the paper's own text notes "For the normal distribution used here,
the weighted variance is inherently limited, therefore its span is not
always fully covered." This script now uses
``distributions.target_effective_distribution`` to properly target a
SCATTERING-weighted V for every shape, and simply omits (does not plot) a
requested V for shapes/weight-powers where it is provably unreachable
(``VarianceTargetUnreachable``) rather than silently mislabeling the axis.

Usage:
    .venv/bin/python3 scripts/reproduce_fig2.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenor_saxs import distributions, formfactors, protocol, psf, simulation  # noqa: E402

# Diamond B21-like instrument setup (init_TENOR_params.m defaults).
SD_DIST = 360.0
WAVELENGTH = 0.1
DET_SIDE = 3.5
DET_PIX = 500
PSF0 = psf.bartlett2d(3, 15)
PXN = np.array([87, 85, 125, 123])
R0 = 5.0
N_RADII = 11
DISTRIBUTION = "normal"

SHAPES = [
    ("spherical_shell", formfactors.GUINIER_TABLE["spherical_shell"].phi2, formfactors.GUINIER_TABLE["spherical_shell"].weight_power),
    ("solid_sphere", formfactors.GUINIER_TABLE["solid_sphere"].phi2, formfactors.GUINIER_TABLE["solid_sphere"].weight_power),
    ("thin_rod", formfactors.GUINIER_TABLE["thin_rod"].phi2, formfactors.GUINIER_TABLE["thin_rod"].weight_power),
    ("gaussian_chain", formfactors.GUINIER_TABLE["gaussian_chain"].phi2, formfactors.GUINIER_TABLE["gaussian_chain"].weight_power),
    # thin_disk: MATLAB-only shape (Scatter2D.m's Nu==0.000666 sentinel branch,
    # not in the paper's own Table 1) -- per the internal validation notes item 3, this was
    # never actually checked against the theory/simulation comparison before;
    # included here now. Its "phi2" is a numeric sentinel value (not a
    # physically meaningful Guinier curvature), so it deliberately does NOT
    # also get a matching entry on the `v_theory` calibration curve overlay
    # in the same sense as the other four -- see the note where it's plotted.
    ("thin_disk", 0.000666, formfactors.weight_power_for_name("thin_disk")),
    ("intermediate (phi''=0.01)", 0.01, 0),
]

V_GRID = np.linspace(0.0, 0.30, 13)

# Y_M's raw simulated values can swing wildly for isotropic form factors
# (the fitted m1 coefficient it divides by sits at the numerical noise
# floor -- see the package's internal validation notes's Ym210 outlier writeup), which would
# otherwise blow out the y-axis and hide the well-behaved trend for the
# shapes that DO track theory. Clipped to a fixed, paper-comparable range
# (an internal design review: "make the Y_M y-axis limits similar
# to the original Fig 2") rather than auto-scaling to the outliers.
YM_YLIM = (-1.0, 1.0)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    v_theory = np.linspace(0.0, 0.30, 200)

    for name, phi2, weight_power in SHAPES:
        v_achieved, sim_yg100, sim_yg210, sim_ym210 = [], [], [], []
        n_skipped = 0
        for v_target in V_GRID:
            try:
                target = distributions.target_effective_distribution(
                    target_v=float(v_target), target_observed_rg=R0, dist_name=DISTRIBUTION,
                    n=N_RADII, weight_power=weight_power,
                )
            except distributions.VarianceTargetUnreachable:
                n_skipped += 1
                continue
            result = simulation.scatter2d(
                rg=target["requested_rg"], noise=0, v_rel=target["input_variance"], phi2=phi2,
                det_pix=DET_PIX, sd_dist=SD_DIST, wavelength=WAVELENGTH, det_side=DET_SIDE,
                psf0=PSF0, dist_type=DISTRIBUTION, n_radii=N_RADII, weight_power=weight_power,
            )
            r = protocol.tenor_protocol(
                result.intensity, result.qx, result.qy, phi2, pxn=PXN,
                wavelength=WAVELENGTH,
                observables=("Yg100", "Yg210", "Ym210"),
            )
            v_achieved.append(target["realized_v"])
            sim_yg100.append(r.observed["Yg100"])
            sim_yg210.append(r.observed["Yg210"])
            sim_ym210.append(r.observed["Ym210"])

        # For thin_disk, phi2=0.000666 is a dispatch SENTINEL, not a real
        # curvature -- Scatter2D.m's own comment gives the disk's true
        # Guinier curvature as phi''=0 exactly (leading curvature is in the
        # neglected CUBIC term, phi'''=1/270). Using the sentinel as an
        # approximate small-phi'' theory line is intentional (it's numerically
        # close to phi''=0 either way); any theory/simulation disparity seen
        # for this shape is expected to be dominated by that missing cubic
        # term, exactly the same story already documented for the other
        # exact form factors' Y_G2/Y_M disparity.
        theory = protocol.analytical_theory(v_theory, phi2)
        (line,) = axes[0].plot(v_theory, theory["Yg100"], "-", label=name)
        color = line.get_color()
        axes[0].plot(v_achieved, sim_yg100, "--", color=color, marker="o", ms=3)
        axes[1].plot(v_theory, theory["Yg210"], "-", color=color)
        axes[1].plot(v_achieved, sim_yg210, "--", color=color, marker="o", ms=3)
        axes[2].plot(v_theory, theory["Ym210"], "-", color=color)
        axes[2].plot(v_achieved, sim_ym210, "--", color=color, marker="o", ms=3)
        v_max = max(v_achieved) if v_achieved else 0.0
        print(f"{name}: {len(v_achieved)}/{len(V_GRID)} V values reached (max achieved V={v_max:.4f}, {n_skipped} unreachable/skipped)")

    for ax, title in zip(axes, ["Y_G", "Y_G2", "Y_M"]):
        ax.set_xlabel(r"scattering-weighted $V$")
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[2].set_ylim(*YM_YLIM)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Reproduction of paper Fig. 2: theory (solid) vs. simulation (dashed), R0=5nm, normal, no noise\n"
        "(x-axis: scattering-weighted V, via target_effective_distribution -- shapes with high weight_power reach a lower max V, matching the paper's own caveat)"
    )
    fig.tight_layout()
    out_path = out_dir / "fig2_reproduction.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)

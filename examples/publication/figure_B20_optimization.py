"""Publication figure for exact B2c elimination and full-axis B20 optimization."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_surface_3d

STOCK_CONFIGURATION = "qa"
OPTIMIZED_CONFIGURATION = "b20_optimized_qa"
NPHI = 121
SURFACE_RADIUS = 0.075
OUTPUT_STEM = Path("examples/output/publication/B20_optimization")
README_PNG = Path("docs/_static/B20_optimization.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

plt.style.use("seaborn-v0_8-whitegrid")
print("Comparing exact B2c elimination with the optimized Fourier axis...")
stock = qsc.optimize_B2c(qsc.solve_configuration(STOCK_CONFIGURATION, nphi=NPHI))
optimized = qsc.solve_configuration(OPTIMIZED_CONFIGURATION, nphi=NPHI)
optimized_diagnostics = qsc.b20_diagnostics(optimized)
verification = qsc.verify_B20_resolution(optimized, multipliers=(1, 2))
improvement = float(stock.diagnostics.weighted_l2 / optimized_diagnostics.weighted_l2)
stock_angle = np.asarray(stock.solution.varphi * stock.solution.inputs.axis.nfp / (2 * np.pi))
optimized_angle = np.asarray(optimized.varphi * optimized.inputs.axis.nfp / (2 * np.pi))

figure = plt.figure(figsize=(13.2, 8.2))
surface_axis = figure.add_subplot(2, 2, 1, projection="3d")
plot_surface_3d(
    optimized,
    radius=SURFACE_RADIUS,
    ntheta=36,
    ax=surface_axis,
    cmap="viridis",
)
surface_axis.view_init(elev=24, azim=38)
surface_axis.set_title(
    "optimized QA surface\n"
    rf"$\iota={float(optimized.iota):.3f}$, "
    rf"$r_\mathrm{{sing}}={float(optimized.r_singularity):.3f}$ m"
)

stock_axis = figure.add_subplot(2, 2, 2)
stock_axis.plot(stock_angle, np.asarray(stock.diagnostics.anomaly), linewidth=2.2)
stock_axis.set_title(r"stock axis + exact $B_{2c}$")
stock_axis.set_xlabel("Boozer angle / field period")
stock_axis.set_ylabel(r"$B_{20}-\langle B_{20}\rangle$ [T/m$^2$]")

optimized_axis = figure.add_subplot(2, 2, 3)
optimized_axis.plot(
    optimized_angle,
    np.asarray(optimized_diagnostics.anomaly),
    linewidth=2.2,
    color="tab:green",
)
optimized_axis.set_title("optimized Fourier axis")
optimized_axis.set_xlabel("Boozer angle / field period")
optimized_axis.set_ylabel(r"$B_{20}-\langle B_{20}\rangle$ [T/m$^2$]")
diagnostic_names = ("weighted $L^2$", "dense maximum", "peak-to-peak")
stock_values = (
    float(stock.diagnostics.weighted_l2),
    float(stock.diagnostics.grid_maximum),
    float(stock.diagnostics.peak_to_peak),
)
optimized_values = (
    float(optimized_diagnostics.weighted_l2),
    float(optimized_diagnostics.grid_maximum),
    float(optimized_diagnostics.peak_to_peak),
)
summary_axis = figure.add_subplot(2, 2, 4)
summary_axis.bar(
    np.arange(3) - 0.18,
    stock_values,
    width=0.36,
    label=r"exact $B_{2c}$ only",
)
summary_axis.bar(
    np.arange(3) + 0.18,
    optimized_values,
    width=0.36,
    label="axis + $B_{2c}$",
    color="tab:green",
)
summary_axis.set_xticks(np.arange(3), diagnostic_names, rotation=12)
summary_axis.set_yscale("log")
summary_axis.set_ylabel(r"$B_{20}$ nonuniformity [T/m$^2$]")
summary_axis.legend(fontsize=8)
clearance = float(optimized.r_singularity) / SURFACE_RADIUS
summary_axis.text(
    0.04,
    0.05,
    f"{improvement:,.0f}x lower residual\n"
    rf"shown surface: $r={SURFACE_RADIUS:.3f}$ m"
    f" ({clearance:.1f}x inside singular radius)",
    transform=summary_axis.transAxes,
    fontweight="bold",
    bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
)
figure.tight_layout()

try:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    commit = "unavailable"
metadata = {
    "git_commit": commit,
    "stock_configuration": STOCK_CONFIGURATION,
    "optimized_configuration": OPTIMIZED_CONFIGURATION,
    "nphi": NPHI,
    "stock_exact_B2c": float(stock.B2c_optimal),
    "optimized_B2c": float(optimized.inputs.B2c),
    "stock_weighted_l2": float(stock.diagnostics.weighted_l2),
    "optimized_weighted_l2": float(optimized_diagnostics.weighted_l2),
    "optimized_grid_maximum": float(optimized_diagnostics.grid_maximum),
    "optimized_peak_to_peak": float(optimized_diagnostics.peak_to_peak),
    "optimized_singular_radius": float(optimized.r_singularity),
    "surface_radius": SURFACE_RADIUS,
    "singular_radius_clearance": clearance,
    "improvement_factor": improvement,
    "verification_nphi": np.asarray(verification.resolutions).tolist(),
    "verification_weighted_l2": np.asarray(verification.weighted_l2).tolist(),
    "optimizer": "bounded scipy least_squares after exact B2c elimination",
}
if SAVE_OUTPUT:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    README_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(README_PNG, dpi=120, bbox_inches="tight")
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("saved:", OUTPUT_STEM)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

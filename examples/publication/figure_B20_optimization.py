"""Publication figure for exact B2c elimination and full-axis B20 optimization."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

STOCK_CONFIGURATION = "qa"
OPTIMIZED_CONFIGURATION = "b20_optimized_qa"
NPHI = 121
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

figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
axes[0].plot(stock_angle, np.asarray(stock.diagnostics.anomaly), linewidth=2.2)
axes[0].set_title(r"stock axis + exact $B_{2c}$")
axes[0].set_xlabel("Boozer angle / field period")
axes[0].set_ylabel(r"$B_{20}-\langle B_{20}\rangle$ [T/m$^2$]")
axes[1].plot(
    optimized_angle,
    np.asarray(optimized_diagnostics.anomaly),
    linewidth=2.2,
    color="tab:green",
)
axes[1].set_title("optimized Fourier axis")
axes[1].set_xlabel("Boozer angle / field period")
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
axes[2].bar(
    np.arange(3) - 0.18,
    stock_values,
    width=0.36,
    label=r"exact $B_{2c}$ only",
)
axes[2].bar(
    np.arange(3) + 0.18,
    optimized_values,
    width=0.36,
    label="axis + $B_{2c}$",
    color="tab:green",
)
axes[2].set_xticks(np.arange(3), diagnostic_names, rotation=15)
axes[2].set_yscale("log")
axes[2].set_ylabel(r"$B_{20}$ nonuniformity [T/m$^2$]")
axes[2].legend(fontsize=8)
axes[2].text(
    0.04,
    0.05,
    f"{improvement:,.0f}x lower weighted residual",
    transform=axes[2].transAxes,
    fontweight="bold",
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

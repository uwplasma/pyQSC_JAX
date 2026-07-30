"""Compare a stock QA axis with a dense-grid B20-optimized QA axis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

STOCK_CONFIGURATION = "qa"
OPTIMIZED_CONFIGURATION = "b20_optimized_qa"
NPHI = 121
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/06_optimize_axis_B20.png")

print("Eliminating B2c exactly for the stock QA axis...")
stock = qsc.optimize_B2c(qsc.solve_configuration(STOCK_CONFIGURATION, nphi=NPHI))
print("Loading the independently optimized Fourier axis...")
optimized = qsc.solve_configuration(OPTIMIZED_CONFIGURATION, nphi=NPHI)
optimized_diagnostics = qsc.b20_diagnostics(optimized)
verification = qsc.verify_B20_resolution(optimized, multipliers=(1, 2))
improvement = float(stock.diagnostics.weighted_l2 / optimized_diagnostics.weighted_l2)

print("stock exact-B2c weighted L2:", float(stock.diagnostics.weighted_l2))
print("optimized-axis weighted L2:", float(optimized_diagnostics.weighted_l2))
print("optimized-axis dense maximum:", float(optimized_diagnostics.grid_maximum))
print("improvement factor:", improvement)
print("nphi verification:", np.asarray(verification.resolutions))
print("verified weighted L2:", np.asarray(verification.weighted_l2))

stock_angle = np.asarray(stock.solution.varphi * stock.solution.inputs.axis.nfp / (2 * np.pi))
optimized_angle = np.asarray(optimized.varphi * optimized.inputs.axis.nfp / (2 * np.pi))
figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
axes[0].plot(stock_angle, np.asarray(stock.diagnostics.anomaly), linewidth=2)
axes[0].set_title(r"stock axis + exact $B_{2c}$")
axes[0].set_xlabel("Boozer angle / field period")
axes[0].set_ylabel(r"$B_{20}-\langle B_{20}\rangle$ [T/m$^2$]")
axes[1].plot(
    optimized_angle,
    np.asarray(optimized_diagnostics.anomaly),
    linewidth=2,
    color="tab:green",
)
axes[1].set_title(f"optimized axis ({improvement:,.0f}x smaller)")
axes[1].set_xlabel("Boozer angle / field period")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

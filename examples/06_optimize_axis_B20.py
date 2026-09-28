"""Refine a screened database axis into a nearly constant-B20 stellarator."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

NPHI = 121
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/06_optimize_axis_B20.png")

print("Eliminating B2c exactly for database configuration 57409...")
# Stellarator database configuration 57409 (Curvo et al. 2025,
# https://stellarator.physics.wisc.edu/app/plot/57409).
database = qsc.Qsc(
    rc=[1.0, -0.51677144, -0.009499784, -0.005914526],
    zs=[0.0, -0.5420635, -0.012225689, -0.0059485724],
    nfp=4,
    etabar=-1.3295174,
    B2c=-0.7577404,
    p2=-23501.281,
    order="r3",
    nphi=NPHI,
)
stock = qsc.optimize_B2c(database)
print("Loading the independently optimized Fourier axis...")
# Eight-mode axis refined from 57409 by exact B2c elimination and bounded least squares
# on the B20 anomaly (the optimizer itself is preserved on the
# preserve/pr2-before-refactor-2026-09-27 branch; the result is reproduced here).
optimized = qsc.Qsc(
    rc=[
        1.0,
        -0.5039436500066075,
        -0.043965867334349464,
        -0.00654919263283669,
        -3.047898633100702e-06,
        2.7519909478191202e-05,
        6.071324626161564e-06,
        8.46692978641554e-07,
        5.99743474381913e-08,
    ],
    zs=[
        0.0,
        -0.5050020451312105,
        -0.045010140721391825,
        -0.006585874304053245,
        -1.5550078876295003e-05,
        2.6653739413147386e-05,
        5.978859527401134e-06,
        8.327402553082346e-07,
        5.9109366390399247e-08,
    ],
    nfp=4,
    etabar=-1.3295174,
    B2c=-1.132420959333329,
    p2=-23501.281,
    order="r3",
    nphi=NPHI,
)
optimized_diagnostics = qsc.b20_diagnostics(optimized)
improvement = float(stock.diagnostics.weighted_l2 / optimized_diagnostics.weighted_l2)

print("stock exact-B2c weighted L2:", float(stock.diagnostics.weighted_l2))
print("optimized-axis weighted L2:", float(optimized_diagnostics.weighted_l2))
print("optimized-axis dense maximum:", float(optimized_diagnostics.grid_maximum))
print("improvement factor:", improvement)
print("|iota|:", abs(float(optimized.iota)))
print("singular radius:", float(optimized.r_singularity))

stock_angle = np.asarray(stock.solution.varphi * stock.solution.inputs.axis.nfp / (2 * np.pi))
optimized_angle = np.asarray(optimized.varphi * optimized.inputs.axis.nfp / (2 * np.pi))
figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
axes[0].plot(stock_angle, np.asarray(stock.diagnostics.anomaly), linewidth=2)
axes[0].set_title(r"database ID 57409 + exact $B_{2c}$")
axes[0].set_xlabel("Boozer angle / field period")
axes[0].set_ylabel(r"$B_{20}-\langle B_{20}\rangle$ [T/m$^2$]")
axes[1].plot(
    optimized_angle, np.asarray(optimized_diagnostics.anomaly), linewidth=2, color="tab:green"
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

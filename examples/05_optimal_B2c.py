"""Eliminate the affine B2c subproblem exactly."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

RC = [1.0, 0.155, 0.0102]
ZS = [0.0, 0.154, 0.0111]
NFP = 2
ETABAR = 0.64
INITIAL_B2C = 0.0
NPHI = 61
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/05_optimal_B2c.png")

print("Solving the initial r2 configuration...")
initial = qsc.Qsc(
    rc=RC,
    zs=ZS,
    nfp=NFP,
    etabar=ETABAR,
    B2c=INITIAL_B2C,
    nphi=NPHI,
    order="r2",
)
result = qsc.optimize_B2c(initial)
optimized = result.solution
print("initial B2c:", INITIAL_B2C)
print("optimal B2c:", float(result.B2c_optimal))
print("initial weighted L2:", float(qsc.b20_diagnostics(initial).weighted_l2))
print("optimal weighted L2:", float(result.diagnostics.weighted_l2))
print("affine reconstruction error:", float(result.affine_reconstruction_error))

angle = np.asarray(initial.varphi * NFP / (2 * np.pi))
figure, axis = plt.subplots(figsize=(6.2, 3.8))
axis.plot(angle, np.asarray(initial.B20_anomaly), label="initial", linewidth=2)
axis.plot(angle, np.asarray(optimized.B20_anomaly), label="optimal B2c", linewidth=2)
axis.set_xlabel("Boozer angle / field period")
axis.set_ylabel(r"$B_{20}-\langle B_{20}\rangle$ [T/m$^2$]")
axis.legend()
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

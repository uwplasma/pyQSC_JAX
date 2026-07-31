"""Prescribe rotational transform and solve for etabar."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
TARGET_IOTA = 0.42
ETABAR_SEED = -1.0
NPHI = 31
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/04_target_iota.png")

print("Solving inverse target-iota problem...")
inverse = qsc.solve(
    axis=AXIS,
    etabar=ETABAR_SEED,
    iota=TARGET_IOTA,
    solve_for="etabar",
    nphi=NPHI,
)
forward = qsc.solve(
    axis=AXIS,
    etabar=inverse.inputs.etabar,
    nphi=NPHI,
)
print("target iota:", TARGET_IOTA)
print("solved etabar:", float(inverse.inputs.etabar))
print("forward-check iota:", float(forward.iota))
print("response d(iota)/d(etabar):", float(inverse.response_derivative))
print("branch fold:", bool(inverse.branch_fold))

figure, axis = plt.subplots(figsize=(6.0, 3.6))
axis.plot(
    np.asarray(inverse.varphi),
    np.asarray(inverse.sigma),
    linewidth=2,
)
axis.set_xlabel("Boozer toroidal angle [rad]")
axis.set_ylabel(r"$\sigma$")
axis.set_title("Target-iota periodic solution")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

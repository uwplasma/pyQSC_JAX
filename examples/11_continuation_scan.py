"""Trace an etabar branch with pseudo-arclength continuation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
ETABAR_START = -1.0
ETABAR_NEXT = -0.98
NUMBER_OF_POINTS = 12
NPHI = 31
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/11_continuation_scan.png")

print("Tracing the first-order etabar branch...")
result = qsc.continue_etabar_branch(
    axis=AXIS,
    etabar_start=ETABAR_START,
    etabar_next=ETABAR_NEXT,
    num_points=NUMBER_OF_POINTS,
    nphi=NPHI,
)
etabar = np.asarray(result.etabar)
iota = np.asarray(result.iota)
fold = np.asarray(result.fold_detected)
print("status:", result.status)
print("points:", len(result.solutions))
print("etabar interval:", float(etabar.min()), float(etabar.max()))
print("iota interval:", float(iota.min()), float(iota.max()))
print("detected fold samples:", np.flatnonzero(fold))

figure, axis = plt.subplots(figsize=(6.0, 3.8))
axis.plot(etabar, iota, "-o", label="corrected branch")
if fold.any():
    axis.scatter(etabar[fold], iota[fold], marker="x", s=80, label="fold diagnostic")
axis.set_xlabel(r"$\bar{\eta}$ [m$^{-1}$]")
axis.set_ylabel(r"$\iota$")
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

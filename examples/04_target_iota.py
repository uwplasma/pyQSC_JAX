"""Prescribe rotational transform and solve for etabar."""

from pathlib import Path

import jax
import jax.numpy as jnp
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


def iota_of(etabar):
    return qsc.solve(axis=AXIS, etabar=etabar, nphi=NPHI).iota


# A few Newton steps on iota(etabar) - target, with d(iota)/d(etabar) from the
# implicit derivative of the sigma equation (no inverse-problem framework needed).
print("Solving iota(etabar) = target with Newton...")
residual_and_slope = jax.jit(jax.value_and_grad(lambda e: iota_of(e) - TARGET_IOTA))
etabar = jnp.asarray(ETABAR_SEED)
for _ in range(20):
    residual, slope = residual_and_slope(etabar)
    if abs(float(residual)) < 1e-12:
        break
    etabar = etabar - residual / slope
inverse = qsc.solve(axis=AXIS, etabar=etabar, nphi=NPHI)
print("target iota:", TARGET_IOTA)
print("solved etabar:", float(etabar))
print("achieved iota:", float(inverse.iota))
print("response d(iota)/d(etabar):", float(slope))

figure, axis = plt.subplots(figsize=(6.0, 3.6))
axis.plot(np.asarray(inverse.varphi), np.asarray(inverse.sigma), linewidth=2)
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

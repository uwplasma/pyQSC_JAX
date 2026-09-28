"""Compare an implicit-solve JVP with a centered finite difference."""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
ETABAR = -0.9
NPHI = 31
FINITE_DIFFERENCE_STEP = 1.0e-5
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/12_autodiff_check.png")


def iota_from_etabar(etabar):
    """Return one converged implicit-solve output."""

    return qsc.solve(axis=AXIS, etabar=etabar, nphi=NPHI).iota


print("Differentiating rotational transform through the converged sigma solve...")
value, tangent = jax.jvp(iota_from_etabar, (ETABAR,), (1.0,))
upper = iota_from_etabar(ETABAR + FINITE_DIFFERENCE_STEP)
lower = iota_from_etabar(ETABAR - FINITE_DIFFERENCE_STEP)
finite_difference = (upper - lower) / (2 * FINITE_DIFFERENCE_STEP)
relative_error = abs(tangent - finite_difference) / max(abs(finite_difference), 1.0e-14)
print("iota:", float(value))
print("JVP d(iota)/d(etabar):", float(tangent))
print("finite-difference derivative:", float(finite_difference))
print("relative error:", float(relative_error))

figure, axis = plt.subplots(figsize=(5.5, 3.6))
axis.bar(
    ("JAX implicit JVP", "centered difference"),
    np.asarray((tangent, finite_difference)),
    color=("tab:blue", "tab:orange"),
)
axis.set_ylabel(r"$d\iota/d\bar{\eta}$ [m]")
axis.set_title(f"relative error = {float(relative_error):.2e}")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

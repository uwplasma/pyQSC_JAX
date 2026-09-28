"""Colour a near-axis surface by |B| and by the normal error of an external field."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_surface_3d, surface_normal_field_error

# Landreman & Sengupta (2019), section 5.1 (pyQSC "r2 section 5.1").
RADIUS = 0.08
NTHETA = 32
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/15_surface_coloring.png")

solution = qsc.Qsc(
    rc=[1.0, 0.155, 0.0102],
    zs=[0.0, 0.154, 0.0111],
    nfp=2,
    etabar=0.64,
    B2c=-0.00322,
    nphi=61,
    order="r2",
)


def external_field(points, B0=1.0, R0=1.0, error=1.0e-3):
    """A 1/R toroidal field plus a uniform vertical error field [T] at Cartesian points [m]."""

    x, y = points[:, 0], points[:, 1]
    R2 = x**2 + y**2
    return np.stack((-B0 * R0 * y / R2, B0 * R0 * x / R2, error + 0 * x), axis=-1)


error = surface_normal_field_error(solution, external_field, radius=RADIUS, ntheta=NTHETA)
print("max |B.n|/|B|:", float(np.max(np.abs(error))))

figure = plt.figure(figsize=(10.0, 4.2))
axis_B = figure.add_subplot(1, 2, 1, projection="3d")
axis_Bn = figure.add_subplot(1, 2, 2, projection="3d")
plot_surface_3d(solution, radius=RADIUS, ntheta=NTHETA, ax=axis_B)  # default: colour by |B|
plot_surface_3d(
    solution, radius=RADIUS, ntheta=NTHETA, ax=axis_Bn, color_by="Bn", field=external_field
)
axis_B.set_title("near-axis |B|")
axis_Bn.set_title(r"$\log_{10}(|B\cdot n|/|B|)$, toroidal + 1 mT vertical")
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

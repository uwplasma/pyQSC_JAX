"""Construct and plot a first-order quasi-axisymmetric configuration."""

from pathlib import Path

import matplotlib.pyplot as plt

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_axis

RC = [1.0, 0.045]
ZS = [0.0, -0.045]
NFP = 3
ETABAR = -0.9
NPHI = 31
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/01_first_order_qa.png")

print("Solving first-order QA configuration...")
solution = qsc.Qsc(
    rc=RC,
    zs=ZS,
    nfp=NFP,
    etabar=ETABAR,
    nphi=NPHI,
    order="r1",
)
print("iota:", float(solution.iota))
print("sigma residual:", float(solution.root_report.residual_norm))
print("axis length [m]:", float(solution.axis_length))
print("maximum elongation:", float(solution.elongation.max()))

figure, _ = plot_axis(solution, label="first-order QA")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

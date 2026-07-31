"""Construct and plot a first-order quasi-helically symmetric configuration."""

from pathlib import Path

import matplotlib.pyplot as plt

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_axis

RC = [1.0, 0.265]
ZS = [0.0, -0.21]
NFP = 4
ETABAR = -0.9
NPHI = 31
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/02_first_order_qh.png")

print("Solving first-order QH configuration...")
solution = qsc.Qsc(
    rc=RC,
    zs=ZS,
    nfp=NFP,
    etabar=ETABAR,
    nphi=NPHI,
    order="r1",
)
print("iota:", float(solution.iota))
print("iota_N:", float(solution.iotaN))
print("frame helicity:", int(solution.helicity))
print("sigma residual:", float(solution.root_report.residual_norm))

figure, _ = plot_axis(solution, label="first-order QH", color="tab:orange")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

"""Solve the complete finite-pressure/current second-order system."""

from pathlib import Path

import matplotlib.pyplot as plt

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_b20

RC = [1.0, 0.09]
ZS = [0.0, -0.09]
NFP = 2
ETABAR = 0.95
I2 = 0.9
P2 = -600000.0
B2C = -0.7
NPHI = 61
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/03_second_order_finite_beta.png")

print("Solving the finite-pressure/current r2 configuration...")
solution = qsc.Qsc(
    rc=RC,
    zs=ZS,
    nfp=NFP,
    etabar=ETABAR,
    I2=I2,
    p2=P2,
    B2c=B2C,
    nphi=NPHI,
    order="r2",
)
print("iota:", float(solution.iota))
print("linear residual:", float(solution.linear_report.residual_norm))
print("linear condition number:", float(solution.linear_report.matrix_condition_number))
print("B20 weighted residual:", float(solution.B20_residual))
print("Mercier D r^2:", float(solution.DMerc_times_r2))
print("singular radius [m]:", float(solution.r_singularity))

figure, _ = plot_b20(solution, label="finite beta/current", color="tab:red")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

"""Solve the complete finite-pressure, zero-current second-order system."""

from pathlib import Path

import matplotlib.pyplot as plt

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_b20

CONFIGURATION = "plasma_stellarator"
NPHI = 61
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/03_second_order_finite_beta.png")

print("Solving the finite-pressure, zero-current stellarator...")
solution = qsc.solve_configuration(
    CONFIGURATION,
    nphi=NPHI,
    order="r2",
)
assert float(solution.inputs.I2) == 0.0
assert float(solution.inputs.p2) != 0.0
print("iota:", float(solution.iota))
print("torsion RMS [1/m]:", float((solution.torsion**2).mean() ** 0.5))
print("linear residual:", float(solution.linear_report.residual_norm))
print("linear condition number:", float(solution.linear_report.matrix_condition_number))
print("B20 weighted residual:", float(solution.B20_residual))
print("Mercier D r^2:", float(solution.DMerc_times_r2))
print("singular radius [m]:", float(solution.r_singularity))

figure, _ = plot_b20(solution, label="finite pressure, zero current", color="tab:red")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

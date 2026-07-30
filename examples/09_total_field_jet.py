"""Inspect the total on-axis field, gradient, and Hessian."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

CONFIGURATION = "qa"
NPHI = 61
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/09_total_field_jet.png")

print("Solving the total-field-jet reference configuration...")
solution = qsc.solve_configuration(CONFIGURATION, nphi=NPHI)
jet = solution.field_jet
if jet is None:
    raise RuntimeError("The r2 solution did not produce a total field jet.")
print("field reconstruction error:", float(jet.maximum_field_error))
print("gradient reconstruction error:", float(jet.maximum_gradient_error))
print("maximum divergence:", float(jet.maximum_divergence))
print("maximum Hessian derivative asymmetry:", float(jet.maximum_derivative_asymmetry))
print("maximum gradient of divergence:", float(jet.maximum_divergence_gradient))
print("minimum Hessian scale length [m]:", float(jet.L_grad_grad_B.min()))

angle = np.asarray(solution.varphi * solution.inputs.axis.nfp / (2 * np.pi))
hessian = np.asarray(jet.hessian)
figure, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
axes[0].plot(angle, np.asarray(solution.B_axis[:, 0]), label=r"$B_x$")
axes[0].plot(angle, np.asarray(solution.B_axis[:, 1]), label=r"$B_y$")
axes[0].plot(angle, np.asarray(solution.B_axis[:, 2]), label=r"$B_z$")
axes[0].set_ylabel("field [T]")
axes[0].legend(ncol=3)
axes[1].plot(angle, hessian[:, 0, 0, 0], label=r"$\partial_{xx}B_x$")
axes[1].plot(angle, hessian[:, 0, 1, 1], label=r"$\partial_{yy}B_x$")
axes[1].plot(angle, hessian[:, 0, 2, 2], label=r"$\partial_{zz}B_x$")
axes[1].set_xlabel("Boozer angle / field period")
axes[1].set_ylabel(r"Hessian component [T/m$^2$]")
axes[1].legend(ncol=3)
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

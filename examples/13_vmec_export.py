"""Export a diagnosed VMEC boundary without invoking VMEC."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

RADIUS = 0.005
NPHI = 61
NTHETA = 40
MPOL = 12
NTOR = 14
SAVE_OUTPUT = True
SHOW_FIGURE = False
VMEC_INPUT = Path("examples/output/input.pyqsc_jax_qa")
FIGURE_OUTPUT = Path("examples/output/13_vmec_export.png")

solution = qsc.Qsc(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
    nphi=NPHI,
    order="r2",
)
print("Writing the fixed-boundary VMEC input...")
export = qsc.to_vmec(
    solution,
    VMEC_INPUT,
    r=RADIUS,
    ntheta=NTHETA,
    mpol=MPOL,
    ntor=NTOR,
)
boundary = export.boundary
print("input:", export.path)
print("conversion seconds (compile + first execution):", export.conversion_seconds)
print("toroidal-angle inversion converged:", bool(boundary.toroidal_angle_converged))
print("maximum toroidal-angle residual:", float(boundary.maximum_toroidal_angle_residual))
print("toroidal-angle tolerance:", float(boundary.toroidal_angle_tolerance))
print("maximum R reconstruction error [m]:", float(boundary.maximum_R_reconstruction_error))
print("maximum Z reconstruction error [m]:", float(boundary.maximum_Z_reconstruction_error))
print("run with: xvmec", export.path.name)

figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
phi_indices = np.linspace(0, NPHI - 1, 7, dtype=int)
for index in phi_indices:
    axes[0].plot(
        np.asarray(boundary.R[:, index]),
        np.asarray(boundary.Z[:, index]),
        linewidth=1.3,
    )
axes[0].set_aspect("equal")
axes[0].set_xlabel("R [m]")
axes[0].set_ylabel("Z [m]")
axes[0].set_title("uniform cylindrical-toroidal sections")
mode_amplitude = np.sqrt(
    np.asarray(boundary.RBC) ** 2
    + np.asarray(boundary.RBS) ** 2
    + np.asarray(boundary.ZBC) ** 2
    + np.asarray(boundary.ZBS) ** 2
)
axes[1].semilogy(np.sort(mode_amplitude.ravel())[::-1], marker=".", linewidth=1)
axes[1].set_xlabel("coefficient rank")
axes[1].set_ylabel("combined Fourier amplitude [m]")
axes[1].set_title("VMEC boundary spectrum")
figure.tight_layout()
if SAVE_OUTPUT:
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", FIGURE_OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

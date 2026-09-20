"""Construct and plot a first-order quasi-axisymmetric configuration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qscX
from pyqsc_jax.plotting import plot_surface_3d, plot_b20, plot_axis, surface_coordinates
import requests

ID_CONFIG = 139379
# ID_CONFIG = 139391
url = f"https://stellarator.physics.wisc.edu/backend/api/download/{ID_CONFIG}?format=json"
config = requests.get(url).json()
nphi = 51
ntheta = 51
arrow_length = 0.2

# RC = [1.0, 0.045]
# ZS = [0.0, -0.045]
# NFP = 3
# ETABAR = -0.9
# NPHI = 31
# radius = 0.3
# B2c = 0.1

print("Solving configuration...")
solution = qscX.Qsc(
    rc=[1, config["rc1"], config["rc2"], config["rc3"]],
    zs=[0, config["zs1"], config["zs2"], config["zs3"]],
    nfp=config["nfp"],
    etabar=config["etabar"],
    nphi=nphi,
    I2=0.0,
    order="r3",
    B2c=config["B2c"],
    p2=config["p2"],
)
print("iota:", float(solution.iota))
print("sigma residual:", float(solution.root_report.residual_norm))
print("axis length [m]:", float(solution.axis_length))
print("maximum elongation:", float(solution.elongation.max()))
print("r_singularity:", float(solution.r_singularity))
beta = (
    -qscX.second_order.MU0 * solution.inputs.p2 * solution.r_singularity**2 / solution.inputs.B0**2
)
print("plasma beta:", float(beta))

radius = solution.r_singularity

# plot_surface_3d(solution, radius=radius, ntheta=ntheta)
# plot_b20(solution)

L = solution.axis_length
etabar = solution.inputs.etabar
curvature = solution.curvature
sigma = solution.sigma
iotaN = solution.iotaN
D = sigma**2 + (1 + etabar**2 / curvature**2) ** 2
t = solution.geometry.tangent_cartesian
n = solution.geometry.normal_cartesian
b = solution.geometry.binormal_cartesian

Bt = -beta * t
Bn = -beta * (2 * L * etabar**2 / (iotaN * curvature * D) * sigma)[:, None] * n
Bb = (
    -beta
    * (2 * L * etabar**2 / (iotaN * curvature * D) * (-(1 + etabar**2 / curvature**2)))[:, None]
    * b
)

Bplasma = Bt + Bn + Bb
Btotal = t
Bcoils = Btotal - Bplasma

phi = solution.phi
R = solution.R0
Z = solution.Z0
axis_xyz = np.stack(
    [
        np.asarray(R) * np.cos(np.asarray(phi)),
        np.asarray(R) * np.sin(np.asarray(phi)),
        np.asarray(Z),
    ],
    axis=-1,
)

fig, ax = plot_axis(solution, label="Axis", color="k")
ax.quiver(
    axis_xyz[:, 0],
    axis_xyz[:, 1],
    axis_xyz[:, 2],
    Bcoils[:, 0],
    Bcoils[:, 1],
    Bcoils[:, 2],
    length=arrow_length,
    normalize=False,
    color="r",
    label=r"$B_{coils}$",
)
ax.quiver(
    axis_xyz[:, 0],
    axis_xyz[:, 1],
    axis_xyz[:, 2],
    Bplasma[:, 0],
    Bplasma[:, 1],
    Bplasma[:, 2],
    length=arrow_length,
    normalize=False,
    color="b",
    label=r"$B_{plasma}$",
)
ax.quiver(
    axis_xyz[:, 0],
    axis_xyz[:, 1],
    axis_xyz[:, 2],
    Btotal[:, 0],
    Btotal[:, 1],
    Btotal[:, 2],
    length=arrow_length,
    normalize=False,
    color="g",
    label=r"$B_{total}$",
)

x, y, z = surface_coordinates(solution, radius=radius, ntheta=ntheta)
ax.plot_surface(np.asarray(x), np.asarray(y), np.asarray(z), alpha=0.3, color="k", label="Surface")

ax.legend()

plt.show()

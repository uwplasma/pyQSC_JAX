"""Generate an available-order boundary through the compatibility adapter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqsc_jax.near_axis import near_axis

RC = [1.0, 0.155, 0.0102]
ZS = [0.0, 0.154, 0.0111]
NFP = 2
ETABAR = 0.64
B2C = -0.00322
RADIUS = 0.05
NTHETA = 24
NPHI = 49
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/08_boundary_and_coordinates.png")

print("Constructing an r2 compatibility surface...")
field = near_axis(
    rc=RC,
    zs=ZS,
    nfp=NFP,
    etabar=ETABAR,
    B2c=B2C,
    nphi=31,
    order="r2",
)
x, y, z, radius = field.get_boundary(
    r=RADIUS,
    ntheta=NTHETA,
    nphi=NPHI,
)
print("boundary array shape:", x.shape)
print("R range [m]:", float(radius.min()), float(radius.max()))
print("Z range [m]:", float(z.min()), float(z.max()))

figure = plt.figure(figsize=(8.0, 3.8))
axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
axis_3d.plot_surface(
    np.asarray(x),
    np.asarray(y),
    np.asarray(z),
    cmap="viridis",
    alpha=0.8,
    linewidth=0,
)
axis_3d.set_box_aspect((1, 1, 1))
axis_3d.set_xlabel("x [m]")
axis_3d.set_ylabel("y [m]")
axis_3d.set_zlabel("z [m]")
axis_cross = figure.add_subplot(1, 2, 2)
axis_cross.plot(np.asarray(radius[:, 0]), np.asarray(z[:, 0]), linewidth=2)
axis_cross.set_aspect("equal")
axis_cross.set_xlabel("R [m]")
axis_cross.set_ylabel("Z [m]")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

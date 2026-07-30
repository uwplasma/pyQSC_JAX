"""Publication figure comparing QA and QH near-axis surfaces."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqsc_jax.near_axis import near_axis

CONFIGURATIONS = {
    "QA": {
        "rc": [1.0, 0.155, 0.0102],
        "zs": [0.0, 0.154, 0.0111],
        "nfp": 2,
        "etabar": 0.64,
        "B2c": -0.00322,
    },
    "QH": {
        "rc": [1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
        "zs": [0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
        "nfp": 4,
        "etabar": 1.569,
        "B2c": 0.1348,
    },
}
RADIUS = 0.05
NTHETA = 36
NPHI = 121
OUTPUT_STEM = Path("examples/output/publication/axis_and_surfaces")
SAVE_OUTPUT = True
SHOW_FIGURE = False

plt.style.use("seaborn-v0_8-whitegrid")
figure = plt.figure(figsize=(11.0, 4.6))
metadata = {"formal_surface_radius": RADIUS, "configurations": CONFIGURATIONS}

print("Rendering QA and QH available-order surfaces...")
for panel, (name, parameters) in enumerate(CONFIGURATIONS.items(), start=1):
    solution = near_axis(**parameters, nphi=61, order="r2")
    x, y, z, _ = solution.get_boundary(r=RADIUS, ntheta=NTHETA, nphi=NPHI)
    axis = figure.add_subplot(1, 2, panel, projection="3d")
    axis.plot_surface(
        np.asarray(x),
        np.asarray(y),
        np.asarray(z),
        cmap="viridis" if name == "QA" else "plasma",
        linewidth=0,
        antialiased=True,
        alpha=0.88,
    )
    axis.plot(
        np.asarray(x).mean(axis=0),
        np.asarray(y).mean(axis=0),
        np.asarray(z).mean(axis=0),
        color="black",
        linewidth=2.0,
    )
    axis.set_title(f"{name}: $\\iota={float(solution.iota):.3f}$")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_box_aspect((1, 1, 0.5))
figure.tight_layout()

try:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    commit = "unavailable"
metadata["git_commit"] = commit
if SAVE_OUTPUT:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("saved:", OUTPUT_STEM)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

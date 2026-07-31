"""Separate total, plasma, and external vacuum field jets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_field_split_components

CONFIGURATION = "plasma_stellarator"
FORMAL_RADIUS = 0.15
NPHI = 61
ANGULAR_RESOLUTION = 96
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/10_plasma_external_field_jet.png")

print("Solving the finite-pressure, zero-current stellarator...")
solution = qsc.solve_configuration(CONFIGURATION, nphi=NPHI)
assert float(solution.inputs.I2) == 0.0
assert float(solution.inputs.p2) != 0.0
print("torsion RMS [1/m]:", float((solution.torsion**2).mean() ** 0.5))
print("Separating the surface-free plasma and external jets...")
result = qsc.plasma_hessian_on_axis(
    solution,
    formal_radius=FORMAL_RADIUS,
    angular_resolution=ANGULAR_RESOLUTION,
)
enclosed_current = result.field.field.current_source.enclosed_toroidal_current
plasma_fraction = (
    (result.field.field.field**2).sum(axis=-1) / (solution.B_axis**2).sum(axis=-1)
) ** 0.5
print("enclosed toroidal current [A]:", float(enclosed_current))
print("minimum |B_plasma| / |B_total|:", float(plasma_fraction.min()))
print("mean |B_plasma| / |B_total|:", float(plasma_fraction.mean()))
plasma_norm = np.linalg.norm(np.asarray(result.field.field.field), axis=-1)
print("plasma |B| peak-to-peak / mean:", float(np.ptp(plasma_norm) / np.mean(plasma_norm)))
print("formal radius / singular radius:", float(FORMAL_RADIUS / solution.r_singularity))
print("external gradient STF components:", result.field.external_gradient_independent.shape[-1])
print("external Hessian STF components:", result.external_hessian_independent.shape[-1])
print("maximum external-gradient trace:", float(result.field.maximum_external_trace))
print("maximum external-Hessian trace:", float(result.maximum_external_trace))
print("estimated plasma-field remainder [T]:", float(result.field.field.estimated_field_remainder))
print(
    "estimated plasma-Hessian remainder [T/m^2]:",
    float(result.estimated_hessian_remainder),
)

figure, _ = plot_field_split_components(result, solution)
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

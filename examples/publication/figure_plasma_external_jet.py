"""Publication figure for the total/plasma/external 3+5+7 field jet."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_field_jet_norms

CONFIGURATION = "plasma_dominant_channel"
FORMAL_RADIUS = 0.2
NPHI = 121
ANGULAR_RESOLUTION = 128
OUTPUT_STEM = Path("examples/output/publication/plasma_external_jet")
README_PNG = Path("docs/_static/plasma_external_jet.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

plt.style.use("seaborn-v0_8-whitegrid")
print("Evaluating the surface-free plasma/external jet...")
solution = qsc.solve_configuration(CONFIGURATION, nphi=NPHI)
result = qsc.plasma_hessian_on_axis(
    solution,
    formal_radius=FORMAL_RADIUS,
    angular_resolution=ANGULAR_RESOLUTION,
)
plasma_fraction = (
    (
        (result.field.field.field**2).sum(axis=-1)
        / (solution.B_axis**2).sum(axis=-1)
    )
    ** 0.5
)
figure, axes = plot_field_jet_norms(result)
for label, axis in zip(("a", "b", "c"), axes, strict=True):
    axis.text(
        0.01,
        0.91,
        f"({label})",
        transform=axis.transAxes,
        fontweight="bold",
    )

try:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    commit = "unavailable"
metadata = {
    "git_commit": commit,
    "configuration": CONFIGURATION,
    "formal_radius": FORMAL_RADIUS,
    "nphi": NPHI,
    "angular_resolution": ANGULAR_RESOLUTION,
    "enclosed_current_amperes": float(result.field.field.current_source.enclosed_toroidal_current),
    "minimum_plasma_field_fraction": float(plasma_fraction.min()),
    "mean_plasma_field_fraction": float(plasma_fraction.mean()),
    "formal_radius_to_singular_radius": float(FORMAL_RADIUS / solution.r_singularity),
    "maximum_external_gradient_trace": float(result.field.maximum_external_trace),
    "maximum_external_hessian_trace": float(result.maximum_external_trace),
    "field_remainder": float(result.field.field.estimated_field_remainder),
    "hessian_remainder": float(result.estimated_hessian_remainder),
}
if SAVE_OUTPUT:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    README_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(README_PNG, dpi=120, bbox_inches="tight")
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("saved:", OUTPUT_STEM)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

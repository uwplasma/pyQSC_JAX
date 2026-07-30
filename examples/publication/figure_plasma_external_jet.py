"""Publication figure for the total/plasma/external 3+5+7 field jet."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc
from pyqsc_jax.plotting import field_split_frenet_components, plot_surface_3d

CONFIGURATION = "plasma_stellarator"
FORMAL_RADIUS = 0.45
DISPLAY_RADIUS = 0.18
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
    (result.field.field.field**2).sum(axis=-1) / (solution.B_axis**2).sum(axis=-1)
) ** 0.5
components = np.asarray(field_split_frenet_components(result, solution))
angle = np.asarray(solution.varphi * solution.inputs.axis.nfp / (2 * np.pi))
plasma_norm = np.linalg.norm(np.asarray(result.field.field.field), axis=1)
external_norm = np.linalg.norm(np.asarray(result.field.external_field), axis=1)

figure = plt.figure(figsize=(12.8, 8.0))
surface_axis = figure.add_subplot(2, 2, 1, projection="3d")
plot_surface_3d(
    solution,
    radius=DISPLAY_RADIUS,
    ntheta=32,
    ax=surface_axis,
    cmap="plasma",
)
surface_axis.view_init(elev=25, azim=35)
surface_axis.set_title(
    "screened finite-current stellarator\n"
    rf"$|\iota|={abs(float(solution.iota)):.3f}$, "
    rf"$r_\mathrm{{sing}}={float(solution.r_singularity):.3f}$ m"
)

fraction_axis = figure.add_subplot(2, 2, 2)
fraction_axis.plot(angle, np.asarray(plasma_fraction), color="tab:red", linewidth=2.3)
fraction_axis.axhline(0.30, color="black", linestyle="--", linewidth=1.2, label="30% gate")
fraction_axis.fill_between(angle, 0.30, np.asarray(plasma_fraction), alpha=0.18, color="tab:red")
fraction_axis.set_xlabel("Boozer angle / field period")
fraction_axis.set_ylabel(r"$|B_\mathrm{plasma}|/|B_\mathrm{total}|$")
fraction_axis.set_title("plasma contribution varies with angle")
fraction_axis.legend()

tangent_axis = figure.add_subplot(2, 2, 3)
for field_index, label in enumerate(("total", "plasma", "external")):
    tangent_axis.plot(angle, components[field_index, 0], label=label)
tangent_axis.set_xlabel("Boozer angle / field period")
tangent_axis.set_ylabel(r"$B_t$ [T]")
tangent_axis.set_title("tangent field component")
tangent_axis.legend(ncol=3)

transverse_axis = figure.add_subplot(2, 2, 4)
transverse_axis.plot(angle, components[1, 1], label=r"plasma $B_n$")
transverse_axis.plot(angle, components[2, 1], "--", label=r"external $B_n$")
transverse_axis.plot(angle, components[1, 2], label=r"plasma $B_b$")
transverse_axis.plot(angle, components[2, 2], "--", label=r"external $B_b$")
transverse_axis.set_xlabel("Boozer angle / field period")
transverse_axis.set_ylabel("transverse field [T]")
transverse_axis.set_title("plasma and external parts cancel to the total")
transverse_axis.legend(ncol=2, fontsize=8)
figure.tight_layout()

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
    "display_radius": DISPLAY_RADIUS,
    "nphi": NPHI,
    "angular_resolution": ANGULAR_RESOLUTION,
    "enclosed_current_amperes": float(result.field.field.current_source.enclosed_toroidal_current),
    "minimum_plasma_field_fraction": float(plasma_fraction.min()),
    "mean_plasma_field_fraction": float(plasma_fraction.mean()),
    "maximum_plasma_field_fraction": float(plasma_fraction.max()),
    "plasma_norm_peak_to_peak_over_mean": float(np.ptp(plasma_norm) / np.mean(plasma_norm)),
    "external_norm_peak_to_peak_over_mean": float(np.ptp(external_norm) / np.mean(external_norm)),
    "singular_radius": float(solution.r_singularity),
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

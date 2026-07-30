"""Publication figure for the differentiable pyQSC_JAX-to-VMEX bridge."""

import json
import subprocess
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_surface_3d

CASES = (
    ("qa", "vacuum QA"),
    ("plasma_stellarator", r"finite $p_2$, $I_2=0$"),
)
RADIUS = 0.02
QS_SURFACES = (0.2, 0.4, 0.6, 0.8, 1.0)
OUTPUT_STEM = Path("examples/output/publication/vmex_radial_profiles")
README_PNG = Path("docs/_static/vmex_radial_profiles.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

print("Solving vacuum and finite-beta radial equilibria with VMEX...")
records = []
try:
    for configuration, label in CASES:
        near_axis = qsc.solve_configuration(configuration, nphi=31)
        problem = qsc.to_vmex_problem(
            near_axis,
            r=RADIUS,
            qs_surfaces=QS_SURFACES,
            ntheta=8,
            mpol=3,
            ntor=2,
            ns_array=(7,),
            ftol=1.0e-7,
            max_iterations=1200,
            adjoint_tol=1.0e-8,
            multigrid=False,
        )
        equilibrium = problem.solve()
        records.append((label, near_axis, problem, equilibrium.quantities))
except ImportError as error:
    print(error)
    print("Skipping the optional VMEX publication figure.")
    raise SystemExit(0) from None

finite_label, finite_axis, finite_problem, finite_quantities = records[-1]
print("Differentiating the finite-beta magnetic well...")
well, gradient = jax.value_and_grad(
    lambda parameters: (
        qsc.vmex_radial_quantities(
            finite_problem,
            parameters,
        ).magnetic_well
    )
)(finite_problem.parameters)

plt.style.use("seaborn-v0_8-whitegrid")
figure = plt.figure(figsize=(12.8, 8.0))
surface_axis = figure.add_subplot(2, 2, 1, projection="3d")
plot_surface_3d(
    finite_axis,
    radius=0.08,
    ntheta=32,
    ax=surface_axis,
    cmap="magma",
)
surface_axis.view_init(elev=25, azim=35)
surface_axis.set_title(r"VMEX source boundary" "\n" r"finite $p_2$, exactly $I_2=0$")

iota_axis = figure.add_subplot(2, 2, 2)
qs_axis = figure.add_subplot(2, 2, 3)
well_axis = figure.add_subplot(2, 2, 4)
for label, near_axis, _problem, quantities in records:
    iota_axis.plot(
        np.asarray(quantities.s),
        np.asarray(quantities.iota),
        "-o",
        label=label,
    )
    iota_axis.scatter(
        [0],
        [float(near_axis.iota)],
        marker="x",
        s=55,
    )
    qs_axis.semilogy(
        np.asarray(quantities.qs_surfaces),
        np.asarray(quantities.quasisymmetry),
        "-o",
        label=label,
    )
iota_axis.set_xlabel(r"normalized toroidal flux $s$")
iota_axis.set_ylabel(r"$\iota(s)$ in pyQSC convention")
iota_axis.set_title("on-axis construction → radial equilibrium")
iota_axis.legend()
qs_axis.set_xlabel(r"normalized toroidal flux $s$")
qs_axis.set_ylabel("VMEX QS residual")
qs_axis.set_title("differentiable quasisymmetry profile")
qs_axis.legend()

wells = [float(item[3].magnetic_well) for item in records]
well_axis.bar([item[0] for item in records], wells, color=("tab:blue", "tab:red"))
well_axis.axhline(0, color="black", linewidth=1)
well_axis.set_ylabel(r"$(V'(0)-V'(1))/V'(0)$")
well_axis.set_title("magnetic well + implicit gradient")
well_axis.text(
    0.04,
    0.06,
    rf"$\partial W/\partial p_\mathrm{{scale}}={float(gradient.pres_scale):+.2e}$"
    "\n"
    rf"$\|\partial W/\partial RBC\|={float(np.linalg.norm(np.asarray(gradient.rbc))):.2e}$",
    transform=well_axis.transAxes,
    fontsize=9,
)
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
    "vmex_version": finite_problem.vmex_version,
    "vmex_validated_commit": finite_problem.validated_commit,
    "radius": RADIUS,
    "adjoint_tolerance": finite_problem.adjoint_tol,
    "cases": {
        label: {
            "iota": np.asarray(quantities.iota).tolist(),
            "quasisymmetry": np.asarray(quantities.quasisymmetry).tolist(),
            "magnetic_well": float(quantities.magnetic_well),
            "thermal_energy": float(quantities.thermal_energy),
        }
        for label, _near_axis, _problem, quantities in records
    },
    "finite_beta_well": float(well),
    "finite_beta_well_gradient_pres_scale": float(gradient.pres_scale),
    "finite_beta_well_gradient_rbc_norm": float(np.linalg.norm(np.asarray(gradient.rbc))),
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

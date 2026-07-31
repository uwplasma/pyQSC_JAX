"""Solve differentiable radial equilibrium quantities with optional VMEX."""

import os
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

CONFIGURATION = "qa"
RADIUS = 0.02
QS_SURFACES = (0.25, 0.5, 0.75, 1.0)
OUTPUT = Path("examples/output/14_vmex_radial_profiles.png")
SAVE_FIGURE = True
SHOW_FIGURE = False

if os.environ.get("PYQSC_RUN_VMEX") != "1":
    print("Set PYQSC_RUN_VMEX=1 to run the optional VMEX equilibrium example.")
    raise SystemExit(0)

print("Constructing the near-axis boundary...")
solution = qsc.solve_configuration(CONFIGURATION, nphi=31)
try:
    problem = qsc.to_vmex_problem(
        solution,
        r=RADIUS,
        qs_surfaces=QS_SURFACES,
        ntheta=8,
        mpol=3,
        ntor=2,
        ns_array=(7,),
        ftol=1.0e-7,
        max_iterations=1200,
        multigrid=False,
    )
except ImportError as error:
    print(error)
    print("Skipping the optional VMEX equilibrium example.")
    raise SystemExit(0) from None

print("Solving the converged fixed-boundary equilibrium with VMEX...")
result = problem.solve()
quantities = result.quantities
print("VMEX version:", problem.vmex_version)
print("iota(s):", np.asarray(quantities.iota))
print("quasisymmetry profile:", np.asarray(quantities.quasisymmetry))
print("magnetic well:", float(quantities.magnetic_well))

print("Differentiating magnetic well through the converged VMEX fixed point...")
well, gradient = jax.value_and_grad(
    lambda parameters: qsc.vmex_radial_quantities(problem, parameters).magnetic_well
)(problem.parameters)
print("magnetic well:", float(well))
print("d(well)/d(pres_scale):", float(gradient.pres_scale))
print("||d(well)/d(RBC)||:", float(np.linalg.norm(np.asarray(gradient.rbc))))

figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
axes[0].plot(np.asarray(quantities.s), np.asarray(quantities.iota), "-o")
axes[0].axhline(float(solution.iota), color="black", linestyle="--", label="near axis")
axes[0].set_xlabel(r"normalized toroidal flux $s$")
axes[0].set_ylabel(r"$\iota(s)$")
axes[0].legend()
axes[1].semilogy(
    np.asarray(quantities.qs_surfaces),
    np.asarray(quantities.quasisymmetry),
    "-o",
)
axes[1].set_xlabel(r"normalized toroidal flux $s$")
axes[1].set_ylabel("VMEX QS residual")
figure.tight_layout()

if SAVE_FIGURE:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

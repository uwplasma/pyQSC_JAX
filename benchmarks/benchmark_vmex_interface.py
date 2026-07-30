"""Record synchronized VMEX forward and implicit-gradient timings."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

import pyqsc_jax as qsc

CONFIGURATION = "plasma_stellarator"
RADIUS = 0.02
OUTPUT = Path(
    os.environ.get(
        "PYQSC_VMEX_BENCHMARK_OUTPUT",
        "benchmarks/results/vmex_interface.json",
    )
)

near_axis = qsc.solve_configuration(CONFIGURATION, nphi=31)
problem = qsc.to_vmex_problem(
    near_axis,
    r=RADIUS,
    qs_surfaces=(0.2, 0.4, 0.6, 0.8, 1.0),
    ntheta=8,
    mpol=3,
    ntor=2,
    ns_array=(7,),
    ftol=1.0e-7,
    max_iterations=1200,
    adjoint_tol=1.0e-8,
    multigrid=False,
)

jax.clear_caches()
start = perf_counter()
equilibrium = problem.solve()
jax.block_until_ready(equilibrium.quantities)
forward_seconds = perf_counter() - start


def objective(parameters):
    """Return the scalar used to benchmark VMEX's implicit gradient."""

    return qsc.vmex_radial_quantities(problem, parameters).magnetic_well


start = perf_counter()
well, gradient = jax.value_and_grad(objective)(problem.parameters)
jax.block_until_ready((well, gradient))
value_and_gradient_seconds = perf_counter() - start

repository = Path(__file__).resolve().parents[1]
try:
    commit = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    commit = "unavailable"
report = {
    "git_commit": commit,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "jax": jax.__version__,
    "jax_backend": jax.default_backend(),
    "jax_enable_x64": bool(jax.config.jax_enable_x64),
    "vmex": problem.vmex_version,
    "vmex_validated_commit": problem.validated_commit,
    "case": {
        "configuration": CONFIGURATION,
        "radius": RADIUS,
        "nphi": near_axis.inputs.nphi,
        "ntheta": problem.ntheta,
        "mpol_maximum": problem.mpol,
        "ntor": problem.ntor,
        "ns_array": np.asarray(problem.input.ns_array).tolist(),
        "adjoint_tolerance": problem.adjoint_tol,
    },
    "forward_seconds": forward_seconds,
    "value_and_gradient_seconds": value_and_gradient_seconds,
    "magnetic_well": float(well),
    "gradient_pres_scale": float(gradient.pres_scale),
    "gradient_rbc_norm": float(np.linalg.norm(np.asarray(gradient.rbc))),
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("VMEX forward [s]:", forward_seconds)
print("VMEX magnetic-well value + gradient [s]:", value_and_gradient_seconds)
print("saved:", OUTPUT)

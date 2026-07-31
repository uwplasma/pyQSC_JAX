"""Compare optimizers from screened stellarator-database configuration 57409."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

import pyqsc_jax as qsc

NPHI = 121
MODES = (1, 2, 3)
OUTPUT = Path(
    os.environ.get("PYQSC_B20_BENCHMARK_OUTPUT", "benchmarks/results/b20_optimizers.json")
)
RC = (
    1.0,
    -0.51677144,
    -0.009499784,
    -0.005914526,
)
ZS = (
    0.0,
    -0.5420635,
    -0.012225689,
    -0.0059485724,
)
AXIS = qsc.Axis(rc=RC, zs=ZS, nfp=4)
VARIABLE_INDICES = qsc.stellarator_symmetric_variable_indices(AXIS, modes=MODES)
INITIAL = AXIS.dofs[jnp.asarray(VARIABLE_INDICES)]
HALF_WIDTH = jnp.asarray((0.14, 0.07, 0.03, 0.14, 0.07, 0.03))
LOWER = INITIAL - HALF_WIDTH
UPPER = INITIAL + HALF_WIDTH
ETABAR = -1.3295174
P2 = -23501.281
SOURCE_DATABASE_ID = 57409


def projected_residual(variables):
    axis = AXIS.with_dofs(AXIS.dofs.at[jnp.asarray(VARIABLE_INDICES)].set(variables))
    solution = qsc.solve(
        axis=axis,
        etabar=ETABAR,
        B0=1.0,
        B2c=0.0,
        p2=P2,
        nphi=NPHI,
        order="r2",
    )
    optimized = qsc.optimize_B2c(solution)
    weights = optimized.solution.geometry.d_l_d_phi
    return (
        jnp.sqrt(weights / jnp.sum(weights))
        * optimized.diagnostics.anomaly
        / optimized.solution.inputs.B0
    )


compiled_residual = jax.jit(projected_residual)
compiled_jacobian = jax.jit(jax.jacrev(projected_residual))
compiled_value_and_gradient = jax.jit(
    jax.value_and_grad(lambda value: 0.5 * jnp.sum(projected_residual(value) ** 2))
)


def residual_numpy(variables):
    return np.asarray(compiled_residual(jnp.asarray(variables)), dtype=float)


def jacobian_numpy(variables):
    return np.asarray(compiled_jacobian(jnp.asarray(variables)), dtype=float)


def value_and_gradient_numpy(variables):
    value, gradient = compiled_value_and_gradient(jnp.asarray(variables))
    return float(value), np.asarray(gradient, dtype=float)


def weighted_l2(variables):
    return float(np.linalg.norm(residual_numpy(variables)))


print("Compiling shared B20 residual and derivatives...")
compiled_residual(INITIAL).block_until_ready()
compiled_jacobian(INITIAL).block_until_ready()
compiled_value_and_gradient(INITIAL)[0].block_until_ready()
rows = [
    {
        "method": "exact B2c only",
        "seconds": 0.0,
        "function_evaluations": 1,
        "weighted_l2": weighted_l2(INITIAL),
    }
]

start = perf_counter()
lbfgsb = minimize(
    value_and_gradient_numpy,
    np.asarray(INITIAL),
    method="L-BFGS-B",
    jac=True,
    bounds=list(zip(np.asarray(LOWER), np.asarray(UPPER), strict=True)),
    options={"maxiter": 60, "ftol": 1.0e-18, "gtol": 1.0e-12},
)
rows.append(
    {
        "method": "SciPy L-BFGS-B",
        "seconds": perf_counter() - start,
        "function_evaluations": int(lbfgsb.nfev),
        "weighted_l2": weighted_l2(lbfgsb.x),
    }
)

start = perf_counter()
least_squares_result = least_squares(
    residual_numpy,
    np.asarray(INITIAL),
    jac=jacobian_numpy,
    bounds=(np.asarray(LOWER), np.asarray(UPPER)),
    max_nfev=60,
    xtol=1.0e-13,
    ftol=1.0e-13,
    gtol=1.0e-13,
    x_scale="jac",
)
rows.append(
    {
        "method": "SciPy least_squares",
        "seconds": perf_counter() - start,
        "function_evaluations": int(least_squares_result.nfev),
        "weighted_l2": weighted_l2(least_squares_result.x),
    }
)

start = perf_counter()
differential = differential_evolution(
    lambda variables: weighted_l2(variables) ** 2,
    list(zip(np.asarray(LOWER), np.asarray(UPPER), strict=True)),
    seed=7,
    popsize=4,
    maxiter=4,
    polish=False,
    updating="immediate",
)
rows.append(
    {
        "method": "SciPy differential_evolution, low budget",
        "seconds": perf_counter() - start,
        "function_evaluations": int(differential.nfev),
        "weighted_l2": weighted_l2(differential.x),
    }
)

problem = qsc.AxisSearchProblem(
    axis=AXIS,
    variable_indices=VARIABLE_INDICES,
    lower_bounds=LOWER,
    upper_bounds=UPPER,
    etabar=ETABAR,
    B0=1.0,
    p2=P2,
    nphi=NPHI,
    criteria=qsc.Criteria.from_curvo_2025(minimum_abs_iota=0.4),
)
options = qsc.AxisSearchOptions(
    coarse_samples=12,
    local_starts=3,
    maximum_iterations=20,
    verification_multipliers=(1,),
    verification_tail_tolerance=1.0,
)
start = perf_counter()
multistart = qsc.search_axis(problem, options=options)
rows.append(
    {
        "method": "pyQSC_JAX multistart Levenberg-Marquardt",
        "seconds": perf_counter() - start,
        "function_evaluations": multistart.search_budget,
        "weighted_l2": None if multistart.best is None else multistart.best.primary_residual,
        "status": multistart.status,
        "distinct_basins": multistart.distinct_basins,
    }
)

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
    "jax": jax.__version__,
    "jax_backend": jax.default_backend(),
    "jax_enable_x64": bool(jax.config.jax_enable_x64),
    "nphi": NPHI,
    "modes": MODES,
    "source_database_id": SOURCE_DATABASE_ID,
    "source_url": (f"https://stellarator.physics.wisc.edu/app/plot/{SOURCE_DATABASE_ID}"),
    "selected_configuration": "b20_optimized_good",
    "timing_note": (
        "SciPy timings exclude shared residual/Jacobian compilation; the "
        "pyQSC_JAX multistart timing includes compilation of its independent closure."
    ),
    "results": rows,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for row in rows:
    print(row)
print("saved:", OUTPUT)

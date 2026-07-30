"""Run a small deterministic global-to-local B20 axis search."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

INITIAL_AXIS = qsc.Axis(
    rc=[1.0, 0.155, 0.0102],
    zs=[0.0, 0.154, 0.0111],
    nfp=2,
)
ETABAR = 0.64
NPHI = 15
COARSE_SAMPLES = 3
LOCAL_STARTS = 1
MAXIMUM_ITERATIONS = 3
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/06_optimize_axis_B20.png")

variable_indices = qsc.stellarator_symmetric_variable_indices(
    INITIAL_AXIS,
    modes=(1,),
)
problem = qsc.AxisSearchProblem(
    axis=INITIAL_AXIS,
    variable_indices=variable_indices,
    lower_bounds=jnp.asarray([0.11, 0.11]),
    upper_bounds=jnp.asarray([0.19, 0.19]),
    etabar=ETABAR,
    nphi=NPHI,
    selector="minimum_maximum_elongation",
)
options = qsc.AxisSearchOptions(
    coarse_samples=COARSE_SAMPLES,
    local_starts=LOCAL_STARTS,
    maximum_iterations=MAXIMUM_ITERATIONS,
    verification_multipliers=(1,),
    verification_tail_tolerance=1.0,
)

print("Searching bounded axis coefficients...")
result = qsc.search_axis(problem, options=options)
print("status:", result.status)
print("search budget:", result.search_budget)
print("distinct basins:", result.distinct_basins)
print("message:", result.message)
if result.best is None:
    raise RuntimeError("The example search did not produce a candidate.")
print("best variables:", np.asarray(result.best.variables))
print("best B20 residual:", result.best.primary_residual)
print("global certificate:", result.global_certificate)

figure, axis = plt.subplots(figsize=(6.2, 3.8))
axis.scatter(
    np.arange(result.coarse_residuals.size),
    np.asarray(result.coarse_residuals),
    label="coarse samples",
)
axis.axhline(
    result.best.primary_residual,
    color="tab:red",
    linewidth=2,
    label="best refined basin",
)
axis.set_yscale("log")
axis.set_xlabel("deterministic sample index")
axis.set_ylabel("weighted B20 residual")
axis.legend()
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

"""Apply the Curvo screen during a reproducible zero-current stellarator search."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

CONFIGURATION = "plasma_stellarator"
I2 = 0.0
NPHI = 31
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/07_good_stellarator_search.png")

configuration = qsc.get_configuration(CONFIGURATION)
axis = qsc.Axis(
    rc=configuration.rc,
    zs=configuration.zs,
    nfp=configuration.nfp,
)
variable_indices = qsc.stellarator_symmetric_variable_indices(axis, modes=(3,))
criteria = qsc.Criteria.from_curvo_2025(minimum_abs_iota=0.4)
problem = qsc.AxisSearchProblem(
    axis=axis,
    variable_indices=variable_indices,
    lower_bounds=jnp.asarray([0.002, 0.002]),
    upper_bounds=jnp.asarray([0.008, 0.008]),
    etabar=configuration.etabar,
    I2=I2,
    p2=configuration.p2,
    nphi=NPHI,
    criteria=criteria,
    selector="maximum_singular_radius",
)
options = qsc.AxisSearchOptions(
    coarse_samples=2,
    local_starts=1,
    maximum_iterations=2,
    verification_multipliers=(1, 2),
    verification_tail_tolerance=1.0,
)

print("Running criteria-aware search from database ID 52521...")
result = qsc.search_axis(problem, options=options)
if result.best is None or result.best.criteria_report is None:
    raise RuntimeError("The criteria-aware example did not produce a report.")
report = result.best.criteria_report
print("status:", result.status)
print("criteria passed:", report.passed)
print("I2:", float(result.best.solution.inputs.I2))
print("p2:", float(result.best.solution.inputs.p2))
print("iota:", float(result.best.solution.iota))
print("singular radius [m]:", float(result.best.solution.r_singularity))
print("torsion RMS [1/m]:", float((result.best.solution.torsion**2).mean() ** 0.5))
for evaluation in report.evaluations:
    print(
        evaluation.name,
        "value=",
        float(evaluation.value),
        "margin=",
        float(evaluation.margin),
        evaluation.units,
    )

names = [evaluation.name for evaluation in report.evaluations]
margins = np.asarray([evaluation.margin for evaluation in report.evaluations])
figure, axis = plt.subplots(figsize=(7.4, 4.0))
axis.barh(names, margins, color=np.where(margins >= 0, "tab:green", "tab:red"))
axis.axvline(0, color="black", linewidth=1)
axis.set_xlabel("signed criterion margin")
figure.tight_layout()
if SAVE_OUTPUT:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print("saved:", OUTPUT)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

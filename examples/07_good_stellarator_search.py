"""Apply a named screening profile during a reproducible axis search."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

AXIS = qsc.Axis(rc=[1.0], zs=[0.0], nfp=1)
ETABAR = 1.0
I2 = 0.1
NPHI = 15
SAVE_OUTPUT = True
SHOW_FIGURE = False
OUTPUT = Path("examples/output/07_good_stellarator_search.png")

# This deliberately permissive profile demonstrates mechanics; it is not a
# claim that a circular channel passes the published Curvo screening profile.
criteria = qsc.Criteria.from_curvo_2025(
    minimum_abs_iota=0.0,
    maximum_elongation=20.0,
    minimum_L_grad_B=0.0,
    minimum_axis_radius=0.0,
    minimum_singular_radius=0.0,
    minimum_L_grad_grad_B=0.0,
    maximum_B20_variation=10.0,
    minimum_beta=0.0,
    minimum_DMerc_times_r2=-1.0e6,
)
problem = qsc.AxisSearchProblem(
    axis=AXIS,
    variable_indices=(0,),
    lower_bounds=jnp.asarray([0.8]),
    upper_bounds=jnp.asarray([1.2]),
    etabar=ETABAR,
    I2=I2,
    nphi=NPHI,
    criteria=criteria,
    selector="maximum_criteria_margin",
)
options = qsc.AxisSearchOptions(
    coarse_samples=2,
    local_starts=1,
    maximum_iterations=2,
    verification_multipliers=(1, 2),
)

print("Running criteria-aware search...")
result = qsc.search_axis(problem, options=options)
if result.best is None or result.best.criteria_report is None:
    raise RuntimeError("The criteria-aware example did not produce a report.")
report = result.best.criteria_report
print("status:", result.status)
print("criteria passed:", report.passed)
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

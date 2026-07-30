"""Publication figure for spectral resolution convergence."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

CONFIGURATION = "database_qa_139524"
RESOLUTIONS = (15, 31, 61, 121)
OUTPUT_STEM = Path("examples/output/publication/convergence")
README_PNG = Path("docs/_static/convergence.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

plt.style.use("seaborn-v0_8-whitegrid")
print("Solving database QA ID 139524 on successively finer grids...")
solutions = [qsc.solve_configuration(CONFIGURATION, nphi=resolution) for resolution in RESOLUTIONS]
reference = solutions[-1]
iota_errors = np.asarray(
    [abs(float(solution.iota - reference.iota)) for solution in solutions[:-1]]
)
b20_values = np.asarray(
    [float(qsc.b20_diagnostics(solution).weighted_l2) for solution in solutions]
)
b20_errors = np.abs(b20_values[:-1] - b20_values[-1])
floor = np.finfo(float).eps

figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
axes[0].loglog(
    RESOLUTIONS[:-1],
    np.maximum(iota_errors, floor),
    "-o",
    linewidth=2,
)
axes[0].set_xlabel("toroidal grid points")
axes[0].set_ylabel(r"$|\iota_N-\iota_{121}|$")
axes[1].loglog(
    RESOLUTIONS[:-1],
    np.maximum(b20_errors, floor),
    "-o",
    linewidth=2,
    color="tab:red",
)
axes[1].set_xlabel("toroidal grid points")
axes[1].set_ylabel(r"$|R_{B20,N}-R_{B20,121}|$")
figure.suptitle(r"Resolution audit: $\iota$ converges early, $B_{20}$ needs a finer grid")
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
    "resolutions": RESOLUTIONS,
    "iota": [float(solution.iota) for solution in solutions],
    "B20_weighted_l2": b20_values.tolist(),
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

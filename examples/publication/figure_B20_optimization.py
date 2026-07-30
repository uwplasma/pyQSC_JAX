"""Publication figure for exact affine B2c elimination."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

CONFIGURATION = "qa"
INITIAL_B2C = 0.0
NPHI = 121
OUTPUT_STEM = Path("examples/output/publication/B20_optimization")
README_PNG = Path("docs/_static/B20_optimization.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

plt.style.use("seaborn-v0_8-whitegrid")
print("Computing exact B2c elimination at publication resolution...")
initial = qsc.solve_configuration(CONFIGURATION, B2c=INITIAL_B2C, nphi=NPHI)
result = qsc.optimize_B2c(initial)
optimized = result.solution
angle = np.asarray(initial.varphi * initial.inputs.axis.nfp / (2 * np.pi))

figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
axes[0].plot(angle, np.asarray(initial.B20_anomaly), linewidth=2.2, label="initial")
axes[0].plot(
    angle,
    np.asarray(optimized.B20_anomaly),
    linewidth=2.2,
    label=r"exact $B_{2c}^{\star}$",
)
axes[0].set_xlabel("Boozer angle / field period")
axes[0].set_ylabel(r"$B_{20}-\langle B_{20}\rangle$ [T/m$^2$]")
axes[0].legend()
diagnostic_names = ("weighted $L^2$", "dense maximum", "peak-to-peak")
initial_diagnostics = qsc.b20_diagnostics(initial)
axes[1].bar(
    np.arange(3) - 0.18,
    (
        float(initial_diagnostics.weighted_l2),
        float(initial_diagnostics.grid_maximum),
        float(initial_diagnostics.peak_to_peak),
    ),
    width=0.36,
    label="initial",
)
axes[1].bar(
    np.arange(3) + 0.18,
    (
        float(result.diagnostics.weighted_l2),
        float(result.diagnostics.grid_maximum),
        float(result.diagnostics.peak_to_peak),
    ),
    width=0.36,
    label="optimized",
)
axes[1].set_xticks(np.arange(3), diagnostic_names, rotation=15)
axes[1].set_ylabel(r"$B_{20}$ nonuniformity [T/m$^2$]")
axes[1].legend()
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
    "nphi": NPHI,
    "initial_B2c": INITIAL_B2C,
    "optimal_B2c": float(result.B2c_optimal),
    "initial_weighted_l2": float(initial_diagnostics.weighted_l2),
    "optimal_weighted_l2": float(result.diagnostics.weighted_l2),
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

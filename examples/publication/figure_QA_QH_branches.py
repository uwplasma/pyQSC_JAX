"""Publication figure comparing QA and QH etabar branches."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

BRANCHES = {
    "QA": {
        "axis": qsc.Axis(
            rc=[1.0, -0.06883207, 0.0017516185, 0.023231717],
            zs=[0.0, -0.28447896, 0.074662544, 0.07483574],
            nfp=1,
        ),
        "etabar": np.linspace(-1.15, -0.4, 13),
        "color": "tab:blue",
        "database_id": 139524,
    },
    "QH": {
        "axis": qsc.Axis(
            rc=[1.0, -0.53677857, -0.046455786, -0.0070183445],
            zs=[0.0, -0.5888703, -0.04447083, -0.009581006],
            nfp=4,
        ),
        "etabar": np.linspace(0.8, 1.8, 13),
        "color": "tab:orange",
        "database_id": 3,
    },
}
NPHI = 31
OUTPUT_STEM = Path("examples/output/publication/QA_QH_branches")
README_PNG = Path("docs/_static/QA_QH_branches.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
metadata = {"nphi": NPHI, "branches": {}}
print("Sampling QA and QH first-order branches...")
for name, branch in BRANCHES.items():
    etabar_values = branch["etabar"]
    solutions = [
        qsc.solve(axis=branch["axis"], etabar=etabar, nphi=NPHI) for etabar in etabar_values
    ]
    iota = np.asarray([float(solution.iota) for solution in solutions])
    elongation = np.asarray([float(solution.elongation.max()) for solution in solutions])
    axes[0].plot(etabar_values, iota, "-o", color=branch["color"], label=name)
    axes[1].plot(etabar_values, elongation, "-o", color=branch["color"], label=name)
    metadata["branches"][name] = {
        "source_database_id": branch["database_id"],
        "etabar": etabar_values.tolist(),
        "iota": iota.tolist(),
        "maximum_elongation": elongation.tolist(),
    }
axes[0].set_xlabel(r"$\bar{\eta}$ [m$^{-1}$]")
axes[0].set_ylabel(r"$\iota$")
axes[0].axhline(0.0, color="0.3", linewidth=0.8)
axes[1].set_xlabel(r"$\bar{\eta}$ [m$^{-1}$]")
axes[1].set_ylabel("maximum elongation")
for axis in axes:
    axis.legend()
figure.suptitle("QA and QH topology respond differently to the same design variable")
figure.tight_layout()

try:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    commit = "unavailable"
metadata["git_commit"] = commit
if SAVE_OUTPUT:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    README_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(README_PNG, dpi=130, bbox_inches="tight")
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("saved:", OUTPUT_STEM)
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(figure)

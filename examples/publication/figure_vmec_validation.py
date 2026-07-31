"""Publication figure for VMEC export timing and on-axis-iota convergence."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "reports"
    / "2026-07-30-b20-vmec-apple-m4.json"
)
OUTPUT_STEM = Path("examples/output/publication/vmec_validation")
README_PNG = Path("docs/_static/vmec_validation.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

print("Loading the frozen VMEC validation report...")
report = json.loads(REPORT.read_text(encoding="utf-8"))
convergence = report["vmec_radius_convergence"]
radii = np.asarray([row["radius"] for row in convergence["results"]])
errors = np.asarray([row["relative_iota_error"] for row in convergence["results"]])
timing = report["vmec_export"]

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
axes[0].loglog(radii, errors, "o-", linewidth=2.2, label="VMEC")
axes[0].loglog(
    radii,
    errors[0] * (radii / radii[0]) ** 2,
    "--",
    linewidth=1.7,
    label=r"$O(r^2)$",
)
axes[0].set_xlabel("export radius [m]")
axes[0].set_ylabel("relative on-axis iota error")
axes[0].set_title("near-axis convergence")
axes[0].legend()
cold_ms = 1000 * timing["cold_compile_and_execute_seconds"]
warm_ms = 1000 * timing["warm_median_seconds"]
axes[1].bar(("compile + execute", "warm execute"), (cold_ms, warm_ms), color=("0.45", "tab:blue"))
axes[1].set_yscale("log")
axes[1].set_ylabel("conversion time [ms]")
axes[1].set_title("vectorized JAX exporter")
axes[1].text(
    0.5,
    0.07,
    f"{timing['maximum_R_reconstruction_error_m'] * 1e6:.2f} µm max R error\n"
    f"{timing['maximum_Z_reconstruction_error_m'] * 1e6:.2f} µm max Z error",
    ha="center",
    transform=axes[1].transAxes,
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
    "source_report": str(REPORT),
    "near_axis_iota": convergence["near_axis_iota"],
    "radius_convergence": convergence["results"],
    "vmec_version": convergence["vmec_version"],
    "export": timing,
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

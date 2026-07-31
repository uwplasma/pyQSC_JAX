"""Publication figure comparing B20 optimizer strategies and the final refinement."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyqsc_jax as qsc

REPOSITORY = Path(__file__).resolve().parents[2]
REPORT = REPOSITORY / "benchmarks" / "reports" / "2026-07-30-b20-optimizers-apple-m4.json"
OUTPUT_STEM = Path("examples/output/publication/optimizer_comparison")
README_PNG = Path("docs/_static/optimizer_comparison.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

report = json.loads(REPORT.read_text(encoding="utf-8"))
rows = report["results"]
production = qsc.solve_configuration("b20_optimized_good", nphi=report["nphi"])
production_residual = float(qsc.b20_diagnostics(production).weighted_l2)
short_names = (
    r"exact $B_{2c}$",
    "L-BFGS-B",
    "least squares",
    "differential\nevolution",
    "multistart LM",
    "staged 8-mode\nrefinement",
)
residuals = np.asarray([row["weighted_l2"] for row in rows] + [production_residual])
colors = ("#7f8c8d", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#117864")

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.45))
bars = axes[0].bar(short_names, residuals, color=colors, width=0.72)
axes[0].set_yscale("log")
axes[0].set_ylabel(r"weighted $\|P B_{20}\|_2$ [T m$^{-2}$]")
axes[0].set_title("Same dense target, visibly different outcomes")
axes[0].tick_params(axis="x", rotation=20)
axes[0].bar_label(
    bars,
    labels=[f"{value:.1e}" for value in residuals],
    padding=3,
    fontsize=8,
)
axes[0].text(
    0.98,
    0.36,
    f"final improvement\n{residuals[0] / production_residual:,.1e}×",
    transform=axes[0].transAxes,
    ha="right",
    va="bottom",
    color="#117864",
    fontweight="bold",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
)

screened_rows = rows[1:]
evaluations = np.asarray([row["function_evaluations"] for row in screened_rows])
screened_residuals = np.asarray([row["weighted_l2"] for row in screened_rows])
seconds = np.asarray([row["seconds"] for row in screened_rows])
point_colors = colors[1:5]
point_labels = ("L-BFGS-B", "least squares", "low-budget DE", "multistart LM")
sizes = 75.0 + 55.0 * np.log10(1.0 + seconds / seconds.min())
for evaluation, residual, size, color, label, elapsed in zip(
    evaluations,
    screened_residuals,
    sizes,
    point_colors,
    point_labels,
    seconds,
    strict=True,
):
    axes[1].scatter(
        evaluation,
        residual,
        s=size,
        color=color,
        edgecolor="white",
        linewidth=1.0,
        label=f"{label} • {elapsed:.2f} s",
        zorder=3,
    )
axes[1].set_yscale("log")
axes[1].set_xlabel("objective evaluations")
axes[1].set_ylabel(r"weighted $\|P B_{20}\|_2$")
axes[1].set_title("Budget, basin coverage, and local accuracy")
axes[1].legend(fontsize=8, frameon=True)
axes[1].annotate(
    "best local screen",
    (evaluations[1], screened_residuals[1]),
    xytext=(15, 30),
    textcoords="offset points",
    ha="left",
    arrowprops={"arrowstyle": "->", "color": "#117864"},
    color="#117864",
    fontweight="bold",
)

figure.suptitle(r"Screen broadly, refine locally: $B_{20}$ becomes nearly constant")
figure.text(
    0.5,
    -0.015,
    (
        "Database ID 57409 • nphi=121 • exact B2c elimination in every axis "
        "evaluation • final point uses staged Fourier continuation"
    ),
    ha="center",
    fontsize=8,
    color="0.35",
)
figure.tight_layout()

metadata = {
    "report": str(REPORT.relative_to(REPOSITORY)),
    "report_git_commit": report["git_commit"],
    "source_database_id": report["source_database_id"],
    "production_configuration": "b20_optimized_good",
    "production_weighted_l2": production_residual,
    "production_improvement_over_exact_B2c": float(residuals[0] / production_residual),
}
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

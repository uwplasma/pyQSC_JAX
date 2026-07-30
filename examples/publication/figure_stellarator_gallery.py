"""Publication gallery of the bundled QA, QH, B20, and plasma stellarators."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_surface_3d

CASES = (
    ("qa", "QA reference", 0.055, "viridis"),
    ("qh", "QH reference", 0.055, "plasma"),
    ("b20_optimized_qa", r"$B_{20}$-optimized QA", 0.075, "cividis"),
    ("plasma_stellarator", "finite-current plasma case", 0.12, "magma"),
)
NPHI = 121
OUTPUT_STEM = Path("examples/output/publication/stellarator_gallery")
README_PNG = Path("docs/_static/stellarator_gallery.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

plt.style.use("seaborn-v0_8-whitegrid")
figure = plt.figure(figsize=(12.0, 9.2))
metadata = {"nphi": NPHI, "configurations": {}}

print("Rendering the bundled stellarator gallery...")
for panel, (name, title, radius, cmap) in enumerate(CASES, start=1):
    solution = qsc.solve_configuration(name, nphi=NPHI)
    axis = figure.add_subplot(2, 2, panel, projection="3d")
    plot_surface_3d(
        solution,
        radius=radius,
        ntheta=36,
        ax=axis,
        cmap=cmap,
    )
    axis.view_init(elev=24, azim=35)
    if name == "b20_optimized_qa":
        diagnostic = rf"$\|P B_{{20}}\|_2={float(solution.B20_residual):.2e}$"
    elif name == "plasma_stellarator":
        diagnostic = r"$\min |B_p|/|B|=32.7\%$ at $a=0.2$ m"
    else:
        diagnostic = rf"$\iota={float(solution.iota):.3f}$"
    axis.set_title(
        title + "\n" + diagnostic + rf", $r_\mathrm{{sing}}={float(solution.r_singularity):.3f}$ m"
    )
    metadata["configurations"][name] = {
        "surface_radius": radius,
        "iota": float(solution.iota),
        "B20_residual": float(solution.B20_residual),
        "singular_radius": float(solution.r_singularity),
        "surface_to_singular_radius": radius / float(solution.r_singularity),
    }
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

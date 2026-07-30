"""Publication figure for measured JIT, JVP, and VMAP performance."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
REPORT = REPOSITORY / "benchmarks" / "reports" / "2026-07-29-apple-m4.json"
OUTPUT_STEM = Path("examples/output/publication/core_performance")
README_PNG = Path("docs/_static/core_performance.png")
SAVE_OUTPUT = True
SHOW_FIGURE = False

report = json.loads(REPORT.read_text(encoding="utf-8"))
rows = sorted(report["results"], key=lambda row: row["nphi"])
nphi = np.asarray([row["nphi"] for row in rows])
cold_ms = np.asarray([1.0e3 * row["cold_compile_and_execute_seconds"] for row in rows])
warm_ms = np.asarray(
    [
        1.0e3
        * row.get(
            "warm_median_seconds",
            float(np.median(row["warm_samples_seconds"])),
        )
        for row in rows
    ]
)
speedups = cold_ms / warm_ms
finest = rows[-1]
batch_size = int(finest["vmap_batch_size"])
latencies_ms = np.asarray(
    [
        1.0e3
        * finest.get(
            "warm_median_seconds",
            float(np.median(finest["warm_samples_seconds"])),
        ),
        1.0e3
        * finest.get(
            "jvp_warm_median_seconds",
            float(np.median(finest["jvp_warm_samples_seconds"])),
        ),
        1.0e3
        * finest.get(
            "vmap_warm_median_seconds",
            float(np.median(finest["vmap_warm_samples_seconds"])),
        ),
    ]
)

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
width = 5.5
axes[0].bar(
    nphi - width / 2,
    cold_ms,
    width=width,
    label="compile + first execution",
    color="#34495e",
)
axes[0].bar(
    nphi + width / 2,
    warm_ms,
    width=width,
    label="warm median",
    color="#1abc9c",
)
axes[0].set_yscale("log")
axes[0].set_xlabel("toroidal grid points")
axes[0].set_ylabel("wall time [ms, log scale]")
axes[0].set_xticks(nphi)
axes[0].legend(frameon=True)
for x_value, warm_value, speedup in zip(nphi, warm_ms, speedups, strict=True):
    axes[0].annotate(
        f"{speedup:,.0f}×",
        (x_value + width / 2, warm_value),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#087f6d",
        fontweight="bold",
    )

labels = ("solve", "solve + JVP", f"VMAP × {batch_size}")
colors = ("#3498db", "#9b59b6", "#f39c12")
bars = axes[1].bar(labels, latencies_ms, color=colors, width=0.62)
axes[1].set_ylabel("warm wall time [ms]")
axes[1].set_title(f"Composable transforms at $n_\\phi={finest['nphi']}$")
axes[1].bar_label(bars, fmt="%.3f ms", padding=4)
per_case_us = 1.0e3 * latencies_ms[-1] / batch_size
axes[1].text(
    0.05,
    0.92,
    f"{batch_size} batched solutions\n{per_case_us:.1f} μs per case",
    transform=axes[1].transAxes,
    ha="left",
    va="top",
    color="#9c640c",
    fontweight="bold",
)
axes[1].set_ylim(0.0, 1.25 * float(latencies_ms.max()))

figure.suptitle("JAX compilation is amortized; derivatives and batches stay fast")
figure.text(
    0.5,
    -0.01,
    (
        f"{report.get('hardware', report['platform'])} • "
        f"JAX {report['jax']} {report['jax_backend']} "
        f"• x64={report['jax_enable_x64']} • synchronized medians"
    ),
    ha="center",
    fontsize=8,
    color="0.35",
)
figure.tight_layout()

metadata = {
    "report": str(REPORT.relative_to(REPOSITORY)),
    "report_git_commit": report["git_commit"],
    "cold_to_warm_speedup": dict(
        zip((str(value) for value in nphi), speedups.tolist(), strict=True)
    ),
    "finest_nphi": int(finest["nphi"]),
    "finest_warm_latency_ms": latencies_ms.tolist(),
    "finest_vmap_per_case_microseconds": float(per_case_us),
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

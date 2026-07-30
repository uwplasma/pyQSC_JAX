"""Record synchronized cold and warm VMEC boundary-export timings."""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path

import jax

import pyqsc_jax as qsc

RADIUS = 0.03
NPHI = 61
NTHETA = 40
MPOL = 12
NTOR = 14
WARM_REPETITIONS = 7
OUTPUT = Path(
    os.environ.get("PYQSC_VMEC_BENCHMARK_OUTPUT", "benchmarks/results/vmec_export.json")
)

solution = qsc.Qsc(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
    nphi=NPHI,
    order="r2",
)
with tempfile.TemporaryDirectory() as temporary_directory:
    destination = Path(temporary_directory) / "input.benchmark"
    jax.clear_caches()
    cold = qsc.to_vmec(
        solution,
        destination,
        r=RADIUS,
        ntheta=NTHETA,
        mpol=MPOL,
        ntor=NTOR,
    )
    warm = [
        qsc.to_vmec(
            solution,
            destination,
            r=RADIUS,
            ntheta=NTHETA,
            mpol=MPOL,
            ntor=NTOR,
        ).conversion_seconds
        for _ in range(WARM_REPETITIONS)
    ]

repository = Path(__file__).resolve().parents[1]
try:
    commit = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    commit = "unavailable"
report = {
    "git_commit": commit,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "jax": jax.__version__,
    "jax_backend": jax.default_backend(),
    "jax_enable_x64": bool(jax.config.jax_enable_x64),
    "case": {
        "radius": RADIUS,
        "nphi": NPHI,
        "ntheta": NTHETA,
        "mpol": MPOL,
        "ntor": NTOR,
    },
    "cold_compile_and_execute_seconds": cold.conversion_seconds,
    "warm_median_seconds": statistics.median(warm),
    "warm_samples_seconds": warm,
    "maximum_toroidal_angle_residual": float(
        cold.boundary.maximum_toroidal_angle_residual
    ),
    "maximum_R_reconstruction_error": float(
        cold.boundary.maximum_R_reconstruction_error
    ),
    "maximum_Z_reconstruction_error": float(
        cold.boundary.maximum_Z_reconstruction_error
    ),
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("cold compile + execution [s]:", cold.conversion_seconds)
print("warm median [s]:", report["warm_median_seconds"])
print("saved:", OUTPUT)

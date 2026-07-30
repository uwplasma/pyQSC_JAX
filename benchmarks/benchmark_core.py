"""Record synchronized first-order compile, execution, JVP, and VMAP timings."""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import pyqsc_jax as qsc

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
ETABAR = -0.9
RESOLUTIONS = (15, 31, 61)
WARM_REPETITIONS = 7
BATCH_SIZE = 8
OUTPUT = Path(os.environ.get("PYQSC_BENCHMARK_OUTPUT", "benchmarks/results/local.json"))


def elapsed_seconds(callable_, *arguments):
    """Synchronize a scalar JAX result and return value plus wall time."""

    start = time.perf_counter()
    value = callable_(*arguments)
    jax.tree.map(lambda leaf: leaf.block_until_ready(), value)
    return value, time.perf_counter() - start


def benchmark_resolution(nphi):
    """Measure one static toroidal resolution."""

    def transform(etabar):
        return qsc.solve(axis=AXIS, etabar=etabar, nphi=nphi).iota

    compiled = jax.jit(transform)
    jax.clear_caches()
    value, cold_seconds = elapsed_seconds(compiled, jnp.asarray(ETABAR))
    warm_seconds = [
        elapsed_seconds(compiled, jnp.asarray(ETABAR))[1] for _ in range(WARM_REPETITIONS)
    ]

    differentiated = jax.jit(lambda etabar: jax.jvp(transform, (etabar,), (jnp.ones_like(etabar),)))
    _, jvp_cold_seconds = elapsed_seconds(differentiated, jnp.asarray(ETABAR))
    jvp_warm_seconds = [
        elapsed_seconds(differentiated, jnp.asarray(ETABAR))[1] for _ in range(WARM_REPETITIONS)
    ]

    batched = jax.jit(jax.vmap(transform))
    batch_parameters = jnp.linspace(-1.0, -0.8, BATCH_SIZE)
    _, vmap_cold_seconds = elapsed_seconds(batched, batch_parameters)
    vmap_warm_seconds = [
        elapsed_seconds(batched, batch_parameters)[1] for _ in range(WARM_REPETITIONS)
    ]
    return {
        "nphi": nphi,
        "iota": float(value),
        "cold_compile_and_execute_seconds": cold_seconds,
        "warm_median_seconds": statistics.median(warm_seconds),
        "warm_samples_seconds": warm_seconds,
        "jvp_cold_compile_and_execute_seconds": jvp_cold_seconds,
        "jvp_warm_median_seconds": statistics.median(jvp_warm_seconds),
        "jvp_warm_samples_seconds": jvp_warm_seconds,
        "vmap_batch_size": BATCH_SIZE,
        "vmap_cold_compile_and_execute_seconds": vmap_cold_seconds,
        "vmap_warm_median_seconds": statistics.median(vmap_warm_seconds),
        "vmap_warm_samples_seconds": vmap_warm_seconds,
    }


results = []
print("Benchmarking first-order solve kernels...")
for nphi in RESOLUTIONS:
    row = benchmark_resolution(nphi)
    results.append(row)
    print(
        "nphi=",
        nphi,
        "cold=",
        f"{row['cold_compile_and_execute_seconds']:.3f}s",
        "warm=",
        f"{row['warm_median_seconds']:.6f}s",
        "JVP warm=",
        f"{row['jvp_warm_median_seconds']:.6f}s",
        "VMAP warm=",
        f"{row['vmap_warm_median_seconds']:.6f}s",
    )

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
    "jax_devices": [str(device) for device in jax.devices()],
    "jax_enable_x64": bool(jax.config.jax_enable_x64),
    "warm_repetitions": WARM_REPETITIONS,
    "results": results,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("saved:", OUTPUT)

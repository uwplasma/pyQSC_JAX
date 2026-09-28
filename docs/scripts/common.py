"""Shared helpers for the documentation figure scripts.

Figures are written to ``docs/_static/figures`` as palette-quantized PNGs. The
numbers quoted in the text are stored in ``docs/_static/figures/measurements.json``
and pulled into the pages as MyST substitutions, so text and figures always come
from one run of the current code. Run ``python docs/scripts/make_all.py`` with
``JAX_ENABLE_X64=1`` and the package importable.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIGURE_DIR = HERE.parent / "_static" / "figures"
MEASUREMENTS = FIGURE_DIR / "measurements.json"

# Okabe-Ito, colour-blind safe.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000"]
matplotlib.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": matplotlib.cycler(color=COLORS),
        "savefig.bbox": "tight",
    }
)


def compress(path: Path, colors: int = 256) -> None:
    """Quantize a PNG to an adaptive palette (visually lossless for line plots)."""

    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
            path, optimize=True
        )
    except Exception as error:  # Pillow missing
        print(f"    (compression skipped: {error})")


def savefig(figure, name: str, *, dpi: int = 110) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    compress(path)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} kB)")
    return path


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


def record(script: str, values: dict) -> None:
    """Merge ``values`` into measurements.json with provenance for ``script``."""

    data = json.loads(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}
    data.update(values)
    provenance = data.setdefault("_provenance", {})
    provenance[script] = {
        "commit": _commit(),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    MEASUREMENTS.parent.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS.write_text(json.dumps(dict(sorted(data.items())), indent=1) + "\n")


def sci(value: float, digits: int = 2) -> str:
    """Format a number as LaTeX-free scientific text, e.g. ``3.1e-14``."""

    return f"{float(value):.{digits}g}" if 1e-3 <= abs(float(value)) < 1e4 else f"{float(value):.{digits - 1}e}"

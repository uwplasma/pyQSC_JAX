"""Regenerate every documentation figure and measurements.json.

Run from anywhere with the package importable and 64-bit JAX:

    JAX_ENABLE_X64=1 python docs/scripts/make_all.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ("measurements.py", "fig_examples.py"):
    print(f"== {script}")
    subprocess.run((sys.executable, script), cwd=HERE, check=True)

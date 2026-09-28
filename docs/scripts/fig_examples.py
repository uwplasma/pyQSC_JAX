"""Run every ``examples/NN_*.py`` script and keep its figure and printed output.

Each script runs in a fresh process inside a temporary directory, exactly as a
user would run it. Its PNG is downscaled and palette-compressed into
``docs/_static/figures/example_NN.png`` and its standard output is stored in
``docs/examples/output/NN.txt`` for the example pages. Scripts that need the
network (00) or VMEX (14) are skipped with a message when that is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import FIGURE_DIR, ROOT, compress, record

OUTPUT_TEXT = ROOT / "docs" / "examples" / "output"
MAX_WIDTH = 1100

# plt.show() is replaced so that scripts which only show a figure still leave a PNG.
RUNNER = """
import runpy, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
target = Path(sys.argv[1])
def _show(*args, **kwargs):
    out = Path("examples/output"); out.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(out / (target.stem + ".png"), dpi=180, bbox_inches="tight")
plt.show = _show
sys.argv = [str(target)]
runpy.run_path(str(target), run_name="__main__")
"""


def run(script: Path) -> tuple[bool, str]:
    environment = os.environ.copy()
    environment.update({"JAX_ENABLE_X64": "1", "MPLBACKEND": "Agg", "PYQSC_RUN_VMEX": "1"})
    with tempfile.TemporaryDirectory() as directory:
        completed = subprocess.run(
            (sys.executable, "-c", RUNNER, str(script)),
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=540,
        )
        text = completed.stdout
        if completed.returncode != 0:
            return False, text + completed.stderr[-2000:]
        pngs = sorted((Path(directory) / "examples" / "output").glob("*.png"))
        if not pngs:
            return False, text
        from PIL import Image

        number = script.name[:2]
        target = FIGURE_DIR / f"example_{number}.png"
        image = Image.open(pngs[0]).convert("RGB")
        if image.width > MAX_WIDTH:
            height = round(image.height * MAX_WIDTH / image.width)
            image = image.resize((MAX_WIDTH, height), Image.Resampling.LANCZOS)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        image.save(target)
        compress(target)
        return True, text


def main() -> None:
    OUTPUT_TEXT.mkdir(parents=True, exist_ok=True)
    ran = []
    for script in sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")):
        number = script.name[:2]
        ok, text = run(script)
        # Drop the machine-specific "saved:" lines and absolute paths.
        lines = [line for line in text.splitlines() if not line.startswith("saved:")]
        if ok:
            (OUTPUT_TEXT / f"{number}.txt").write_text("\n".join(lines) + "\n")
            ran.append(number)
            print(f"{script.name}: ok")
        else:
            print(f"{script.name}: FAILED or skipped\n{text[-1500:]}")
    record("fig_examples.py", {"examples_run": ", ".join(ran)})


if __name__ == "__main__":
    main()

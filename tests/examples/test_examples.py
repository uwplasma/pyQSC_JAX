"""Every checked-in example must execute as a direct, headless script."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SCRIPTS = tuple(sorted((REPOSITORY_ROOT / "examples").glob("[0-9][0-9]_*.py")))
PUBLICATION_SCRIPTS = tuple(
    sorted((REPOSITORY_ROOT / "examples" / "publication").glob("figure_*.py"))
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "script",
    EXAMPLE_SCRIPTS + PUBLICATION_SCRIPTS,
    ids=lambda path: path.stem,
)
def test_example_executes_as_direct_script(script: Path, tmp_path: Path) -> None:
    """Execute the public script contract without relying on repository cwd."""

    environment = os.environ.copy()
    environment.update(
        {
            "JAX_ENABLE_X64": "true",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
        }
    )
    subprocess.run(
        (sys.executable, str(script)),
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=180,
    )

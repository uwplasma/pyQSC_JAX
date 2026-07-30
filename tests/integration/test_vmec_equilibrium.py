from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

import pyqsc_jax as qsc

REFERENCE_DIRECTORY = Path(__file__).resolve().parents[1] / "reference" / "vmec"


def _read_vmec_result(path: Path) -> dict[str, float | int]:
    with netcdf_file(path, "r", mmap=False) as dataset:
        return {
            "ier_flag": int(dataset.variables["ier_flag"].data),
            "iota_axis": float(-dataset.variables["iotaf"].data[0]),
            "fsqr": float(dataset.variables["fsqr"].data),
            "fsqz": float(dataset.variables["fsqz"].data),
            "fsql": float(dataset.variables["fsql"].data),
        }


@pytest.mark.integration
def test_frozen_vmec_wout_recovers_near_axis_iota():
    manifest = json.loads((REFERENCE_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    vmec_input = REFERENCE_DIRECTORY / "input.qa_r0025"
    wout = REFERENCE_DIRECTORY / "wout_qa_r0025.nc"
    input_digest = hashlib.sha256(vmec_input.read_bytes()).hexdigest()
    digest = hashlib.sha256(wout.read_bytes()).hexdigest()
    result = _read_vmec_result(wout)
    solution = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=121,
        order="r2",
    )
    relative_iota_error = abs(result["iota_axis"] - float(solution.iota)) / abs(
        float(solution.iota)
    )

    assert input_digest == manifest["files"]["input.qa_r0025"]
    assert digest == manifest["files"]["wout_qa_r0025.nc"]
    assert result["ier_flag"] == 0
    assert max(result["fsqr"], result["fsqz"], result["fsql"]) < 1.0e-9
    assert relative_iota_error < 1.0e-3
    np.testing.assert_allclose(
        result["iota_axis"],
        manifest["vmec_result"]["iota_axis"],
        rtol=0,
        atol=2.0e-15,
    )


@pytest.mark.integration
@pytest.mark.slow
def test_local_vmec_rerun_when_executable_is_requested(tmp_path):
    executable = os.environ.get("PYQSC_VMEC_EXECUTABLE")
    if not executable:
        pytest.skip("Set PYQSC_VMEC_EXECUTABLE to rerun the fixed-boundary VMEC case.")
    solution = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=121,
        order="r2",
    )
    qsc.to_vmec(
        solution,
        tmp_path / "input.qa_r0025",
        r=0.0025,
        ntheta=32,
        mpol=6,
        ntor=6,
        parameters={
            "ns_array": (31,),
            "ftol_array": (1.0e-10,),
            "niter_array": (3000,),
        },
    )
    subprocess.run(
        [executable, "input.qa_r0025"],
        cwd=tmp_path,
        check=True,
        timeout=120,
    )
    result = _read_vmec_result(tmp_path / "wout_qa_r0025.nc")

    assert result["ier_flag"] == 0
    assert max(result["fsqr"], result["fsqz"], result["fsql"]) < 1.0e-9
    np.testing.assert_allclose(result["iota_axis"], solution.iota, rtol=1.0e-3)

from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import jax
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.near_axis import near_axis


def qa_solution(*, nphi: int = 61):
    return qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=nphi,
        order="r2",
    )


def test_vectorized_surface_matches_independent_legacy_root_solve():
    field = near_axis(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=31,
        order="r2",
    )
    radius = 0.01
    old_R, old_Z, old_phi0 = field.Frenet_to_cylindrical(radius, ntheta=8)
    new_R, new_Z, new_phi0, residual = qsc.uniform_cylindrical_surface(
        field.solution,
        radius,
        ntheta=8,
    )

    np.testing.assert_allclose(new_R, old_R, atol=4.0e-13)
    np.testing.assert_allclose(new_Z, old_Z, atol=4.0e-13)
    np.testing.assert_allclose(new_phi0, old_phi0, atol=4.0e-13)
    assert float(residual) < 2.0e-15


def test_vmec_boundary_is_accurate_and_fast_after_compilation(tmp_path):
    solution = qa_solution()
    arguments = {"ntheta": 40, "mpol": 12, "ntor": 14}
    compiled = qsc.vmec_boundary(solution, 0.03, **arguments)
    jax.block_until_ready(compiled.RBC)

    start = perf_counter()
    boundary = qsc.vmec_boundary(solution, 0.03, **arguments)
    jax.block_until_ready(boundary.RBC)
    warm_seconds = perf_counter() - start
    qsc.to_vmec(solution, tmp_path / "input.first", r=0.03, **arguments)
    start = perf_counter()
    qsc.to_vmec(solution, tmp_path / "input.warm", r=0.03, **arguments)
    total_warm_seconds = perf_counter() - start

    assert float(boundary.maximum_toroidal_angle_residual) < 2.0e-15
    assert float(boundary.maximum_R_reconstruction_error) < 5.0e-6
    assert float(boundary.maximum_Z_reconstruction_error) < 5.0e-6
    assert warm_seconds < 0.5
    assert total_warm_seconds < 0.5


def test_to_vmec_is_deterministic_and_legacy_adapter_exposes_coefficients(tmp_path):
    solution = qa_solution(nphi=121)
    output = tmp_path / "input.qa_r0025"
    repeated_output = tmp_path / "input.qa_r0025.repeated"
    export = qsc.to_vmec(
        solution,
        output,
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
    qsc.to_vmec(
        solution,
        repeated_output,
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

    assert sha256(output.read_bytes()).digest() == sha256(repeated_output.read_bytes()).digest()
    assert export.phiedge == np.pi * 0.0025**2
    assert not export.lasym
    assert export.boundary.RBC.shape == (13, 7)
    assert "&INDATA" in output.read_text(encoding="utf-8")

    field = near_axis(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=31,
        order="r2",
    )
    legacy_export = field.to_vmec(
        tmp_path / "input.legacy",
        r=0.01,
        params={"mpol": 4, "ntor": 4},
        ntheta=16,
        ntorMax=4,
    )
    assert legacy_export.path.exists()
    assert field.RBC.shape == field.RBS.shape == field.ZBC.shape == field.ZBS.shape == (5, 9)


def test_asymmetric_axis_emits_all_four_vmec_coefficient_families(tmp_path):
    solution = qsc.Qsc(
        rc=[1.0, 0.04],
        rs=[0.0, 0.005],
        zc=[0.0, 0.01],
        zs=[0.0, -0.04],
        nfp=3,
        etabar=-0.9,
        nphi=31,
        order="r2",
    )
    export = qsc.to_vmec(
        solution,
        tmp_path / "input.asymmetric",
        r=0.01,
        ntheta=16,
        mpol=4,
        ntor=4,
    )

    assert export.lasym
    assert float(np.max(np.abs(export.boundary.RBS))) > 1.0e-4
    assert float(np.max(np.abs(export.boundary.ZBC))) > 1.0e-4
    contents = export.path.read_text(encoding="utf-8")
    assert "LASYM = T" in contents
    assert "RBS(" in contents
    assert "ZBC(" in contents


def test_first_and_third_order_surfaces_and_ntor_cap(tmp_path):
    first_order = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=15,
        order="r1",
    )
    first_boundary = qsc.vmec_boundary(
        first_order,
        0.005,
        ntheta=8,
        mpol=3,
        ntor=2,
    )
    assert np.all(np.isfinite(first_boundary.R))

    third_order = qsc.solve_configuration("qa", nphi=15, order="r3")
    export = qsc.to_vmec(
        third_order,
        tmp_path / "input.r3",
        r=0.005,
        ntheta=10,
        mpol=4,
        ntor=4,
        ntor_max=2,
        parameters=qsc.VmecInputParameters(
            ns_array=(15,),
            ftol_array=(1.0e-10,),
            niter_array=(2000,),
        ),
    )
    assert export.boundary.ntor == 2
    assert np.all(np.isfinite(export.boundary.Z))


def test_finite_pressure_current_profiles_are_written(tmp_path):
    solution = qsc.solve_configuration("plasma_dominant_channel", nphi=15)
    export = qsc.to_vmec(
        solution,
        tmp_path / "input.plasma",
        r=0.1,
        ntheta=10,
        mpol=4,
        ntor=2,
    )
    contents = export.path.read_text(encoding="utf-8")

    np.testing.assert_allclose(export.curtor, 210000.0)
    np.testing.assert_allclose(export.pressure_axis, 1000.0)
    assert "AM =" in contents
    assert "CURTOR =" in contents


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"ns_array": (), "ftol_array": (), "niter_array": ()}, "nonempty"),
        (
            {
                "ns_array": (15,),
                "ftol_array": (1.0e-10, 1.0e-11),
                "niter_array": (2000,),
            },
            "aligned",
        ),
        ({"delt": 0.0}, "positive"),
        ({"nstep": 0}, "positive"),
        ({"tcon0": 0.0}, "positive"),
        (
            {"ns_array": (2,), "ftol_array": (1.0e-10,), "niter_array": (2000,)},
            "at least 3",
        ),
        (
            {"ns_array": (15,), "ftol_array": (0.0,), "niter_array": (2000,)},
            "tolerance",
        ),
        (
            {"ns_array": (15,), "ftol_array": (1.0e-10,), "niter_array": (0,)},
            "iteration",
        ),
    ],
)
def test_vmec_input_parameter_guards(kwargs, message):
    with pytest.raises(ValueError, match=message):
        qsc.VmecInputParameters(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"r": 0.0}, "r must be positive"),
        ({"ntor_max": -1}, "ntor_max"),
        ({"coefficient_tolerance": -1.0}, "coefficient_tolerance"),
        ({"ntheta": 7, "mpol": 3}, "ntheta"),
        ({"mpol": 0}, "positive"),
        ({"ntor": -1}, "nonnegative"),
        ({"newton_iterations": 0}, "positive"),
        ({"ntheta": 8.0}, "integers"),
        ({"ntor": 8}, "nphi"),
    ],
)
def test_to_vmec_resolution_and_scalar_guards(tmp_path, kwargs, message):
    options = {
        "r": 0.01,
        "ntheta": 8,
        "mpol": 3,
        "ntor": 2,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        qsc.to_vmec(
            qa_solution(nphi=15),
            tmp_path / "input.invalid",
            **options,
        )


def test_to_vmec_rejects_unknown_control(tmp_path):
    with pytest.raises(ValueError, match="Unknown VMEC input"):
        qsc.to_vmec(
            qa_solution(nphi=15),
            tmp_path / "input.invalid",
            r=0.01,
            ntheta=8,
            mpol=3,
            ntor=2,
            parameters={"missing": 1},
        )

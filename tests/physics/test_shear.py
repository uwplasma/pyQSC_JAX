from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.near_axis import near_axis

PAPER_CASES = (
    (
        {
            "rc": [1.0, 0.155, 0.0102],
            "zs": [0.0, 0.154, 0.0111],
            "nfp": 2,
            "etabar": 0.64,
            "B2c": -0.00322,
        },
        4.665537859250996,
    ),
    (
        {
            "rc": [1.0, 0.09],
            "zs": [0.0, -0.09],
            "nfp": 2,
            "etabar": 0.95,
            "I2": 0.9,
            "B2c": -0.7,
            "p2": -600000.0,
        },
        0.2758333187134502,
    ),
    (
        {
            "rc": [1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
            "zs": [0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
            "nfp": 4,
            "etabar": 1.569,
            "B2c": 0.1348,
        },
        -1.1837987800965397,
    ),
)


def qa_solution(nphi=61, **overrides):
    parameters = {
        "rc": [1.0, 0.155, 0.0102],
        "zs": [0.0, 0.154, 0.0111],
        "nfp": 2,
        "etabar": 0.64,
        "B2c": -0.00322,
        "order": "r3",
        "nphi": nphi,
    }
    parameters.update(overrides)
    return qsc.Qsc(**parameters)


@pytest.mark.parametrize(("parameters", "expected_iota2"), PAPER_CASES)
def test_magnetic_shear_matches_upstream_paper_cases(parameters, expected_iota2):
    solution = qsc.solve_magnetic_shear(qsc.Qsc(**parameters, order="r3", nphi=61))

    np.testing.assert_allclose(solution.iota2, expected_iota2, rtol=2.0e-12, atol=2.0e-12)
    assert bool(solution.stellarator_symmetric)
    np.testing.assert_allclose(
        solution.iota2,
        solution.inputs.B0 * solution.numerator / (2 * solution.denominator),
    )
    assert np.all(np.isfinite(np.asarray(solution.Lambda_tilde)))
    assert np.all(np.isfinite(np.asarray(solution.integrating_factor)))


def test_general_asymmetric_integration_matches_upstream():
    solution = qsc.Qsc(
        rc=[1.0, 0.1],
        rs=[0.0, 0.01],
        zc=[0.0, -0.005],
        zs=[0.0, 0.1],
        nfp=2,
        etabar=1.0,
        sigma0=0.2,
        I2=1.0,
        order="r3",
        nphi=61,
    )
    solution = qsc.solve_magnetic_shear(solution)

    np.testing.assert_allclose(solution.iota2, -2131.8417260171145, rtol=3.0e-12)
    assert not bool(solution.stellarator_symmetric)
    assert abs(float(solution.sigma_average)) > 0.1


def test_B31c_response_and_resolution_convergence():
    coarse = qsc.solve_magnetic_shear(qa_solution(nphi=31), B31c=0.23)
    medium = qsc.solve_magnetic_shear(qa_solution(nphi=51), B31c=0.23)
    fine = qsc.solve_magnetic_shear(qa_solution(nphi=91), B31c=0.23)

    np.testing.assert_allclose(medium.iota2, fine.iota2, rtol=5.0e-11)
    np.testing.assert_allclose(fine.iota2, 4.676607885614116, rtol=3.0e-12)
    assert abs(float(coarse.iota2 - fine.iota2)) < 1.2e-5
    np.testing.assert_allclose(fine.B31c, 0.23)


def test_magnetic_shear_is_jittable_and_differentiable():
    solution = qa_solution(nphi=31)

    def shear_for_B31c(B31c):
        return qsc.solve_magnetic_shear(solution, B31c=B31c).iota2

    value = shear_for_B31c(jnp.asarray(0.1))
    jitted = jax.jit(shear_for_B31c)(jnp.asarray(0.1))
    tangent = jax.jvp(shear_for_B31c, (jnp.asarray(0.1),), (jnp.asarray(1.0),))[1]
    cotangent = jax.vjp(shear_for_B31c, jnp.asarray(0.1))[1](jnp.asarray(1.0))[0]
    step = 1.0e-4
    finite_difference = (shear_for_B31c(0.1 + step) - shear_for_B31c(0.1 - step)) / (2 * step)

    np.testing.assert_allclose(jitted, value, rtol=2.0e-13)
    np.testing.assert_allclose(tangent, cotangent, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-8, atol=2.0e-10)


def test_magnetic_shear_differentiates_through_near_axis_solve():
    def shear_for_etabar(etabar):
        solution = qa_solution(nphi=31, etabar=etabar)
        return qsc.solve_magnetic_shear(solution).iota2

    etabar = jnp.asarray(0.64)
    tangent = jax.jvp(shear_for_etabar, (etabar,), (jnp.asarray(1.0),))[1]
    step = 1.0e-5
    finite_difference = (shear_for_etabar(0.64 + step) - shear_for_etabar(0.64 - step)) / (2 * step)

    np.testing.assert_allclose(tangent, finite_difference, rtol=8.0e-8, atol=5.0e-8)


def test_legacy_calculate_shear_and_stale_value_removal():
    adapter = near_axis(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=0.64,
        B2c=-0.00322,
        order="r3",
        nphi=61,
    )
    assert adapter.calculate_shear() is None
    np.testing.assert_allclose(adapter.iota2, 4.665537859250996, rtol=2.0e-12)
    adapter.dofs = adapter.dofs.at[-1].set(0.65)
    assert not hasattr(adapter, "iota2")


def test_magnetic_shear_input_guards_and_missing_result():
    with pytest.raises(ValueError, match="second-order"):
        qsc.solve_magnetic_shear(qa_solution(order="r1"))
    with pytest.raises(ValueError, match="scalar"):
        qsc.solve_magnetic_shear(qa_solution(), B31c=jnp.ones(2))
    with pytest.raises(NotImplementedError, match="sG=spsi=1"):
        qsc.solve_magnetic_shear(qa_solution(sG=-1))
    with pytest.raises(AttributeError, match="not been calculated"):
        _ = qa_solution().iota2

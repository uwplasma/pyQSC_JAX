import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.near_axis import near_axis

CASES = [
    (
        {
            "rc": [1.0, 0.155, 0.0102],
            "zs": [0.0, 0.154, 0.0111],
            "nfp": 2,
            "etabar": 0.64,
            "B2c": -0.00322,
        },
        0.2257896241404959,
        [0.7304158315686611, 0.2505553226671735, 0.2935268282653716, 0.23909537193609554],
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
        0.22161114270880408,
        [0.7427937841922576, 0.3271954156702964, 0.2659539057552679, 0.3034006922051161],
    ),
    (
        {
            "rc": [1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
            "zs": [0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
            "nfp": 4,
            "etabar": 1.569,
            "B2c": 0.1348,
        },
        0.3583276532138262,
        [0.8681431131590702, 0.4034129514765815, 0.3728231585587465, 0.39141407123791455],
    ),
]


@pytest.mark.parametrize("parameters, expected_minimum, expected_samples", CASES)
def test_singular_radius_matches_upstream_pyqsc(
    parameters,
    expected_minimum,
    expected_samples,
):
    solution = qsc.Qsc(**parameters, nphi=61, order="r2")

    np.testing.assert_allclose(solution.r_singularity, expected_minimum, rtol=0, atol=5e-8)
    np.testing.assert_allclose(
        np.asarray(solution.r_singularity_vs_varphi)[[0, 15, 30, 45]],
        expected_samples,
        rtol=0,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        solution.inv_r_singularity_vs_varphi,
        1 / solution.r_singularity_vs_varphi,
    )
    np.testing.assert_allclose(
        solution.r_singularity_basic_vs_varphi,
        solution.r_singularity_vs_varphi,
    )


def test_determinant_coefficients_and_refined_residual():
    solution = qsc.Qsc(**CASES[0][0], nphi=61, order="r2")
    diagnostics = solution.singularity
    expected_g0 = solution.geometry.abs_G0_over_B0 * solution.X1c * solution.Y1s

    np.testing.assert_allclose(diagnostics.g0, expected_g0, rtol=2e-13, atol=2e-13)
    assert jnp.max(jnp.abs(diagnostics.g1s)) < 2e-13
    assert diagnostics.maximum_residual_norm < 2e-13
    assert diagnostics.angular_resolution == 256
    assert diagnostics.newton_iterations == 8

    radius = diagnostics.r_singularity_vs_varphi
    theta = diagnostics.theta_singularity_vs_varphi
    linear = diagnostics.g1c * jnp.cos(theta) + diagnostics.g1s * jnp.sin(theta)
    linear_prime = -diagnostics.g1c * jnp.sin(theta) + diagnostics.g1s * jnp.cos(theta)
    quadratic = (
        diagnostics.g20
        + diagnostics.g2s * jnp.sin(2 * theta)
        + diagnostics.g2c * jnp.cos(2 * theta)
    )
    quadratic_prime = 2 * diagnostics.g2s * jnp.cos(2 * theta) - 2 * diagnostics.g2c * jnp.sin(
        2 * theta
    )
    np.testing.assert_allclose(
        diagnostics.g0 + radius * linear + radius**2 * quadratic,
        0,
        rtol=0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        radius * linear_prime + radius**2 * quadratic_prime,
        0,
        rtol=0,
        atol=2e-13,
    )


def test_newton_refinement_removes_angular_grid_dependence():
    solution = qsc.Qsc(**CASES[0][0], nphi=31, order="r2")
    coarse = qsc.singularity_diagnostics(
        solution,
        angular_resolution=32,
        newton_iterations=8,
    )
    fine = qsc.singularity_diagnostics(
        solution,
        angular_resolution=512,
        newton_iterations=8,
    )

    np.testing.assert_allclose(coarse.r_singularity, fine.r_singularity, rtol=0, atol=2e-11)
    assert coarse.maximum_residual_norm < 3e-13


def test_singular_radius_supports_jit_and_jvp():
    def radius(etabar):
        return qsc.Qsc(
            rc=[1.0, 0.155, 0.0102],
            zs=[0.0, 0.154, 0.0111],
            nfp=2,
            etabar=etabar,
            B2c=-0.00322,
            nphi=31,
            order="r2",
        ).r_singularity

    eager = radius(0.64)
    compiled = jax.jit(radius)(0.64)
    _, tangent = jax.jvp(radius, (0.64,), (1.0,))
    step = 1e-5
    finite_difference = (radius(0.64 + step) - radius(0.64 - step)) / (2 * step)

    np.testing.assert_allclose(compiled, eager, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(tangent, finite_difference, rtol=3e-7, atol=3e-8)


def test_singularity_validation_and_legacy_adapter():
    first_order = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=31,
    )
    with pytest.raises(ValueError, match="second-order"):
        qsc.singularity_diagnostics(first_order)
    with pytest.raises(ValueError, match="angular_resolution"):
        qsc.singularity_diagnostics(
            qsc.Qsc(**CASES[0][0], nphi=15, order="r2"),
            angular_resolution=4,
        )
    with pytest.raises(ValueError, match="newton_iterations"):
        qsc.singularity_diagnostics(
            qsc.Qsc(**CASES[0][0], nphi=15, order="r2"),
            newton_iterations=-1,
        )
    with pytest.raises(AttributeError, match="singular-radius"):
        _ = first_order.r_singularity

    legacy = near_axis(**CASES[0][0], nphi=31, order="r2")
    np.testing.assert_allclose(legacy.r_singularity, legacy.solution.r_singularity)
    np.testing.assert_allclose(
        legacy.r_singularity_residual_sqnorm,
        legacy.solution.r_singularity_residual_sqnorm,
    )

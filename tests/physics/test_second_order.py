import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.second_order import MU0


def vacuum_qa(nphi=31, **kwargs):
    parameters = {
        "rc": [1.0, 0.155, 0.0102],
        "zs": [0.0, 0.154, 0.0111],
        "nfp": 2,
        "etabar": 0.64,
        "B2c": -0.00322,
        "nphi": nphi,
        "order": "r2",
    }
    parameters.update(kwargs)
    return qsc.Qsc(**parameters)


def test_complete_second_order_equations_and_linear_report():
    solution = vacuum_qa()
    residuals = qsc.second_order_residuals(solution)

    assert solution.second_order is not None
    assert bool(solution.linear_report.converged)
    assert bool(solution.linear_report.finite)
    assert bool(solution.linear_report.well_conditioned)
    assert solution.linear_report.relative_residual_norm < 3e-14
    assert residuals.maximum_absolute < 1e-11
    for residual in (
        residuals.force_balance_1,
        residuals.force_balance_2,
        residuals.area_constraint_1,
        residuals.area_constraint_2,
    ):
        assert residual.shape == (solution.inputs.nphi,)


def test_second_order_definitions_and_derivatives():
    solution = vacuum_qa()
    weights = solution.geometry.d_l_d_phi
    weighted_mean = jnp.sum(solution.B20 * weights) / jnp.sum(weights)

    np.testing.assert_allclose(solution.B20_mean, weighted_mean, rtol=2e-13)
    np.testing.assert_allclose(solution.B20_anomaly, solution.B20 - weighted_mean)
    np.testing.assert_allclose(
        solution.B20_variation,
        jnp.max(solution.B20) - jnp.min(solution.B20),
    )
    np.testing.assert_allclose(
        solution.d_X20_d_varphi,
        solution.geometry.d_d_varphi @ solution.X20,
    )
    np.testing.assert_allclose(
        solution.d2_X1c_d_varphi2,
        solution.geometry.d_d_varphi @ (solution.geometry.d_d_varphi @ solution.X1c),
    )


def test_finite_pressure_and_current_relations():
    solution = qsc.Qsc(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=0.9,
        B2c=-0.7,
        p2=-600000.0,
        nphi=31,
        order="r2",
    )
    expected_beta_1s = (
        -4
        * solution.inputs.sG
        * solution.inputs.spsi
        * MU0
        * solution.inputs.p2
        * solution.inputs.etabar
        * solution.geometry.abs_G0_over_B0
        / (solution.iotaN * solution.inputs.B0**2)
    )
    expected_G2 = (
        -MU0 * solution.inputs.p2 * solution.G0 / solution.inputs.B0**2
        - solution.iota * solution.inputs.I2
    )

    np.testing.assert_allclose(solution.beta_1s, expected_beta_1s)
    np.testing.assert_allclose(solution.G2, expected_G2)
    assert qsc.second_order_residuals(solution).maximum_absolute < 1e-11


def test_qh_untwisting_preserves_harmonic_norm():
    solution = qsc.Qsc(
        rc=[1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
        zs=[0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
        nfp=4,
        etabar=1.569,
        B2c=0.1348,
        nphi=31,
        order="r2",
    )

    assert int(solution.helicity) == 1
    np.testing.assert_allclose(solution.X20_untwisted, solution.X20)
    np.testing.assert_allclose(
        solution.X2s_untwisted**2 + solution.X2c_untwisted**2,
        solution.X2s**2 + solution.X2c**2,
        rtol=2e-13,
        atol=2e-13,
    )
    assert qsc.second_order_residuals(solution).maximum_absolute < 2e-11


def test_second_order_residual_requires_r2_solution():
    first_order = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=15,
    )

    assert not hasattr(first_order, "X20")
    with pytest.raises(AttributeError, match="no 'X20'"):
        _ = first_order.X20
    with pytest.raises(ValueError, match="second-order"):
        qsc.second_order_residuals(first_order)


def test_second_order_resolution_convergence():
    medium = vacuum_qa(nphi=31)
    fine = vacuum_qa(nphi=61)

    np.testing.assert_allclose(medium.B20_mean, fine.B20_mean, rtol=6e-7)
    np.testing.assert_allclose(medium.B20_residual, fine.B20_residual, rtol=2e-7)

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc


def qa_solution(nphi=31, order="r2", **overrides):
    parameters = {
        "rc": [1.0, 0.155, 0.0102],
        "zs": [0.0, 0.154, 0.0111],
        "nfp": 2,
        "etabar": 0.64,
        "B2c": -0.00322,
        "nphi": nphi,
        "order": order,
    }
    parameters.update(overrides)
    return qsc.Qsc(**parameters)


def test_dense_B20_diagnostics_match_direct_definitions():
    solution = qa_solution()
    diagnostics = qsc.b20_diagnostics(solution, smooth_maximum_power=12)
    weights = solution.geometry.d_l_d_phi
    mean = np.sum(np.asarray(weights * solution.B20)) / np.sum(np.asarray(weights))
    anomaly = np.asarray(solution.B20) - mean

    np.testing.assert_allclose(diagnostics.weighted_mean, mean)
    np.testing.assert_allclose(diagnostics.anomaly, anomaly)
    np.testing.assert_allclose(
        diagnostics.weighted_l2,
        np.sqrt(
            np.sum(np.asarray(weights) * (anomaly / solution.inputs.B0) ** 2) / np.sum(weights)
        ),
    )
    np.testing.assert_allclose(diagnostics.weighted_l2, solution.B20_residual)
    np.testing.assert_allclose(
        diagnostics.grid_maximum,
        np.max(np.abs(anomaly / solution.inputs.B0)),
    )
    np.testing.assert_allclose(
        diagnostics.peak_to_peak,
        solution.B20_variation / solution.inputs.B0,
    )
    assert diagnostics.smooth_maximum_power == 12
    assert diagnostics.fourier_modes.shape == (15,)
    assert diagnostics.fourier_coefficients.shape == (15,)
    assert diagnostics.weighted_l2 <= diagnostics.smooth_maximum <= diagnostics.grid_maximum
    assert 0 <= float(diagnostics.fourier_tail_ratio) <= 1


def test_affine_B2c_elimination_is_exact_and_stationary():
    solution = qa_solution()
    result = qsc.optimize_B2c(solution)
    minus = qa_solution(B2c=result.B2c_optimal - 0.1)
    plus = qa_solution(B2c=result.B2c_optimal + 0.1)

    np.testing.assert_allclose(result.B2c_optimal, -0.49172683641534204, rtol=3.0e-12)
    assert result.affine_reconstruction_error < 5.0e-14
    assert not bool(result.degenerate)
    assert result.diagnostics.weighted_l2 < solution.B20_residual
    assert result.diagnostics.weighted_l2 < minus.B20_residual
    assert result.diagnostics.weighted_l2 < plus.B20_residual

    def squared_residual(B2c):
        return qa_solution(nphi=15, B2c=B2c).B20_residual ** 2

    optimum = qsc.optimal_B2c_value(qa_solution(nphi=15))
    derivative = jax.grad(squared_residual)(optimum)
    assert abs(float(derivative)) < 2.0e-11


def test_B2c_optimum_is_jittable_and_differentiable():
    def optimum(etabar):
        return qsc.optimal_B2c_value(qa_solution(nphi=15, etabar=etabar))

    value = optimum(jnp.asarray(0.64))
    jitted = jax.jit(optimum)(jnp.asarray(0.64))
    tangent = jax.jvp(optimum, (jnp.asarray(0.64),), (jnp.asarray(1.0),))[1]
    step = 1.0e-5
    finite_difference = (optimum(0.64 + step) - optimum(0.64 - step)) / (2 * step)

    np.testing.assert_allclose(jitted, value, rtol=2.0e-11)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-6)


def test_optimal_solution_recomputes_r3_and_shear_without_stale_data():
    original = qsc.solve_magnetic_shear(qa_solution(nphi=15, order="r3"))
    result = qsc.optimize_B2c(original)

    assert result.solution.third_order is not None
    assert result.solution.shear is not None
    np.testing.assert_allclose(result.solution.B31c, original.B31c)
    assert result.solution.flux_constraint_residual < 2.0e-13
    assert result.solution.consistency_error < 2.0e-10


def test_circular_axis_has_degenerate_nonconstant_B2c_response():
    solution = qsc.Qsc(
        rc=[1.0],
        zs=[0.0],
        nfp=1,
        etabar=1.0,
        I2=0.1,
        B2c=0.2,
        nphi=15,
        order="r2",
    )
    result = qsc.optimize_B2c(solution)

    assert bool(result.degenerate)
    np.testing.assert_allclose(result.B2c_optimal, 0.2)
    assert result.diagnostics.weighted_l2 < 2.0e-14


def test_resolution_verification_recomputes_fixed_candidate():
    result = qsc.optimize_B2c(qa_solution())
    verification = qsc.verify_B20_resolution(result.solution, multipliers=(1, 2))

    np.testing.assert_array_equal(verification.resolutions, [31, 61])
    np.testing.assert_allclose(verification.weighted_l2[0], result.diagnostics.weighted_l2)
    assert verification.relative_weighted_l2_change[1] < 2.0e-6
    assert verification.fourier_tail_ratio[1] < verification.fourier_tail_ratio[0]
    assert verification.relative_grid_maximum_change[1] < 0.03


def test_B20_optimization_input_guards():
    first_order = qa_solution(order="r1")
    with pytest.raises(ValueError, match="second-order"):
        qsc.b20_diagnostics(first_order)
    with pytest.raises(ValueError, match="second-order"):
        qsc.optimal_B2c_value(first_order)
    with pytest.raises(ValueError, match="smooth_maximum_power"):
        qsc.b20_diagnostics(qa_solution(), smooth_maximum_power=1)
    with pytest.raises(ValueError, match="degeneracy_tolerance"):
        qsc.optimal_B2c_value(qa_solution(), degeneracy_tolerance=-1.0)
    with pytest.raises(ValueError, match="multipliers"):
        qsc.verify_B20_resolution(qa_solution(), multipliers=())
    with pytest.raises(ValueError, match="multipliers"):
        qsc.verify_B20_resolution(qa_solution(), multipliers=(True,))

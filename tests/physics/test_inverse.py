from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)


def test_target_iota_etabar_round_trip_and_sign_branches():
    forward = qsc.solve(axis=AXIS, etabar=-0.9, nphi=31)
    negative = qsc.solve(
        axis=AXIS,
        etabar=-1.0,
        iota=forward.iota,
        solve_for="etabar",
        nphi=31,
    )
    positive = qsc.solve(
        axis=AXIS,
        etabar=1.0,
        iota=forward.iota,
        solve_for="etabar",
        nphi=31,
    )

    np.testing.assert_allclose(negative.inputs.etabar, -0.9, rtol=2.0e-12)
    np.testing.assert_allclose(positive.inputs.etabar, 0.9, rtol=2.0e-12)
    np.testing.assert_allclose(negative.iota, forward.iota)
    np.testing.assert_allclose(negative.sigma, forward.sigma, atol=2.0e-12)
    assert bool(negative.root_report.converged)
    assert negative.root_report.residual_norm < 2.0e-13
    assert negative.inverse is not None
    assert negative.parameter == "etabar"
    assert int(negative.parameter_sign) == -1
    assert not bool(negative.branch_fold)


def test_target_iota_exposes_distinct_local_etabar_branches():
    target = qsc.solve(axis=AXIS, etabar=-0.9, nphi=31).iota
    outer = qsc.solve(
        axis=AXIS,
        etabar=-1.0,
        iota=target,
        solve_for="etabar",
        nphi=31,
    )
    inner = qsc.solve(
        axis=AXIS,
        etabar=-0.8,
        iota=target,
        solve_for="etabar",
        nphi=31,
    )
    inner_forward = qsc.solve(axis=AXIS, etabar=inner.inputs.etabar, nphi=31)

    assert abs(float(outer.inputs.etabar - inner.inputs.etabar)) > 0.1
    np.testing.assert_allclose(inner.iota, target)
    np.testing.assert_allclose(inner_forward.iota, target, atol=2.0e-13)
    assert np.sign(float(outer.response_derivative)) != np.sign(float(inner.response_derivative))


@pytest.mark.parametrize("actual_I2", [-0.5, 0.2, 1.0])
def test_target_iota_I2_round_trip(actual_I2):
    forward = qsc.solve(axis=AXIS, etabar=-0.9, I2=actual_I2, nphi=31)
    inverse = qsc.solve(
        axis=AXIS,
        etabar=-0.9,
        I2=0.0,
        iota=forward.iota,
        solve_for="I2",
        nphi=31,
    )

    np.testing.assert_allclose(inverse.inputs.I2, actual_I2, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(inverse.iota, forward.iota)
    assert inverse.parameter == "I2"
    assert bool(inverse.root_report.converged)


def test_inverse_mode_propagates_to_second_order():
    forward = qsc.solve(axis=AXIS, etabar=-0.9, I2=0.2, nphi=31, order="r2")
    inverse = qsc.solve(
        axis=AXIS,
        etabar=-1.0,
        I2=0.2,
        iota=forward.iota,
        solve_for="etabar",
        nphi=31,
        order="r2",
    )

    np.testing.assert_allclose(inverse.inputs.etabar, -0.9, rtol=2.0e-12)
    np.testing.assert_allclose(inverse.B20, forward.B20, rtol=3.0e-11, atol=3.0e-11)
    assert bool(inverse.linear_report.converged)


def test_inverse_solve_is_jittable_and_implicitly_differentiable():
    def etabar_for_iota(target_iota):
        return qsc.solve(
            axis=AXIS,
            etabar=-1.0,
            iota=target_iota,
            solve_for="etabar",
            nphi=31,
        ).inputs.etabar

    target = qsc.solve(axis=AXIS, etabar=-0.9, nphi=31).iota
    solution = qsc.solve(
        axis=AXIS,
        etabar=-1.0,
        iota=target,
        solve_for="etabar",
        nphi=31,
    )
    value = etabar_for_iota(target)
    jitted = jax.jit(etabar_for_iota)(target)
    tangent = jax.jvp(etabar_for_iota, (target,), (jnp.asarray(1.0),))[1]
    step = 1.0e-5
    finite_difference = (etabar_for_iota(target + step) - etabar_for_iota(target - step)) / (
        2 * step
    )

    np.testing.assert_allclose(jitted, value, rtol=2.0e-12)
    np.testing.assert_allclose(tangent, 1 / solution.response_derivative, rtol=2.0e-11)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-6)


def test_inverse_input_guards_and_forward_diagnostic_absence():
    forward = qsc.solve(axis=AXIS, etabar=-0.9, nphi=31)
    with pytest.raises(AttributeError, match="Forward solution"):
        _ = forward.target_iota
    with pytest.raises(ValueError, match="etabar is required"):
        qsc.solve(axis=AXIS)
    with pytest.raises(ValueError, match="prescribed only"):
        qsc.solve(axis=AXIS, etabar=-0.9, iota=0.4)
    with pytest.raises(ValueError, match="iota is required"):
        qsc.solve(axis=AXIS, etabar=-0.9, solve_for="I2")
    with pytest.raises(ValueError, match="etabar is required"):
        qsc.solve(axis=AXIS, iota=0.4, solve_for="I2")
    with pytest.raises(ValueError, match="scalar"):
        qsc.solve(
            axis=AXIS,
            etabar=-1.0,
            iota=jnp.ones(2),
            solve_for="etabar",
        )
    with pytest.raises(ValueError, match="fold_tolerance"):
        qsc.solve(
            axis=AXIS,
            etabar=-1.0,
            iota=0.4,
            solve_for="etabar",
            fold_tolerance=-1.0,
        )

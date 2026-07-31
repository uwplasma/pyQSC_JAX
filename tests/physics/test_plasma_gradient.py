from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc


def finite_current_solution(*, I2=0.9, p2=-600000.0, nphi=61):
    return qsc.Qsc(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=I2,
        p2=p2,
        B2c=-0.7,
        nphi=nphi,
        order="r2",
    )


def test_straight_circular_channel_gradient():
    current_density_mu0 = 1.7
    gradient = qsc.elliptical_channel_gradient(
        1.0,
        0.0,
        parallel_current_mu0=current_density_mu0,
        chi=1,
    )
    expected = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -current_density_mu0 / 2],
            [0.0, current_density_mu0 / 2, 0.0],
        ]
    )

    np.testing.assert_allclose(gradient, expected)
    np.testing.assert_allclose(np.trace(gradient), 0.0)
    np.testing.assert_allclose(
        gradient[2, 1] - gradient[1, 2],
        current_density_mu0,
    )


def test_straight_sheared_elliptical_channel_obeys_ampere_and_divergence():
    current_density_mu0 = -0.8
    gradient = qsc.elliptical_channel_gradient(
        1.6,
        -0.35,
        parallel_current_mu0=current_density_mu0,
        chi=-1,
    )

    np.testing.assert_allclose(np.trace(gradient), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        gradient[2, 1] - gradient[1, 2],
        current_density_mu0,
    )
    assert gradient[1, 1] == -gradient[2, 2]
    assert not np.isclose(gradient[2, 1], -gradient[1, 2])


def test_gradient_rotates_covariantly_from_frenet_to_cartesian():
    angle = 0.37
    rotation = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), np.sin(angle)],
            [0.0, -np.sin(angle), np.cos(angle)],
        ]
    )
    frenet = qsc.elliptical_channel_gradient(
        1.3,
        0.2,
        parallel_current_mu0=0.7,
        chi=1,
    )
    cartesian = qsc.elliptical_channel_gradient(
        1.3,
        0.2,
        parallel_current_mu0=0.7,
        chi=1,
        frame=rotation,
    )

    np.testing.assert_allclose(
        cartesian,
        rotation.T @ frenet @ rotation,
        atol=2.0e-15,
    )


def test_external_gradient_is_symmetric_trace_free_and_ampere_cancels():
    solution = finite_current_solution()
    result = qsc.plasma_gradient_on_axis(
        solution,
        formal_radius=0.05,
    )

    assert result.maximum_divergence < 2.0e-15
    assert result.maximum_ampere_error < 2.0e-15
    assert result.maximum_external_asymmetry < 5.0e-9
    assert result.maximum_external_trace < 2.0e-9
    np.testing.assert_allclose(
        result.external_gradient,
        result.external_gradient_stf,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        qsc.unpack_symmetric_trace_free_rank2(result.external_gradient_independent),
        result.external_gradient_stf,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.external_field,
        solution.B_axis - result.field.field,
    )


def test_vacuum_reduction_leaves_total_field_jet_unchanged():
    solution = qsc.Qsc(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=0.64,
        I2=0.0,
        p2=0.0,
        B2c=-0.00322,
        nphi=31,
        order="r2",
    )
    result = qsc.plasma_gradient_on_axis(
        solution,
        formal_radius=0.05,
    )

    np.testing.assert_allclose(result.field.field, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.gradient, 0.0)
    np.testing.assert_allclose(result.external_field, solution.B_axis)
    np.testing.assert_allclose(
        result.external_gradient,
        solution.grad_B_axis,
    )


def test_stf_rank_two_pack_unpack_and_projection():
    tensor = jnp.asarray(
        [
            [2.0, 1.0, -3.0],
            [3.0, -1.0, 4.0],
            [5.0, 2.0, 7.0],
        ]
    )
    projected = qsc.project_symmetric_trace_free_rank2(tensor)
    packed = qsc.pack_symmetric_trace_free_rank2(tensor)
    unpacked = qsc.unpack_symmetric_trace_free_rank2(packed)

    np.testing.assert_allclose(projected, projected.T)
    np.testing.assert_allclose(np.trace(projected), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(unpacked, projected)
    assert packed.shape == (5,)


def test_elliptical_gradient_is_jittable_and_differentiable():
    def component(x):
        return qsc.elliptical_channel_gradient(
            x,
            0.2,
            parallel_current_mu0=0.7,
            chi=1,
        )[1, 2]

    x = 1.3
    value = component(x)
    jitted = jax.jit(component)(x)
    tangent = jax.jvp(component, (x,), (1.0,))[1]
    step = 1.0e-5
    finite_difference = (component(x + step) - component(x - step)) / (2 * step)

    np.testing.assert_allclose(jitted, value)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-10)


def test_gradient_and_stf_guards():
    with pytest.raises(ValueError, match="chi"):
        qsc.elliptical_channel_gradient(
            1.0,
            0.0,
            parallel_current_mu0=1.0,
            chi=0,
        )
    with pytest.raises(ValueError, match="frame"):
        qsc.elliptical_channel_gradient(
            1.0,
            0.0,
            parallel_current_mu0=1.0,
            chi=1,
            frame=[1.0, 2.0],
        )
    with pytest.raises(ValueError, match="shape"):
        qsc.project_symmetric_trace_free_rank2(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="length 5"):
        qsc.unpack_symmetric_trace_free_rank2(jnp.ones(4))

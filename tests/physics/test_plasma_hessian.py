from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.plasma import _principal_transverse_plasma_hessian


def finite_pressure_current_solution(*, nphi=61, I2=0.9):
    return qsc.Qsc(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=I2,
        p2=-600000.0,
        B2c=-0.7,
        nphi=nphi,
        order="r2",
    )


def vacuum_qa_solution(*, nphi=61):
    return qsc.Qsc(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=0.64,
        I2=0.0,
        p2=0.0,
        B2c=-0.00322,
        nphi=nphi,
        order="r2",
    )


def test_stf_rank_three_pack_unpack_and_projection():
    tensor = jnp.arange(54.0).reshape(2, 3, 3, 3)
    projected = qsc.project_symmetric_trace_free_rank3(tensor)
    packed = qsc.pack_symmetric_trace_free_rank3(tensor)
    unpacked = qsc.unpack_symmetric_trace_free_rank3(packed)

    for permutation in permutations((1, 2, 3)):
        np.testing.assert_allclose(
            projected,
            jnp.transpose(projected, (0, *permutation)),
            atol=1.0e-14,
        )
    np.testing.assert_allclose(
        jnp.einsum("...iik->...k", projected),
        0.0,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(unpacked, projected, atol=2.0e-14)
    assert packed.shape == (2, 7)


def test_stf_rank_three_helpers_are_jittable():
    tensor = jnp.arange(27.0).reshape(3, 3, 3)

    eager = qsc.unpack_symmetric_trace_free_rank3(qsc.pack_symmetric_trace_free_rank3(tensor))
    compiled = jax.jit(
        lambda value: qsc.unpack_symmetric_trace_free_rank3(
            qsc.pack_symmetric_trace_free_rank3(value)
        )
    )(tensor)

    np.testing.assert_allclose(compiled, eager)


def test_external_hessian_is_fully_symmetric_and_trace_free():
    solution = finite_pressure_current_solution(nphi=121)
    formal_radius = 0.05
    result = qsc.plasma_hessian_on_axis(
        solution,
        formal_radius=0.05,
    )

    assert result.maximum_derivative_asymmetry < 2.0e-15
    assert result.maximum_external_symmetry_error < 8.0e-11
    assert result.maximum_external_trace < 2.0e-10
    np.testing.assert_allclose(
        result.external_hessian,
        result.external_hessian_stf,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        qsc.unpack_symmetric_trace_free_rank3(result.external_hessian_independent),
        result.external_hessian_stf,
        atol=2.0e-15,
    )
    axis_scale = solution.geometry.abs_G0_over_B0
    expected_remainder = (
        jnp.max(jnp.abs(result.hessian))
        * (formal_radius / axis_scale) ** 2
        * (1 + jnp.abs(jnp.log(formal_radius / axis_scale)))
    )
    np.testing.assert_allclose(result.estimated_hessian_remainder, expected_remainder)


def test_qh_external_hessian_preserves_oriented_ellipse_topology():
    solution = qsc.Qsc(
        rc=[1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
        zs=[0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
        nfp=4,
        etabar=1.569,
        I2=0.2,
        p2=-100000.0,
        B2c=0.1348,
        nphi=121,
        order="r2",
    )
    result = qsc.plasma_hessian_on_axis(
        solution,
        formal_radius=0.04,
    )

    assert int(solution.helicity) == 1
    assert result.maximum_derivative_asymmetry < 2.0e-15
    assert result.maximum_external_symmetry_error < 2.0e-9
    assert result.maximum_external_trace < 3.0e-9


def test_circular_curved_channel_recovers_finite_conductor_limit():
    """Equation (215), with affine and second-order shape terms suppressed."""

    solution = qsc.Qsc(
        rc=[1.0],
        zs=[0.0],
        nfp=1,
        etabar=1.0,
        I2=0.1,
        nphi=31,
        order="r2",
    )
    zeros = jnp.zeros_like(solution.X20)
    circular_second_order = replace(
        solution.second_order,
        X20=zeros,
        X2c=zeros,
        X2s=zeros,
        Y20=zeros,
        Y2c=zeros,
        Y2s=zeros,
        Z20=zeros,
        Z2c=zeros,
        Z2s=zeros,
    )
    curvature_only_solution = replace(
        solution,
        second_order=circular_second_order,
    )
    source = qsc.plasma_current_source(
        solution,
        formal_radius=0.1,
    )
    zero_vector = jnp.zeros_like(source.wstar2_cosine)
    curvature_only_source = replace(
        source,
        wstar2_cosine=zero_vector,
        wstar2_sine=zero_vector,
    )

    transverse, _ = _principal_transverse_plasma_hessian(
        curvature_only_solution,
        curvature_only_source,
    )
    coefficient = source.parallel_current_mu0 * solution.geometry.curvature / 8
    expected = jnp.zeros_like(transverse)
    expected = expected.at[:, 0, 0, 2].set(-coefficient)
    expected = expected.at[:, 0, 1, 1].set(-coefficient)
    expected = expected.at[:, 1, 0, 1].set(-coefficient)
    expected = expected.at[:, 1, 1, 2].set(-3 * coefficient)

    np.testing.assert_allclose(transverse, expected, atol=2.0e-15)


def test_vacuum_reduction_leaves_total_hessian_unchanged():
    solution = vacuum_qa_solution()
    result = qsc.plasma_hessian_on_axis(
        solution,
        formal_radius=0.05,
    )

    np.testing.assert_allclose(result.field.field.field, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.field.gradient, 0.0)
    np.testing.assert_allclose(result.hessian, 0.0)
    np.testing.assert_allclose(
        result.external_hessian,
        solution.grad_grad_B_axis,
    )


def test_plasma_hessian_converges_spectrally():
    medium = qsc.plasma_hessian_on_axis(
        finite_pressure_current_solution(nphi=61),
        formal_radius=0.05,
    )
    fine = qsc.plasma_hessian_on_axis(
        finite_pressure_current_solution(nphi=121),
        formal_radius=0.05,
    )

    np.testing.assert_allclose(
        medium.hessian_frenet[0],
        fine.hessian_frenet[0],
        atol=1.0e-8,
        rtol=1.0e-8,
    )
    assert fine.maximum_external_symmetry_error < 1.0e-4 * (medium.maximum_external_symmetry_error)
    assert fine.maximum_external_trace < 1.0e-4 * medium.maximum_external_trace


def test_plasma_hessian_supports_jit_and_jvp():
    def component(I2):
        result = qsc.plasma_hessian_on_axis(
            finite_pressure_current_solution(nphi=15, I2=I2),
            formal_radius=0.05,
            angular_resolution=32,
        )
        return result.hessian[4, 0, 1, 2]

    value = component(0.9)
    compiled = jax.jit(component)(0.9)
    tangent = jax.jvp(component, (0.9,), (1.0,))[1]
    step = 1.0e-5
    finite_difference = (component(0.9 + step) - component(0.9 - step)) / (2 * step)

    np.testing.assert_allclose(compiled, value, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-8, atol=2.0e-8)


def test_rank_three_guards():
    with pytest.raises(ValueError, match="shape"):
        qsc.project_symmetric_trace_free_rank3(jnp.ones((3, 3)))
    with pytest.raises(ValueError, match="length 7"):
        qsc.unpack_symmetric_trace_free_rank3(jnp.ones(6))

"""Plasma-current field on axis: current source, field, gradient and Hessian splits."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
import pyqsc_jax.plasma as qp
from fixtures import solve_configuration
from pyqsc_jax.second_order import MU0


def finite_current(*, I2=0.9, p2=-600000.0, nphi=31):
    return solve_configuration("finite_pressure_current", I2=I2, p2=p2, nphi=nphi)


def vacuum_qa(nphi=31):
    return solve_configuration("qa", nphi=nphi)


def central_difference(function, x, step):
    return (function(x + step) - function(x - step)) / (2 * step)


# --- Current source -----------------------------------------------------------------------


def test_covariant_and_enclosed_current_conversions_invert_each_other():
    """I(a) = 2 pi chi I2 a^2 / mu0 round-trips, and dI/da = 4 pi chi I2 a / mu0."""

    I2, radius, chi = 0.9, 0.08, -1
    current = qp.enclosed_current_from_covariant(I2, formal_radius=radius, chi=chi)
    np.testing.assert_allclose(
        qp.covariant_current_from_enclosed(current, formal_radius=radius, chi=chi), I2
    )
    derivative = jax.grad(
        lambda a: qp.enclosed_current_from_covariant(I2, formal_radius=a, chi=chi)
    )(radius)
    np.testing.assert_allclose(derivative, 4 * np.pi * chi * I2 * radius / MU0)
    for kwargs, message in (
        ({"formal_radius": 0.0, "chi": 1}, "positive"),
        ({"formal_radius": -0.1, "chi": 1}, "positive"),
        ({"formal_radius": 0.1, "chi": 0}, "chi"),
    ):
        with pytest.raises(ValueError, match=message):
            qp.enclosed_current_from_covariant(1.0, **kwargs)
    with pytest.raises(ValueError, match="scalar"):
        qp.covariant_current_from_enclosed(1.0, formal_radius=[0.1], chi=1)
    with pytest.raises(ValueError, match="chi"):
        qp.covariant_current_from_enclosed(1.0, formal_radius=0.1, chi=0)


def test_current_source_obeys_ampere_and_vanishes_at_the_axis():
    """Enclosed current = pi a^2 j / mu0; the weighted current is O(r) and batch-shaped."""

    solution = finite_current(nphi=15)
    radius = 0.07
    source = qsc.plasma_current_source(solution, formal_radius=radius)
    np.testing.assert_allclose(
        source.enclosed_toroidal_current, np.pi * radius**2 * source.parallel_current_mu0 / MU0
    )
    np.testing.assert_allclose(
        source.w1, source.parallel_current_mu0 * solution.geometry.tangent_cartesian
    )

    weighted = qp.evaluate_weighted_current(
        source, jnp.asarray([0.0, 0.01]), jnp.asarray([0.2, 0.7])
    )
    assert weighted.shape == (2, 15, 3)
    np.testing.assert_allclose(weighted[0], 0.0)
    assert jax.jit(qp.evaluate_weighted_current)(source, 0.01, 0.3).shape == (15, 3)


def test_current_source_is_well_defined_at_zero_current():
    """At I2 = 0 the pressure-driven part survives; with p2 = 0 too, the source vanishes."""

    source = qsc.plasma_current_source(finite_current(I2=0.0), formal_radius=0.06)
    np.testing.assert_allclose(source.w1, 0.0)
    assert np.max(np.abs(source.wstar2_cosine)) > 0 and np.all(np.isfinite(source.wstar2_sine))

    vacuum = qsc.plasma_current_source(finite_current(I2=0.0, p2=0.0), formal_radius=0.06)
    np.testing.assert_allclose(vacuum.wstar2_cosine, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(vacuum.wstar2_sine, 0.0, atol=1.0e-14)


# --- Field --------------------------------------------------------------------------------


def test_matching_reference_length_cancels_exactly():
    """The matched axis-plus-core kernel is independent of the arbitrary reference length."""

    solution = finite_current()
    source = qsc.plasma_current_source(solution, formal_radius=0.06)
    integral = qsc.regularized_axis_integral(solution)
    first = qp.matched_plasma_field_kernel(solution, source, integral, reference_length=2.3)
    second = qp.matched_plasma_field_kernel(solution, source, integral, reference_length=7.1)
    plasma = qsc.plasma_field_on_axis(solution, formal_radius=0.06)

    np.testing.assert_allclose(first, second, rtol=0, atol=2.0e-15)
    assert plasma.maximum_matching_scale_error < 2.0e-15
    np.testing.assert_allclose(plasma.matched_axis_and_core, first, atol=2.0e-15)
    with pytest.raises(ValueError, match="reference_length"):
        qp.matched_plasma_field_kernel(solution, source, integral, reference_length=0.0)


def test_field_converges_in_angular_and_toroidal_resolution():
    """Shape correction and field converge in theta; the axis integral converges in nphi."""

    solution = finite_current(nphi=15)
    coarse = qsc.plasma_field_on_axis(solution, formal_radius=0.05, angular_resolution=64)
    fine = qsc.plasma_field_on_axis(solution, formal_radius=0.05, angular_resolution=128)
    np.testing.assert_allclose(coarse.field, fine.field, rtol=2.0e-9, atol=5.0e-15)
    np.testing.assert_allclose(
        coarse.second_order_shape_correction,
        fine.second_order_shape_correction,
        rtol=2.0e-9,
        atol=5.0e-15,
    )
    with pytest.raises(ValueError, match="angular_resolution"):
        qsc.plasma_field_on_axis(solution, formal_radius=0.05, angular_resolution=7)

    coarse_integral = qsc.regularized_axis_integral(
        solve_configuration("r1_qa", nphi=31, order="r1")
    )
    fine_integral = qsc.regularized_axis_integral(solve_configuration("r1_qa", nphi=61, order="r1"))
    np.testing.assert_allclose(coarse_integral[0], fine_integral[0], rtol=2.0e-3, atol=2.0e-5)


def test_vacuum_limit_and_pressure_only_remainder_indicator():
    """Vacuum gives zero plasma field; pressure-only fields are not certified exact.

    The remainder indicator must scale as |B_p| s^2 (1 + |log s|) with s = a / (G0/B0)
    rather than vanish when I2 = 0.
    """

    vacuum = qsc.plasma_field_on_axis(vacuum_qa(), formal_radius=0.05)
    np.testing.assert_allclose(vacuum.field, 0.0, atol=1.0e-14)
    assert float(vacuum.estimated_field_remainder) == 0

    pressure_only = finite_current(I2=0.0)
    field = qsc.plasma_field_on_axis(pressure_only, formal_radius=0.05)
    assert np.max(np.abs(field.field)) > 0
    np.testing.assert_allclose(field.field, field.second_order_shape_correction)
    scale = 0.05 / float(pressure_only.geometry.abs_G0_over_B0)
    expected = np.max(np.linalg.norm(field.field, axis=-1)) * scale**2 * (1 + abs(np.log(scale)))
    np.testing.assert_allclose(field.estimated_field_remainder, expected, rtol=1e-12)


def test_field_radius_derivative_matches_finite_difference():
    """d B_p / d a by forward-mode autodiff equals a central difference."""

    solution = finite_current(nphi=15)

    def component(radius):
        return qsc.plasma_field_on_axis(
            solution, formal_radius=radius, angular_resolution=16
        ).field[0, 2]

    tangent = jax.jit(lambda a: jax.jvp(component, (a,), (1.0,))[1])(0.05)
    np.testing.assert_allclose(tangent, central_difference(component, 0.05, 1e-5), rtol=2.0e-7)


# --- Gradient -----------------------------------------------------------------------------


def test_external_gradient_is_symmetric_trace_free_and_ampere_cancels():
    """Subtracting the plasma gradient leaves a curl- and divergence-free external gradient."""

    solution = finite_current(nphi=61)
    result = qsc.plasma_gradient_on_axis(solution, formal_radius=0.05)

    assert result.maximum_divergence < 2.0e-15
    assert result.maximum_ampere_error < 2.0e-15
    assert result.maximum_external_asymmetry < 5.0e-9
    assert result.maximum_external_trace < 2.0e-9
    np.testing.assert_allclose(result.external_gradient, result.external_gradient_stf, atol=2.0e-9)
    np.testing.assert_allclose(result.external_field, solution.B_axis - result.field.field)


def test_vacuum_reduction_leaves_total_field_jet_unchanged():
    """In vacuum the plasma gradient and Hessian are zero and external == total jet."""

    solution = vacuum_qa(nphi=61)
    gradient = qsc.plasma_gradient_on_axis(solution, formal_radius=0.05)
    np.testing.assert_allclose(gradient.gradient, 0.0)
    np.testing.assert_allclose(gradient.external_gradient, solution.grad_B_axis)
    hessian = qsc.plasma_hessian_on_axis(solution, formal_radius=0.05)
    np.testing.assert_allclose(hessian.hessian, 0.0)
    np.testing.assert_allclose(hessian.external_hessian, solution.grad_grad_B_axis)


def test_gradient_branch_jumps_at_zero_current_by_the_order_a2_term():
    """The I2 == 0 branch keeps one more order, so the gradient jumps as I2 -> 0 (documented).

    The finite-current gradient tends to zero as I2 -> 0 while the field is continuous; the
    jump is the whole zero-current term and must stay visible rather than be smoothed.
    """

    radius = 0.03
    at_zero = qsc.plasma_gradient_on_axis(finite_current(I2=0.0, nphi=61), formal_radius=radius)
    near_zero = qsc.plasma_gradient_on_axis(finite_current(I2=1e-9, nphi=61), formal_radius=radius)
    zero_term = np.max(np.abs(np.asarray(at_zero.gradient)))
    assert zero_term > 1e-4
    assert np.max(np.abs(np.asarray(near_zero.gradient))) < 1e-6 * zero_term
    np.testing.assert_allclose(near_zero.field.field, at_zero.field.field, atol=1e-8 * zero_term)

    with pytest.raises(ValueError, match="second-order solution"):
        qp.zero_current_gradient(
            replace(finite_current(I2=0.0), second_order=None),
            jnp.zeros((31, 3)),
            formal_radius=0.05,
        )


def test_elliptical_channel_gradient_rotates_covariantly_and_differentiates():
    """G_cart = F^T G_frenet F for a rotated frame; d G / d x matches a central difference."""

    angle = 0.37
    rotation = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), np.sin(angle)], [0.0, -np.sin(angle), np.cos(angle)]]
    )
    frenet = qp.elliptical_channel_gradient(1.3, 0.2, parallel_current_mu0=0.7, chi=1)
    cartesian = qp.elliptical_channel_gradient(
        1.3, 0.2, parallel_current_mu0=0.7, chi=1, frame=rotation
    )
    np.testing.assert_allclose(cartesian, rotation.T @ frenet @ rotation, atol=2.0e-15)

    def component(x):
        return qp.elliptical_channel_gradient(x, 0.2, parallel_current_mu0=0.7, chi=1)[1, 2]

    tangent = jax.jvp(component, (1.3,), (1.0,))[1]
    np.testing.assert_allclose(tangent, central_difference(component, 1.3, 1e-5), rtol=2.0e-10)
    with pytest.raises(ValueError, match="chi"):
        qp.elliptical_channel_gradient(1.0, 0.0, parallel_current_mu0=1.0, chi=0)
    with pytest.raises(ValueError, match="frame"):
        qp.elliptical_channel_gradient(1.0, 0.0, parallel_current_mu0=1.0, chi=1, frame=[1.0, 2.0])


def test_symmetric_trace_free_projections_pack_and_unpack():
    """Rank-2 and rank-3 STF projections are symmetric, trace-free and round-trip packing."""

    tensor = jnp.asarray([[2.0, 1.0, -3.0], [3.0, -1.0, 4.0], [5.0, 2.0, 7.0]])
    projected = qp.project_symmetric_trace_free_rank2(tensor)
    packed = qp.pack_symmetric_trace_free_rank2(tensor)
    np.testing.assert_allclose(projected, projected.T)
    np.testing.assert_allclose(np.trace(projected), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(qp.unpack_symmetric_trace_free_rank2(packed), projected)
    assert packed.shape == (5,)

    tensor = jnp.arange(54.0).reshape(2, 3, 3, 3)
    projected = qp.project_symmetric_trace_free_rank3(tensor)
    packed = qp.pack_symmetric_trace_free_rank3(tensor)
    for permutation in permutations((1, 2, 3)):
        np.testing.assert_allclose(
            projected, jnp.transpose(projected, (0, *permutation)), atol=1e-14
        )
    np.testing.assert_allclose(jnp.einsum("...iik->...k", projected), 0.0, atol=3.0e-14)
    np.testing.assert_allclose(qp.unpack_symmetric_trace_free_rank3(packed), projected, atol=2e-14)
    assert packed.shape == (2, 7)

    for function, argument, message in (
        (qp.project_symmetric_trace_free_rank2, jnp.ones((2, 2)), "shape"),
        (qp.unpack_symmetric_trace_free_rank2, jnp.ones(4), "length 5"),
        (qp.project_symmetric_trace_free_rank3, jnp.ones((3, 3)), "shape"),
        (qp.unpack_symmetric_trace_free_rank3, jnp.ones(6), "length 7"),
    ):
        with pytest.raises(ValueError, match=message):
            function(argument)


# --- Hessian ------------------------------------------------------------------------------


def test_external_hessian_is_trace_free_symmetric_and_converges():
    """External Hessian is STF to 2e-10 at nphi = 121, errors drop 1e4-fold from nphi = 61.

    The remainder indicator equals max|H_p| s^2 (1 + |log s|) with s = a / (G0/B0).
    """

    formal_radius = 0.05
    solution = finite_current(nphi=121)
    medium = qsc.plasma_hessian_on_axis(finite_current(nphi=61), formal_radius=formal_radius)
    fine = qsc.plasma_hessian_on_axis(solution, formal_radius=formal_radius)

    assert fine.maximum_derivative_asymmetry < 2.0e-15
    assert fine.maximum_external_symmetry_error < 8.0e-11
    assert fine.maximum_external_trace < 2.0e-10
    assert fine.maximum_external_symmetry_error < 1.0e-4 * medium.maximum_external_symmetry_error
    assert fine.maximum_external_trace < 1.0e-4 * medium.maximum_external_trace
    np.testing.assert_allclose(
        medium.hessian_frenet[0], fine.hessian_frenet[0], atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        qp.unpack_symmetric_trace_free_rank3(fine.external_hessian_independent),
        fine.external_hessian_stf,
        atol=2.0e-15,
    )
    scale = formal_radius / solution.geometry.abs_G0_over_B0
    expected = jnp.max(jnp.abs(fine.hessian)) * scale**2 * (1 + jnp.abs(jnp.log(scale)))
    np.testing.assert_allclose(fine.estimated_hessian_remainder, expected)


def test_qh_external_hessian_preserves_oriented_ellipse_topology():
    """A helicity-1 QH case with current keeps a symmetric, trace-free external Hessian."""

    solution = solve_configuration("qh", I2=0.2, p2=-100000.0, nphi=121)
    result = qsc.plasma_hessian_on_axis(solution, formal_radius=0.04)

    assert int(solution.helicity) == 1
    assert result.maximum_external_symmetry_error < 2.0e-9
    assert result.maximum_external_trace < 3.0e-9


def test_plasma_hessian_current_derivative_matches_finite_difference():
    """d(grad grad B_p)/d I2 by forward-mode autodiff equals a central difference."""

    def component(I2):
        result = qsc.plasma_hessian_on_axis(
            finite_current(nphi=15, I2=I2), formal_radius=0.05, angular_resolution=32
        )
        return result.hessian[4, 0, 1, 2]

    tangent = jax.jvp(component, (0.9,), (1.0,))[1]
    np.testing.assert_allclose(
        tangent, central_difference(component, 0.9, 1e-5), rtol=2.0e-8, atol=2.0e-8
    )

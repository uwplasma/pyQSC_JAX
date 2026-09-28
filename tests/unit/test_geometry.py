"""Magnetic-axis parametrization and Frenet geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyqsc_jax import Axis
from pyqsc_jax.geometry import compute_axis_geometry, evaluate_axis


def test_axis_coefficients_are_normalized_and_round_trip():
    """Coefficient arrays are zero-padded, dofs round-trip, and the symmetry residual is exact."""

    axis = Axis(rc=[1.0, 0.2, -0.01], rs=[0.0, 0.03], zc=[0.1], zs=[0.0, -0.15], nfp=5)

    assert axis.nfourier == 3
    np.testing.assert_array_equal(axis.rs, [0.0, 0.03, 0.0])
    np.testing.assert_array_equal(axis.zc, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(axis.stellarator_symmetry_residual, 0.1)
    rebuilt = Axis.from_dofs(axis.dofs, nfp=axis.nfp)
    assert rebuilt.nfp == axis.nfp
    np.testing.assert_array_equal(rebuilt.dofs, axis.dofs)
    np.testing.assert_allclose(axis.with_dofs(axis.dofs.at[0].add(0.2)).rc[0], 1.2)


def test_general_axis_matches_closed_form_derivatives():
    """R, Z and their phi derivatives match the hand-differentiated Fourier series."""

    axis = Axis(rc=[1.2, 0.3], rs=[0.0, -0.11], zc=[0.2, 0.04], zs=[0.0, 0.17], nfp=3)
    phi = jnp.array([0.0, 0.13, 0.51])
    samples = evaluate_axis(axis, phi)
    angle = 3 * phi

    np.testing.assert_allclose(samples.R, 1.2 + 0.3 * jnp.cos(angle) - 0.11 * jnp.sin(angle))
    np.testing.assert_allclose(samples.Z, 0.2 + 0.04 * jnp.cos(angle) + 0.17 * jnp.sin(angle))
    np.testing.assert_allclose(samples.d_R_d_phi, -0.9 * jnp.sin(angle) - 0.33 * jnp.cos(angle))
    np.testing.assert_allclose(samples.d_Z_d_phi, -0.12 * jnp.sin(angle) + 0.51 * jnp.cos(angle))
    np.testing.assert_allclose(samples.d2_R_d_phi2, -2.7 * jnp.cos(angle) + 0.99 * jnp.sin(angle))
    np.testing.assert_allclose(samples.d3_Z_d_phi3, 1.08 * jnp.sin(angle) - 4.59 * jnp.cos(angle))


def test_circular_axis_geometry_is_analytic():
    """A planar circle has length 2 pi R, curvature 1/R, zero torsion and a fixed frame."""

    major_radius = 2.0
    nphi = 31
    geometry = compute_axis_geometry(Axis(rc=[major_radius], zs=[0.0]), nphi=nphi)

    np.testing.assert_allclose(geometry.axis_length, 2 * jnp.pi * major_radius, rtol=2e-14)
    np.testing.assert_allclose(geometry.curvature, 1 / major_radius, rtol=2e-14)
    np.testing.assert_allclose(geometry.torsion, 0.0, atol=2e-14)
    np.testing.assert_allclose(geometry.varphi, geometry.samples.phi, rtol=2e-14, atol=2e-14)
    for vector, expected in (
        (geometry.tangent_cylindrical, [0.0, 1.0, 0.0]),
        (geometry.normal_cylindrical, [-1.0, 0.0, 0.0]),
        (geometry.binormal_cylindrical, [0.0, 0.0, 1.0]),
    ):
        np.testing.assert_allclose(vector, jnp.tile(jnp.array(expected), (nphi, 1)), atol=2e-14)
    assert int(geometry.frame_helicity) == 0
    assert bool(geometry.diagnostics.frenet_valid)


def test_frenet_frame_is_right_handed_and_orthonormal():
    """(t, n, b) is orthonormal with determinant +1 on a nonplanar axis."""

    geometry = compute_axis_geometry(Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3), nphi=31)
    frame = jnp.stack(
        (geometry.tangent_cartesian, geometry.normal_cartesian, geometry.binormal_cartesian),
        axis=-2,
    )
    gram = frame @ jnp.swapaxes(frame, -1, -2)

    np.testing.assert_allclose(gram, jnp.broadcast_to(jnp.eye(3), gram.shape), atol=2e-13)
    np.testing.assert_allclose(jnp.linalg.det(frame), 1.0, rtol=2e-13)
    assert geometry.diagnostics.maximum_frame_orthogonality_error < 1e-12


def test_axis_length_gradient_matches_finite_difference():
    """d(axis length)/d(rc1) from autodiff agrees with a central difference."""

    def length(rc1):
        axis = Axis(rc=jnp.array([1.0, rc1]), zs=jnp.array([0.0, -0.045]), nfp=3)
        return compute_axis_geometry(axis, nphi=31).axis_length

    step = 1e-5
    finite_difference = (length(0.045 + step) - length(0.045 - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(length)(0.045), finite_difference, rtol=2e-8)


def test_degenerate_axis_is_flagged_and_inputs_are_validated():
    """A zero-radius axis reports invalid Frenet/cylindrical data; bad inputs raise."""

    geometry = compute_axis_geometry(Axis(rc=[0.0], zs=[0.0]), nphi=15)
    assert not bool(geometry.diagnostics.frenet_valid)
    assert not bool(geometry.diagnostics.cylindrical_coordinates_valid)
    np.testing.assert_allclose(geometry.diagnostics.minimum_speed, 0.0)

    with pytest.raises(ValueError, match="nphi"):
        compute_axis_geometry(Axis(rc=[1.0], zs=[0.0]), nphi=2)
    for kwargs, message in (
        (dict(rc=[1.0], zs=[0.0], nfp=0), "positive integer"),
        (dict(rc=[1.0], zs=[0.0], nfp=True), "positive integer"),
        (dict(rc=[[1.0]], zs=[0.0]), "one-dimensional"),
        (dict(rc=[], zs=[]), "At least one"),
    ):
        with pytest.raises(ValueError, match=message):
            Axis(**kwargs)
    with pytest.raises(ValueError, match="divisible by 4"):
        Axis.from_dofs(jnp.arange(5.0), nfp=1)

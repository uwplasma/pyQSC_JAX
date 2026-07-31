import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyqsc_jax import Axis
from pyqsc_jax.geometry import compute_axis_geometry


def test_circular_axis_geometry_is_analytic():
    major_radius = 2.0
    nphi = 31
    geometry = compute_axis_geometry(Axis(rc=[major_radius], zs=[0.0]), nphi=nphi)

    np.testing.assert_allclose(geometry.axis_length, 2 * jnp.pi * major_radius, rtol=2e-14)
    np.testing.assert_allclose(geometry.curvature, 1 / major_radius, rtol=2e-14)
    np.testing.assert_allclose(geometry.torsion, 0.0, atol=2e-14)
    np.testing.assert_allclose(geometry.varphi, geometry.samples.phi, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        geometry.tangent_cylindrical,
        jnp.tile(jnp.array([0.0, 1.0, 0.0]), (nphi, 1)),
        atol=2e-14,
    )
    np.testing.assert_allclose(
        geometry.normal_cylindrical,
        jnp.tile(jnp.array([-1.0, 0.0, 0.0]), (nphi, 1)),
        atol=2e-14,
    )
    np.testing.assert_allclose(
        geometry.binormal_cylindrical,
        jnp.tile(jnp.array([0.0, 0.0, 1.0]), (nphi, 1)),
        atol=2e-14,
    )
    assert int(geometry.frame_helicity) == 0
    assert bool(geometry.diagnostics.frenet_valid)
    assert bool(geometry.diagnostics.cylindrical_coordinates_valid)


def test_frame_is_right_handed_and_orthonormal():
    axis = Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
    geometry = compute_axis_geometry(axis, nphi=31)
    frame = jnp.stack(
        (geometry.tangent_cartesian, geometry.normal_cartesian, geometry.binormal_cartesian),
        axis=-2,
    )
    gram = frame @ jnp.swapaxes(frame, -1, -2)

    expected_gram = jnp.broadcast_to(jnp.eye(3), gram.shape)
    np.testing.assert_allclose(gram, expected_gram, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(jnp.linalg.det(frame), 1.0, rtol=2e-13, atol=2e-13)
    assert geometry.diagnostics.maximum_frame_orthogonality_error < 1e-12


def test_invalid_axis_has_explicit_diagnostics():
    geometry = compute_axis_geometry(Axis(rc=[0.0], zs=[0.0]), nphi=15)

    assert not bool(geometry.diagnostics.frenet_valid)
    assert not bool(geometry.diagnostics.cylindrical_coordinates_valid)
    np.testing.assert_allclose(geometry.diagnostics.minimum_speed, 0.0)

    with pytest.raises(ValueError, match="nphi"):
        compute_axis_geometry(Axis(rc=[1.0], zs=[0.0]), nphi=2)


def test_geometry_jit_and_axis_length_gradient():
    axis = Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
    compiled = jax.jit(compute_axis_geometry, static_argnames=("nphi",))(axis, nphi=31)
    eager = compute_axis_geometry(axis, nphi=31)
    np.testing.assert_allclose(compiled.curvature, eager.curvature, rtol=2e-13)

    def length(rc1):
        varied = Axis(rc=jnp.array([1.0, rc1]), zs=axis.zs, nfp=axis.nfp)
        return compute_axis_geometry(varied, nphi=31).axis_length

    derivative = jax.grad(length)(0.045)
    step = 1e-5
    finite_difference = (length(0.045 + step) - length(0.045 - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-8, atol=2e-8)

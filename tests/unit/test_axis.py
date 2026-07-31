import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyqsc_jax import Axis
from pyqsc_jax.axis import evaluate_axis


def test_coefficients_are_normalized_and_round_trip():
    axis = Axis(
        rc=[1.0, 0.2, -0.01],
        rs=[0.0, 0.03],
        zc=[0.1],
        zs=[0.0, -0.15],
        nfp=5,
    )

    assert axis.nfourier == 3
    np.testing.assert_array_equal(axis.rs, [0.0, 0.03, 0.0])
    np.testing.assert_array_equal(axis.zc, [0.1, 0.0, 0.0])
    np.testing.assert_array_equal(axis.zs, [0.0, -0.15, 0.0])
    np.testing.assert_allclose(axis.stellarator_symmetry_residual, 0.1)
    rebuilt = Axis.from_dofs(axis.dofs, nfp=axis.nfp)
    assert rebuilt.nfp == axis.nfp
    np.testing.assert_array_equal(rebuilt.dofs, axis.dofs)
    shifted = axis.with_dofs(axis.dofs.at[0].add(0.2))
    np.testing.assert_allclose(shifted.rc[0], 1.2)


def test_axis_input_validation():
    with pytest.raises(ValueError, match="positive integer"):
        Axis(rc=[1.0], zs=[0.0], nfp=0)
    with pytest.raises(ValueError, match="positive integer"):
        Axis(rc=[1.0], zs=[0.0], nfp=True)
    with pytest.raises(ValueError, match="one-dimensional"):
        Axis(rc=[[1.0]], zs=[0.0])
    with pytest.raises(ValueError, match="At least one"):
        Axis(rc=[], zs=[])
    with pytest.raises(ValueError, match="divisible by 4"):
        Axis.from_dofs(jnp.arange(5.0), nfp=1)


def test_general_axis_and_analytic_derivatives():
    axis = Axis(
        rc=[1.2, 0.3],
        rs=[0.0, -0.11],
        zc=[0.2, 0.04],
        zs=[0.0, 0.17],
        nfp=3,
    )
    phi = jnp.array([0.0, 0.13, 0.51])
    samples = evaluate_axis(axis, phi)
    angle = 3 * phi

    np.testing.assert_allclose(samples.R, 1.2 + 0.3 * jnp.cos(angle) - 0.11 * jnp.sin(angle))
    np.testing.assert_allclose(samples.Z, 0.2 + 0.04 * jnp.cos(angle) + 0.17 * jnp.sin(angle))
    np.testing.assert_allclose(samples.d_R_d_phi, -0.9 * jnp.sin(angle) - 0.33 * jnp.cos(angle))
    np.testing.assert_allclose(samples.d_Z_d_phi, -0.12 * jnp.sin(angle) + 0.51 * jnp.cos(angle))
    np.testing.assert_allclose(samples.d2_R_d_phi2, -2.7 * jnp.cos(angle) + 0.99 * jnp.sin(angle))
    np.testing.assert_allclose(samples.d3_Z_d_phi3, 1.08 * jnp.sin(angle) - 4.59 * jnp.cos(angle))


def test_axis_is_jittable_vmappable_and_differentiable():
    phi = jnp.linspace(0.0, 0.9, 11)
    axis = Axis.stellarator_symmetric(rc=[1.0, 0.04], zs=[0.0, -0.04], nfp=3)
    eager = evaluate_axis(axis, phi)
    compiled = jax.jit(evaluate_axis)(axis, phi)
    np.testing.assert_allclose(compiled.R, eager.R, rtol=1e-14, atol=1e-14)

    offsets = jnp.array([-0.01, 0.0, 0.02])

    def radius_at_zero(offset):
        changed = Axis(rc=axis.rc.at[1].add(offset), zs=axis.zs, nfp=axis.nfp)
        return evaluate_axis(changed, 0.0).R

    np.testing.assert_allclose(jax.vmap(radius_at_zero)(offsets), 1.04 + offsets)
    np.testing.assert_allclose(jax.grad(radius_at_zero)(0.0), 1.0)

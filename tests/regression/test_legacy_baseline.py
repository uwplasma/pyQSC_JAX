import jax.numpy as jnp
import numpy as np
import pytest

from pyqsc_jax.near_axis import near_axis


def standard_field(**kwargs):
    parameters = {
        "rc": [1.0, 0.045],
        "zs": [0.0, -0.045],
        "etabar": -0.9,
        "nfp": 3,
        "nphi": 31,
    }
    parameters.update(kwargs)
    return near_axis(**parameters)


def test_standard_first_order_baseline():
    field = standard_field()

    np.testing.assert_allclose(field.iota, 0.41830690943386617, rtol=2e-13)
    np.testing.assert_allclose(field.axis_length, 6.340238817434161, rtol=2e-13)
    np.testing.assert_allclose(
        [jnp.min(field.curvature), jnp.max(field.curvature)],
        [0.5956163516982903, 1.3060121594235536],
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        [jnp.min(field.torsion), jnp.max(field.torsion)],
        [-2.272197039914583, 0.6057564303422814],
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        jnp.max(jnp.abs(field.sigma)), 0.9920706836908618, rtol=2e-13
    )
    assert field.B_axis.shape == (3, 31)
    assert field.grad_B_axis.shape == (3, 3, 31)


def test_first_order_field_methods_are_finite():
    field = standard_field()
    point = jnp.array([0.02, 0.4, 0.1])

    assert jnp.isfinite(field.AbsB(point))
    assert jnp.isfinite(field.jacobian(point))
    assert jnp.all(jnp.isfinite(field.B_covariant(point)))
    assert jnp.all(jnp.isfinite(field.B_contravariant(point)))


@pytest.mark.xfail(
    strict=True,
    reason="Constructor and dofs setter swap normal/binormal phi and z components.",
)
def test_dofs_noop_preserves_derived_frame():
    field = standard_field()
    normal_before = jnp.stack(
        [field.normal_R, field.normal_phi, field.normal_z], axis=1
    )
    binormal_before = jnp.stack(
        [field.binormal_R, field.binormal_phi, field.binormal_z], axis=1
    )

    field.dofs = field.dofs

    normal_after = jnp.stack(
        [field.normal_R, field.normal_phi, field.normal_z], axis=1
    )
    binormal_after = jnp.stack(
        [field.binormal_R, field.binormal_phi, field.binormal_z], axis=1
    )
    np.testing.assert_array_equal(normal_after, normal_before)
    np.testing.assert_array_equal(binormal_after, binormal_before)


@pytest.mark.xfail(
    strict=True,
    reason="The current order argument is inert; complete second order is not implemented.",
)
def test_r2_request_produces_second_order_solution():
    field = standard_field(order="r2", B2c=0.01, p2=-1.0e3)

    assert field.B20.shape == (field.nphi,)
    assert jnp.all(jnp.isfinite(field.B20))

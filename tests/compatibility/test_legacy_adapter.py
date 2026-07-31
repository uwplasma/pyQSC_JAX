import jax
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
        "nphi": 15,
    }
    parameters.update(kwargs)
    return near_axis(**parameters)


def test_legacy_dofs_validation_and_x_alias():
    field = standard_field()
    updated = field.x.at[-1].set(-0.85)
    field.x = updated

    np.testing.assert_array_equal(field.x, updated)
    np.testing.assert_allclose(field.etabar, -0.85)
    with pytest.raises(ValueError, match="dofs must have shape"):
        field.dofs = jnp.zeros(2)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"nphi": 16}, "odd integer"),
        ({"nphi": True}, "odd integer"),
        ({"rc": [[1.0]], "zs": [[0.0]]}, "one-dimensional"),
        ({"rc": [1.0, 0.1], "zs": [0.0]}, "equal length"),
    ],
)
def test_legacy_constructor_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        standard_field(**kwargs)


def test_calculate_and_pytree_round_trip():
    field = standard_field()
    calculated = field.calculate(field.rc, field.zs, field.etabar)
    np.testing.assert_allclose(calculated[7], field.iota)

    leaves, structure = jax.tree_util.tree_flatten(field)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    np.testing.assert_allclose(restored.iota, field.iota)
    assert restored.order == field.order


def test_coordinate_helpers_and_boundary_are_finite():
    field = standard_field()
    X = 0.01 * field.X1c_untwisted
    Y = 0.01 * field.Y1s_untwisted
    R, Z, phi = field.Frenet_to_cylindrical_1_point(0.0, X, Y)
    assert jnp.all(jnp.isfinite(jnp.asarray((R, Z, phi))))

    residual = field.Frenet_to_cylindrical_residual_func(0.0, phi, X, Y)
    np.testing.assert_allclose(residual, 0.0, atol=2e-14)
    interpolated = field.interpolated_array_at_point(field.R0, 2 * jnp.pi / field.nfp)
    np.testing.assert_allclose(interpolated, field.R0[0])

    R_period, Z_period, phi0 = field.Frenet_to_cylindrical(0.01, ntheta=5)
    assert R_period.shape == Z_period.shape == phi0.shape == (5, field.nphi)
    assert jnp.all(jnp.isfinite(R_period))
    RBC, ZBS = field.to_Fourier(R_period, Z_period, field.nfp, mpol=2, ntor=2)
    assert RBC.shape == ZBS.shape == (5, 3)

    x, y, z, cylindrical_R = field.get_boundary(
        r=0.01,
        ntheta=6,
        nphi=8,
        ntheta_fourier=5,
        mpol=2,
        ntor=2,
    )
    assert x.shape == y.shape == z.shape == cylindrical_R.shape == (6, 8)
    assert jnp.all(jnp.isfinite(jnp.stack((x, y, z))))


def test_varphi_coordinate_inversion():
    field = standard_field()
    r = 0.005
    theta = 0.3
    varphi = 0.1
    phi = field.phi_of_theta_varphi(r, theta, varphi)
    assert jnp.isfinite(phi)

    R, Z, phi0 = field.Frenet_to_cylindrical(r, ntheta=3, phi_is_varphi=True)
    assert R.shape == Z.shape == phi0.shape == (3, field.nphi)
    assert jnp.all(jnp.isfinite(R))

    x, y, z, _ = field.get_boundary(
        r=r,
        ntheta=2,
        nphi=3,
        ntheta_fourier=3,
        mpol=1,
        ntor=1,
        phi_is_varphi=True,
    )
    assert jnp.all(jnp.isfinite(jnp.stack((x, y, z))))


def test_legacy_field_magnitude_and_jitted_methods():
    field = standard_field(I2=0.2)
    point = jnp.asarray((0.01, 0.2, 0.1))
    expected = field.B0 * (1 + point[0] * field.etabar * jnp.cos(point[1]))
    np.testing.assert_allclose(field.AbsB(point), expected)
    assert jnp.all(jnp.isfinite(jax.jit(field.B_covariant)(point)))
    assert jnp.all(jnp.isfinite(jax.jit(field.B_contravariant)(point)))
    assert jnp.isfinite(jax.jit(field.jacobian)(point))
    assert jnp.isfinite(field.B_mag(*point))


def test_plot_supports_created_and_supplied_axes(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    field = standard_field()
    figure, axes = field.plot(
        r=0.005,
        ntheta=3,
        nphi=4,
        ntheta_fourier=3,
        show=False,
        close=True,
    )
    assert axes.figure is figure

    supplied_figure = plt.figure()
    supplied_axes = supplied_figure.add_subplot(projection="3d")
    returned_figure, returned_axes = field.plot(
        r=0.005,
        ntheta=3,
        nphi=4,
        ntheta_fourier=3,
        ax=supplied_axes,
        show=True,
        close=True,
        axis_equal=False,
    )
    assert returned_figure is supplied_figure
    assert returned_axes is supplied_axes

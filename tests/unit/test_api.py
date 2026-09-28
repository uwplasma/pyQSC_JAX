"""Public solve/Qsc API, input models, lazy diagnostics, and the legacy ``near_axis`` adapter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from fixtures import CONFIGURATIONS, solve_configuration
from pyqsc_jax.models import NearAxisInputs
from pyqsc_jax.near_axis import near_axis

R1_QA = CONFIGURATIONS["r1_qa"]


def standard_field(**kwargs):
    return near_axis(**{**R1_QA, "nphi": 15, **kwargs})


def test_input_models_reject_invalid_values():
    """NearAxisInputs and the adapter reject bad resolution, order, signs and shapes."""

    for overrides, message in (
        ({"nphi": 2}, "nphi"),
        ({"nphi": True}, "nphi"),
        ({"order": 4}, "order"),
        ({"sG": 0}, "sG"),
        ({"spsi": 0}, "spsi"),
        ({"etabar": jnp.ones(2)}, "etabar"),
    ):
        with pytest.raises(ValueError, match=message):
            NearAxisInputs(**{"axis": qsc.Axis(rc=[1.0], zs=[0.0]), "etabar": -0.9, **overrides})
    with pytest.raises(ValueError, match="order must be"):
        qsc.solve(axis=qsc.Axis(rc=[1.0], zs=[0.0]), etabar=-0.9, order="fourth")
    for kwargs, message in (
        ({"nphi": 16}, "odd integer"),
        ({"nphi": True}, "odd integer"),
        ({"rc": [[1.0]], "zs": [[0.0]]}, "one-dimensional"),
        ({"rc": [1.0, 0.1], "zs": [0.0]}, "equal length"),
        ({"order": "r4"}, "order must be"),
    ):
        with pytest.raises(ValueError, match=message):
            standard_field(**kwargs)


def test_higher_order_diagnostics_reject_first_order_solutions():
    """Every r2/r3 diagnostic raises a clear error on an r1 solution."""

    first_order = solve_configuration("r1_qa", nphi=15, order="r1")
    for function in (
        qsc.mercier_diagnostics,
        qsc.total_field_jet,
        qsc.singularity_diagnostics,
        qsc.second_order_residuals,
        qsc.solve_third_order,
        qsc.b20_diagnostics,
        qsc.optimal_B2c_value,
    ):
        with pytest.raises(ValueError, match="second-order"):
            function(first_order)
    with pytest.raises(ValueError, match="second-order"):
        qsc.plasma_current_source(first_order, formal_radius=0.1)
    for name, message in (
        ("X20", "no 'X20'"),
        ("X3c1", "Lower-order"),
        ("DMerc_times_r2", "Mercier"),
        ("grad_grad_B_axis", "second-derivative"),
        ("r_singularity", "singular"),
    ):
        with pytest.raises(AttributeError, match=message):
            getattr(first_order, name)


def test_diagnostics_are_lazy_unless_requested():
    """Default solves skip diagnostics; requested or attached diagnostics give the same values."""

    kwargs = dict(rc=[1.0, 0.09], zs=[0.0, -0.09], nfp=2, etabar=0.95, order="r2", nphi=31)
    plain = qsc.Qsc(**kwargs)
    assert plain.mercier is None and plain.field_jet is None and plain.singularity is None
    assert plain.second_order.linear_report.condition_number is None
    assert plain.root_report.jacobian_condition_number is None

    stored = qsc.Qsc(**kwargs, diagnostics=True)
    assert stored.second_order.linear_report.condition_number is not None
    assert stored.root_report.jacobian_condition_number is not None
    np.testing.assert_allclose(plain.r_singularity, stored.r_singularity, rtol=1e-12)
    np.testing.assert_allclose(plain.grad_grad_B_axis, stored.grad_grad_B_axis, atol=1e-10)
    np.testing.assert_allclose(plain.DMerc_times_r2, stored.DMerc_times_r2, rtol=1e-12)
    attached = plain.with_diagnostics()
    np.testing.assert_allclose(attached.singularity.r_singularity, stored.r_singularity)
    assert attached.axis is attached.inputs.axis
    for name in (
        "L_grad_grad_B",
        "grad_grad_B_inverse_scale_length_vs_varphi",
        "r_singularity_vs_varphi",
        "r_singularity_basic_vs_varphi",
    ):
        assert np.all(np.isfinite(getattr(attached, name)))


def test_solve_is_compiled_once_and_lowerable_ahead_of_time():
    """New values or Python/NumPy scalar types do not retrace; the pytree lowers under jit."""

    from pyqsc_jax.first_order import _solve

    axis = qsc.Axis(rc=[1.0, 0.09], zs=[0.0, -0.09], nfp=2)
    qsc.solve(axis=axis, etabar=0.95, nphi=23)
    size = _solve._cache_size()
    qsc.solve(axis=axis, etabar=jnp.asarray(0.9), nphi=23)
    qsc.solve(axis=axis, etabar=1, B0=np.float64(1.1), nphi=23)
    assert _solve._cache_size() == size

    lowered = jax.jit(lambda etabar: qsc.solve(axis=axis, etabar=etabar, nphi=23)).lower(0.95)
    assert lowered.compile()(0.95).iota.shape == ()


def test_essos_on_axis_contract():
    """The attributes and methods ESSOS reads from ``near_axis`` exist (ESSOS PR #70)."""

    field = near_axis(**R1_QA, nphi=31)
    attributes = (
        "B0 nfp nphi phi varphi R0 Z0 B_axis grad_B_axis axis_length iota iotaN curvature "
        "torsion elongation L_grad_B x dofs rc zs etabar sigma0 I2 spsi sG"
    ).split()
    methods = (
        "AbsB B_covariant B_contravariant B_mag jacobian get_boundary Frenet_to_cylindrical "
        "phi_of_theta_varphi to_vmec plot"
    ).split()
    assert all(hasattr(field, name) for name in attributes)
    assert all(callable(getattr(field, name)) for name in methods)


def test_legacy_dofs_setters_coerce_and_round_trip():
    """dofs/x setters validate shape, coerce to float64, keep the frame, and survive pytrees."""

    field = standard_field()
    normal_before = jnp.stack([field.normal_R, field.normal_phi, field.normal_z], axis=1)
    field.dofs = field.dofs
    np.testing.assert_array_equal(
        jnp.stack([field.normal_R, field.normal_phi, field.normal_z], axis=1), normal_before
    )
    field.x = field.x.at[-1].set(-0.85)
    np.testing.assert_allclose(field.etabar, -0.85)
    with pytest.raises(ValueError, match="dofs must have shape"):
        field.dofs = jnp.zeros(2)

    integer_inputs = near_axis(rc=[1, 0], zs=[0, 0], etabar=1, B0=1, I2=0, nfp=1, nphi=15)
    assert integer_inputs.rc.dtype == jnp.float64
    integer_inputs.x = [1, 0, 0, 0, 1]
    assert integer_inputs.dofs.dtype == jnp.float64

    leaves, structure = jax.tree_util.tree_flatten(field)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    np.testing.assert_allclose(restored.iota, field.iota)
    assert restored.order == field.order


def test_legacy_field_methods_are_consistent():
    """|B| = B0 (1 + r etabar cos theta) at first order; covariant/contravariant/jacobian jit."""

    field = standard_field(I2=0.2)
    point = jnp.asarray((0.01, 0.2, 0.1))
    expected = field.B0 * (1 + point[0] * field.etabar * jnp.cos(point[1]))
    np.testing.assert_allclose(field.AbsB(point), expected)
    assert jnp.all(jnp.isfinite(jax.jit(field.B_covariant)(point)))
    assert jnp.all(jnp.isfinite(jax.jit(field.B_contravariant)(point)))
    assert jnp.isfinite(jax.jit(field.jacobian)(point))
    assert jnp.isfinite(field.B_mag(*point))
    second_order = standard_field(order="r2", B2c=0.01, p2=-1.0e3)
    assert jnp.isfinite(second_order.B_mag(*point))


def test_legacy_coordinate_helpers_and_boundary():
    """Frenet-to-cylindrical inversion has zero residual; boundary and Fourier fits have shape."""

    field = standard_field()
    X = 0.01 * field.X1c_untwisted
    Y = 0.01 * field.Y1s_untwisted
    R, Z, phi = field.Frenet_to_cylindrical_1_point(0.0, X, Y)
    assert jnp.all(jnp.isfinite(jnp.asarray((R, Z, phi))))
    residual = field.Frenet_to_cylindrical_residual_func(0.0, phi, X, Y)
    np.testing.assert_allclose(residual, 0.0, atol=2e-14)
    np.testing.assert_allclose(
        field.interpolated_array_at_point(field.R0, 2 * jnp.pi / field.nfp), field.R0[0]
    )

    R_period, Z_period, phi0 = field.Frenet_to_cylindrical(0.01, ntheta=5)
    assert R_period.shape == Z_period.shape == phi0.shape == (5, field.nphi)
    RBC, ZBS = field.to_Fourier(R_period, Z_period, field.nfp, mpol=2, ntor=2)
    assert RBC.shape == ZBS.shape == (5, 3)
    x, y, z, _ = field.get_boundary(r=0.01, ntheta=6, nphi=8, ntheta_fourier=5, mpol=2, ntor=2)
    assert x.shape == (6, 8) and jnp.all(jnp.isfinite(jnp.stack((x, y, z))))
    assert jnp.isfinite(field.phi_of_theta_varphi(0.005, 0.3, 0.1))


def test_legacy_plot_and_vtk_outputs(monkeypatch, tmp_path):
    """plot() creates or reuses 3D axes; to_vtk() writes near-axis and external |B|."""

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    field = standard_field()
    figure, axes = field.plot(r=0.005, ntheta=3, nphi=4, ntheta_fourier=3, show=False, close=True)
    assert axes.figure is figure
    supplied_axes = plt.figure().add_subplot(projection="3d")
    _, returned_axes = field.plot(
        r=0.005,
        ntheta=3,
        nphi=4,
        ntheta_fourier=3,
        ax=supplied_axes,
        show=True,
        close=True,
        axis_equal=False,
    )
    assert returned_axes is supplied_axes

    pytest.importorskip("pyevtk")

    class UniformField:
        def AbsB(self, point):
            return jnp.linalg.norm(point) * 0 + 2.0

    field.to_vtk(tmp_path / "boundary", ntheta=6, nphi=8, ntheta_fourier=6, field=UniformField())
    text = (tmp_path / "boundary.vts").read_bytes()
    assert b"B_NearAxis" in text and b"B_BiotSavart" in text

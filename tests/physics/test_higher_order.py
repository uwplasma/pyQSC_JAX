"""Second- and third-order near-axis solves: equations, convergence, derivatives, surfaces."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from fixtures import CONFIGURATIONS, solve_configuration
from pyqsc_jax.near_axis import near_axis
from pyqsc_jax.plotting import surface_coordinates


def b20_residual(etabar):
    return solve_configuration("qa", nphi=15, etabar=etabar).B20_residual


def test_second_order_equations_are_satisfied():
    """All four O(r^2) equations vanish for vacuum QA and finite-pressure/current cases."""

    for name in ("qa", "finite_pressure_current"):
        solution = solve_configuration(name, nphi=31)
        residuals = qsc.second_order_residuals(solution)
        report = solution.linear_report
        assert bool(report.converged) and bool(report.finite) and bool(report.well_conditioned)
        assert report.relative_residual_norm < 3e-14
        assert residuals.maximum_absolute < 1e-11
        assert residuals.force_balance_1.shape == (31,)


def test_qh_untwisting_preserves_second_order_harmonic_norm():
    """Untwisting a helicity-1 QH solution rotates (X2c, X2s) without changing its norm."""

    solution = solve_configuration("qh", nphi=31)

    assert int(solution.helicity) == 1
    np.testing.assert_allclose(solution.X20_untwisted, solution.X20)
    np.testing.assert_allclose(
        solution.X2s_untwisted**2 + solution.X2c_untwisted**2,
        solution.X2s**2 + solution.X2c**2,
        rtol=2e-13,
        atol=2e-13,
    )
    assert qsc.second_order_residuals(solution).maximum_absolute < 2e-11


def test_second_order_resolution_convergence():
    """B20 mean and residual agree between nphi = 31 and 61 to < 1e-6."""

    medium = solve_configuration("qa", nphi=31)
    fine = solve_configuration("qa", nphi=61)
    np.testing.assert_allclose(medium.B20_mean, fine.B20_mean, rtol=6e-7)
    np.testing.assert_allclose(medium.B20_residual, fine.B20_residual, rtol=2e-7)


def test_second_order_derivative_matches_finite_difference():
    """d B20_residual/d etabar by autodiff equals a central difference; JVP == VJP."""

    etabar = 0.64
    step = 2e-5
    finite_difference = (b20_residual(etabar + step) - b20_residual(etabar - step)) / (2 * step)
    np.testing.assert_allclose(
        jax.grad(b20_residual)(etabar), finite_difference, rtol=2e-6, atol=2e-8
    )
    _, jvp = jax.jvp(b20_residual, (etabar,), (1.0,))
    (vjp,) = jax.vjp(b20_residual, etabar)[1](jnp.asarray(1.0))
    np.testing.assert_allclose(jvp, vjp, rtol=3e-11, atol=3e-11)


def test_third_order_structure_and_independent_constraints():
    """Only the m = 1 cosine/sine O(r^3) shapes are nonzero; the two constraint routes agree."""

    solution = solve_configuration("finite_pressure_current", nphi=61, order="r3")
    third = solution.third_order

    np.testing.assert_allclose(third.X3c1, solution.X1c * third.flux_constraint_coefficient)
    np.testing.assert_allclose(third.Y3s1, solution.Y1s * third.flux_constraint_coefficient)
    # Independent routes to the same coefficient agree to the O(r^2) solve resolution.
    np.testing.assert_allclose(
        third.B0_order_a_squared_to_cancel,
        2 * solution.inputs.B0 * third.flux_constraint_coefficient,
        rtol=0,
        atol=1.0e-9,
    )
    for name in ("X3s1", "Z3s1", "Z3c1", "X3s3", "X3c3", "Y3s3", "Y3c3", "Z3s3", "Z3c3"):
        np.testing.assert_array_equal(getattr(third, name), jnp.zeros(61))
    derivative = solution.geometry.d_d_varphi
    np.testing.assert_allclose(third.d_Y3c1_d_varphi, derivative @ third.Y3c1, atol=2.0e-12)


def test_flux_constraint_diagnostics_converge_spectrally():
    """The O(r^3) consistency checks are nonzero when coarse and converge to roundoff.

    pyQSC computes the coefficient from the parent O(r^3) equations and checks it against
    two shortened forms. Taking the coefficient from a shortened form would make both checks
    vanish identically, so they could never flag an under-resolved solve.
    """

    residuals = []
    for nphi in (15, 31, 61, 91):
        third = solve_configuration("finite_pressure_current", nphi=nphi, order="r3").third_order
        residuals.append(float(third.flux_constraint_residual))
        np.testing.assert_allclose(
            third.consistency_error, third.flux_constraint_residual, rtol=1e-6
        )
    assert residuals[0] > 1.0e-4
    assert residuals[1] < 0.1 * residuals[0] and residuals[2] < 0.1 * residuals[1]
    assert residuals[-1] < 1.0e-12


def test_r3_surface_adds_exactly_the_cubic_shape():
    """Adapter surfaces at r3 minus r2 equal r^3 times the untwisted O(r^3) harmonics."""

    parameters = {**CONFIGURATIONS["qh"], "nphi": 61}
    r3 = near_axis(**parameters, order="r3")
    r2 = near_axis(**parameters, order="r2")
    solution = r3.solution
    r, theta = 0.02, 0.37
    x2, y2, z2 = r2._frenet_displacements(r, theta)
    x3, y3, z3 = r3._frenet_displacements(r, theta)

    assert np.linalg.norm(np.asarray(solution.X3s1_untwisted)) > 0
    for difference, prefix in ((x3 - x2, "X"), (y3 - y2, "Y")):
        expected = r**3 * sum(
            getattr(solution, f"{prefix}3{kind}{m}_untwisted") * trig(m * theta)
            for m in (1, 3)
            for kind, trig in (("c", jnp.cos), ("s", jnp.sin))
        )
        np.testing.assert_allclose(difference, expected, atol=2.0e-15)
    np.testing.assert_allclose(z3, z2, atol=2.0e-15)


def test_third_order_derivative_matches_finite_difference():
    """d X3c1/d etabar by forward-mode autodiff equals a central difference."""

    def coefficient(etabar):
        return solve_configuration("qa", nphi=31, order="r3", etabar=etabar).X3c1[7]

    tangent = jax.jvp(coefficient, (jnp.asarray(-0.9),), (jnp.asarray(1.0),))[1]
    step = 2.0e-5
    finite_difference = (coefficient(-0.9 + step) - coefficient(-0.9 - step)) / (2 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=3.0e-5, atol=3.0e-7)


@pytest.mark.physics
def test_database_r3_surfaces_are_smooth():
    """Stellarator-database r3 surfaces are finite with no toroidal jumps above 0.25."""

    for configuration, radius in (
        ("database_qa_139524", 0.03),
        ("database_large_singularity_107579", 0.15),
    ):
        solution = solve_configuration(configuration, nphi=121)
        points = np.stack(surface_coordinates(solution, radius=radius, ntheta=36), axis=-1)
        assert np.all(np.isfinite(points))
        assert np.max(np.linalg.norm(np.diff(points, axis=1), axis=-1)) < 0.25

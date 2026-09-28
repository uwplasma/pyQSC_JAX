"""Second-order diagnostics: field jet, singular radius, Mercier, and B20 / B2c optimization."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from fixtures import CONFIGURATIONS, solve_configuration
from pyqsc_jax.near_axis import near_axis


def central_difference(function, x, step):
    return (function(x + step) - function(x - step)) / (2 * step)


# --- Field jet ----------------------------------------------------------------------------


def test_total_field_jet_satisfies_chain_rule_and_maxwell_identities():
    """Field, gradient and Hessian are divergence-free and curl-consistent to spectral accuracy."""

    for name in ("qa", "finite_pressure_current"):
        solution = solve_configuration(name, nphi=61)
        jet = qsc.total_field_jet(solution)
        assert jet.hessian.shape == (61, 3, 3, 3)
        assert jet.minimum_absolute_coordinate_jacobian > 0.9
        assert jet.maximum_field_error < 5e-15
        assert jet.maximum_gradient_error < 2e-8
        assert jet.maximum_divergence < 3e-8
        assert jet.maximum_derivative_asymmetry < 2e-13
        assert jet.maximum_divergence_gradient < 5e-5
        np.testing.assert_allclose(solution.grad_grad_B_axis, jet.hessian)


def test_vacuum_field_hessian_is_fully_symmetric_and_trace_free():
    """In vacuum B = grad(Phi), so grad grad B is totally symmetric and trace-free."""

    hessian = qsc.total_field_jet(solve_configuration("qa", nphi=61)).hessian
    np.testing.assert_allclose(hessian, np.swapaxes(hessian, 1, 2), atol=7e-7)
    np.testing.assert_allclose(hessian, np.swapaxes(hessian, 1, 3), atol=7e-7)
    np.testing.assert_allclose(np.einsum("niik->nk", hessian), 0, atol=7e-7)


def test_field_hessian_resolution_convergence_and_derivative():
    """grad grad B converges between nphi = 61 and 91; its etabar JVP matches a difference."""

    medium = solve_configuration("qa", nphi=61)
    fine = solve_configuration("qa", nphi=91)
    np.testing.assert_allclose(
        np.asarray(medium.grad_grad_B)[0], np.asarray(fine.grad_grad_B)[0], rtol=2e-6, atol=2e-6
    )

    def component(etabar):
        return solve_configuration("qa", nphi=31, etabar=etabar).grad_grad_B_axis[7, 0, 1, 2]

    tangent = jax.jvp(component, (0.64,), (1.0,))[1]
    np.testing.assert_allclose(tangent, central_difference(component, 0.64, 1e-5), rtol=1e-6)


# --- Singular radius ----------------------------------------------------------------------


def test_singular_radius_solves_the_determinant_equations():
    """At r_c(varphi) both g(r, theta) = 0 and dg/dtheta = 0 hold to roundoff."""

    solution = solve_configuration("qa", nphi=61)
    diagnostics = qsc.singularity_diagnostics(solution)
    np.testing.assert_allclose(
        diagnostics.g0, solution.geometry.abs_G0_over_B0 * solution.X1c * solution.Y1s, rtol=2e-13
    )
    assert jnp.max(jnp.abs(diagnostics.g1s)) < 2e-13
    assert diagnostics.maximum_residual_norm < 2e-13

    radius = diagnostics.r_singularity_vs_varphi
    theta = diagnostics.theta_singularity_vs_varphi
    linear = diagnostics.g1c * jnp.cos(theta) + diagnostics.g1s * jnp.sin(theta)
    linear_prime = -diagnostics.g1c * jnp.sin(theta) + diagnostics.g1s * jnp.cos(theta)
    quadratic = (
        diagnostics.g20
        + diagnostics.g2s * jnp.sin(2 * theta)
        + diagnostics.g2c * jnp.cos(2 * theta)
    )
    quadratic_prime = 2 * diagnostics.g2s * jnp.cos(2 * theta) - 2 * diagnostics.g2c * jnp.sin(
        2 * theta
    )
    np.testing.assert_allclose(
        diagnostics.g0 + radius * linear + radius**2 * quadratic, 0, atol=2e-13
    )
    np.testing.assert_allclose(radius * linear_prime + radius**2 * quadratic_prime, 0, atol=2e-13)


def test_newton_refinement_removes_angular_grid_dependence():
    """r_singularity from a 32-point and a 512-point theta scan agree to 2e-11 after Newton."""

    solution = solve_configuration("qa", nphi=31)
    coarse = qsc.singularity_diagnostics(solution, angular_resolution=32, newton_iterations=8)
    fine = qsc.singularity_diagnostics(solution, angular_resolution=512, newton_iterations=8)
    np.testing.assert_allclose(coarse.r_singularity, fine.r_singularity, rtol=0, atol=2e-11)
    assert coarse.maximum_residual_norm < 3e-13
    for kwargs, message in (
        ({"angular_resolution": 4}, "angular_resolution"),
        ({"newton_iterations": -1}, "newton_iterations"),
    ):
        with pytest.raises(ValueError, match=message):
            qsc.singularity_diagnostics(solution, **kwargs)


def test_r_singularity_gradient_matches_finite_differences():
    """The jitted gradient of the non-smooth min over varphi is finite and matches FD."""

    def r_singularity(etabar, B2c):
        return solve_configuration("qa", nphi=61, etabar=etabar, B2c=B2c).r_singularity

    point = (jnp.asarray(0.64), jnp.asarray(-0.00322))
    gradient = jax.jit(jax.grad(r_singularity, argnums=(0, 1)))(*point)
    step = 1e-6
    for index, value in enumerate(gradient):
        shift = [step if i == index else 0.0 for i in range(2)]
        plus = [p + s for p, s in zip(point, shift, strict=True)]
        minus = [p - s for p, s in zip(point, shift, strict=True)]
        finite_difference = (r_singularity(*plus) - r_singularity(*minus)) / (2 * step)
        assert np.isfinite(value)
        np.testing.assert_allclose(value, finite_difference, rtol=1e-6, atol=1e-9)

    legacy = near_axis(**CONFIGURATIONS["qa"], nphi=31, order="r2")
    np.testing.assert_allclose(legacy.r_singularity, legacy.solution.r_singularity)
    np.testing.assert_allclose(
        legacy.r_singularity_residual_sqnorm, legacy.solution.r_singularity_residual_sqnorm
    )


# --- Mercier ------------------------------------------------------------------------------


def test_vacuum_mercier_pressure_terms_vanish():
    """With p2 = I2 = 0 the geodesic, well and total Mercier terms are exactly zero."""

    solution = solve_configuration("qa", nphi=31)
    diagnostics = qsc.mercier_diagnostics(solution)
    assert diagnostics.DGeod_times_r2 == 0
    assert diagnostics.DWell_times_r2 == 0
    assert diagnostics.DMerc_times_r2 == 0


# --- B20 / B2c ----------------------------------------------------------------------------


def test_B20_diagnostics_are_consistent_norms():
    """weighted L2 == B20_residual <= smooth max <= grid max; peak-to-peak == B20_variation/B0."""

    solution = solve_configuration("qa", nphi=31)
    diagnostics = qsc.b20_diagnostics(solution, smooth_maximum_power=12)

    np.testing.assert_allclose(diagnostics.weighted_l2, solution.B20_residual)
    np.testing.assert_allclose(
        diagnostics.peak_to_peak, solution.B20_variation / solution.inputs.B0
    )
    np.testing.assert_allclose(diagnostics.weighted_mean, solution.B20_mean)
    assert diagnostics.weighted_l2 <= diagnostics.smooth_maximum <= diagnostics.grid_maximum
    assert diagnostics.fourier_coefficients.shape == (15,)
    assert 0 <= float(diagnostics.fourier_tail_ratio) <= 1
    for function, kwargs, message in (
        (qsc.b20_diagnostics, {"smooth_maximum_power": 1}, "smooth_maximum_power"),
        (qsc.optimal_B2c_value, {"degeneracy_tolerance": -1.0}, "degeneracy_tolerance"),
    ):
        with pytest.raises(ValueError, match=message):
            function(solution, **kwargs)


def test_affine_B2c_elimination_is_exact_and_stationary():
    """B20 is affine in B2c, so the closed-form optimum beats +-0.1 and has zero gradient."""

    solution = solve_configuration("qa", nphi=31)
    result = qsc.optimize_B2c(solution)

    assert result.affine_reconstruction_error < 5.0e-14
    assert not bool(result.degenerate)
    for B2c in (result.B2c_optimal - 0.1, result.B2c_optimal + 0.1, solution.inputs.B2c):
        residual = solve_configuration("qa", nphi=31, B2c=B2c).B20_residual
        assert result.diagnostics.weighted_l2 < residual

    optimum = qsc.optimal_B2c_value(solve_configuration("qa", nphi=15))
    derivative = jax.grad(
        lambda B2c: solve_configuration("qa", nphi=15, B2c=B2c).B20_residual ** 2
    )(optimum)
    assert abs(float(derivative)) < 2.0e-11


def test_B2c_optimum_derivative_matches_finite_difference():
    """d B2c*/d etabar by forward-mode autodiff equals a central difference."""

    def optimum(etabar):
        return qsc.optimal_B2c_value(solve_configuration("qa", nphi=15, etabar=etabar))

    tangent = jax.jvp(optimum, (jnp.asarray(0.64),), (jnp.asarray(1.0),))[1]
    np.testing.assert_allclose(tangent, central_difference(optimum, 0.64, 1e-5), rtol=2.0e-6)


def test_optimal_solution_recomputes_r3_without_stale_data():
    """Optimizing B2c on an r3 solution rebuilds the third order from the new second order."""

    original = solve_configuration("qa", nphi=15, order="r3")
    result = qsc.optimize_B2c(original)
    fresh = qsc.solve_third_order(replace(result.solution, third_order=None))

    np.testing.assert_allclose(
        result.solution.flux_constraint_coefficient, fresh.flux_constraint_coefficient, rtol=1e-12
    )
    assert not np.allclose(
        result.solution.flux_constraint_coefficient, original.flux_constraint_coefficient
    )


def test_circular_axis_B2c_response_is_degenerate():
    """On a circular axis B20 is constant, so the optimum is flagged degenerate and unchanged."""

    solution = qsc.Qsc(rc=[1.0], zs=[0.0], nfp=1, etabar=1.0, I2=0.1, B2c=0.2, nphi=15, order="r2")
    result = qsc.optimize_B2c(solution)

    assert bool(result.degenerate)
    np.testing.assert_allclose(result.B2c_optimal, 0.2)
    assert result.diagnostics.weighted_l2 < 2.0e-14


@pytest.mark.physics
def test_documented_optimized_axis_has_nearly_constant_B20():
    """The preserved eight-mode QH refinement of database 57409 has B20 variation < 6e-10."""

    stock = qsc.optimize_B2c(solve_configuration("database_low_b20_57409", nphi=121))
    optimized = solve_configuration("b20_optimized_good", nphi=121)
    diagnostics = qsc.b20_diagnostics(optimized)

    assert float(diagnostics.weighted_l2) < 1.4e-10
    assert float(diagnostics.grid_maximum) < 3.0e-10
    assert float(diagnostics.peak_to_peak) < 6.0e-10
    assert float(stock.diagnostics.weighted_l2 / diagnostics.weighted_l2) > 2.0e8
    assert abs(float(optimized.iota)) > 0.4

"""First-order (O(r)) near-axis solve: sigma equation, Maxwell identities, derivatives."""

import jax
import jax.numpy as jnp
import numpy as np

import pyqsc_jax as qsc
from pyqsc_jax.first_order import sigma_residual

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)


def iota_from_etabar(etabar):
    return qsc.solve(axis=AXIS, etabar=etabar, nphi=31).iota


def test_sigma_equation_is_solved_to_roundoff():
    """The Riccati sigma equation residual is < 2e-13 and matches the reported norm."""

    solution = qsc.solve(axis=AXIS, etabar=-0.9, nphi=31)
    state = solution.sigma.at[0].set(solution.iota)
    residual = sigma_residual(state, inputs=solution.inputs, geometry=solution.geometry)

    assert bool(solution.root_report.converged) and bool(solution.root_report.finite)
    assert solution.root_report.residual_norm < 2e-13
    np.testing.assert_allclose(
        jnp.max(jnp.abs(residual)), solution.root_report.residual_norm, rtol=1e-7, atol=1e-15
    )
    np.testing.assert_allclose(solution.sigma[0], solution.inputs.sigma0)


def test_first_order_flux_surface_area_constraint():
    """X1c Y1s - X1s Y1c = sG spsi (unit toroidal flux per area at O(r))."""

    solution = qsc.solve(axis=AXIS, etabar=-0.9, nphi=31)
    area_jacobian = solution.X1c * solution.Y1s - solution.X1s * solution.Y1c
    np.testing.assert_allclose(area_jacobian, solution.inputs.sG * solution.inputs.spsi, rtol=2e-13)


def test_vacuum_field_gradient_is_divergence_and_curl_free():
    """|B_axis| = B0 and grad B on axis is trace-free and symmetric to spectral accuracy."""

    solution = qsc.solve(axis=AXIS, etabar=-0.9, nphi=61)
    gradient = solution.grad_B_axis

    np.testing.assert_allclose(jnp.linalg.norm(solution.B_axis, axis=-1), solution.inputs.B0)
    assert jnp.max(jnp.abs(jnp.trace(gradient, axis1=-2, axis2=-1))) < 2e-8
    assert jnp.max(jnp.abs(gradient - jnp.swapaxes(gradient, -1, -2))) < 2e-8
    assert jnp.all(solution.L_grad_B > 0)


def test_iota_derivatives_match_finite_differences():
    """d iota/d etabar and d iota/d rc1 by autodiff agree with central differences; JVP == VJP."""

    etabar = -0.9
    step = 2e-5
    finite_difference = (iota_from_etabar(etabar + step) - iota_from_etabar(etabar - step)) / (
        2 * step
    )
    np.testing.assert_allclose(
        jax.grad(iota_from_etabar)(etabar), finite_difference, rtol=2e-7, atol=2e-9
    )
    _, jvp = jax.jvp(iota_from_etabar, (etabar,), (1.0,))
    (vjp,) = jax.vjp(iota_from_etabar, etabar)[1](jnp.asarray(1.0))
    np.testing.assert_allclose(jvp, vjp, rtol=2e-12, atol=2e-12)
    batched = jax.vmap(iota_from_etabar)(jnp.array([-0.95, -0.9, -0.85]))
    np.testing.assert_allclose(batched[1], iota_from_etabar(etabar), rtol=2e-13)

    def iota_from_rc1(rc1):
        axis = qsc.Axis(rc=jnp.array([1.0, rc1]), zs=AXIS.zs, nfp=AXIS.nfp)
        return qsc.solve(axis=axis, etabar=-0.9, nphi=31).iota

    step = 2e-6
    finite_difference = (iota_from_rc1(0.045 + step) - iota_from_rc1(0.045 - step)) / (2 * step)
    np.testing.assert_allclose(
        jax.grad(iota_from_rc1)(0.045), finite_difference, rtol=3e-6, atol=3e-8
    )

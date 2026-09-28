import jax
import jax.numpy as jnp
import numpy as np

import pyqsc_jax as qsc

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)


def iota_from_etabar(etabar):
    return qsc.solve(axis=AXIS, etabar=etabar, nphi=31).iota


def test_jit_vmap_jvp_and_vjp():
    etabar = -0.9
    eager = iota_from_etabar(etabar)
    compiled = jax.jit(iota_from_etabar)(etabar)
    np.testing.assert_allclose(compiled, eager, rtol=2e-13)

    batched = jax.vmap(iota_from_etabar)(jnp.array([-0.95, -0.9, -0.85]))
    assert batched.shape == (3,)
    assert jnp.all(jnp.isfinite(batched))

    _, jvp = jax.jvp(iota_from_etabar, (etabar,), (1.0,))
    _, pullback = jax.vjp(iota_from_etabar, etabar)
    (vjp,) = pullback(jnp.asarray(1.0))
    np.testing.assert_allclose(jvp, vjp, rtol=2e-12, atol=2e-12)


def test_iota_derivative_matches_finite_difference():
    etabar = -0.9
    derivative = jax.grad(iota_from_etabar)(etabar)
    step = 2e-5
    finite_difference = (iota_from_etabar(etabar + step) - iota_from_etabar(etabar - step)) / (
        2 * step
    )

    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-7, atol=2e-9)


def test_axis_coefficient_derivative_matches_finite_difference():
    def iota_from_rc1(rc1):
        axis = qsc.Axis(rc=jnp.array([1.0, rc1]), zs=AXIS.zs, nfp=AXIS.nfp)
        return qsc.solve(axis=axis, etabar=-0.9, nphi=31).iota

    rc1 = 0.045
    derivative = jax.grad(iota_from_rc1)(rc1)
    step = 2e-6
    finite_difference = (iota_from_rc1(rc1 + step) - iota_from_rc1(rc1 - step)) / (2 * step)

    np.testing.assert_allclose(derivative, finite_difference, rtol=3e-6, atol=3e-8)

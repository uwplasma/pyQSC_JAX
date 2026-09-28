import jax
import jax.numpy as jnp
import numpy as np

import pyqsc_jax as qsc


def b20_residual(etabar):
    return qsc.Qsc(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=etabar,
        B2c=-0.00322,
        nphi=15,
        order="r2",
    ).B20_residual


def test_second_order_jit_jvp_and_vjp():
    etabar = 0.64
    eager = b20_residual(etabar)
    np.testing.assert_allclose(jax.jit(b20_residual)(etabar), eager, rtol=2e-13)

    _, jvp = jax.jvp(b20_residual, (etabar,), (1.0,))
    _, pullback = jax.vjp(b20_residual, etabar)
    (vjp,) = pullback(jnp.asarray(1.0))
    np.testing.assert_allclose(jvp, vjp, rtol=3e-11, atol=3e-11)


def test_second_order_gradient_matches_finite_difference():
    etabar = 0.64
    derivative = jax.grad(b20_residual)(etabar)
    step = 2e-5
    finite_difference = (b20_residual(etabar + step) - b20_residual(etabar - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-6, atol=2e-8)


def test_second_order_vmap():
    values = jax.vmap(b20_residual)(jnp.asarray([0.61, 0.64, 0.67]))
    assert values.shape == (3,)
    assert jnp.all(jnp.isfinite(values))

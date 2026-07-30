import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyqsc_jax.spectral import (
    differentiate,
    differentiation_matrix,
    fourier_coefficients,
    fourier_interpolate,
    periodic_grid,
    periodic_integral,
)


@pytest.mark.parametrize("n", [8, 9, 31])
def test_differentiation_matrix_is_spectrally_exact(n):
    period = 3.7
    x = periodic_grid(n, period=period)
    fundamental = 2 * jnp.pi / period
    values = jnp.sin(2 * fundamental * x) + 0.2 * jnp.cos(3 * fundamental * x)
    expected = 2 * fundamental * jnp.cos(2 * fundamental * x) - 0.6 * fundamental * jnp.sin(
        3 * fundamental * x
    )

    derivative = differentiation_matrix(n, period=period) @ values
    np.testing.assert_allclose(derivative, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(differentiate(values, period=period), expected, rtol=2e-13)


@pytest.mark.parametrize("n", [8, 9])
def test_fourier_interpolation_even_and_odd(n):
    x_grid = periodic_grid(n)
    samples = jnp.sin(2 * x_grid) + 0.3 * jnp.cos(3 * x_grid)
    x = jnp.array([0.12, 1.23, 5.72])
    expected = jnp.sin(2 * x) + 0.3 * jnp.cos(3 * x)

    np.testing.assert_allclose(fourier_interpolate(samples, x), expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        fourier_interpolate(samples, x_grid), samples, rtol=2e-13, atol=2e-15
    )


def test_periodic_integral_and_coefficients():
    n = 17
    x = periodic_grid(n)
    values = 2.3 + jnp.cos(3 * x) - 0.4 * jnp.sin(5 * x)
    frequency, coefficients = fourier_coefficients(values)

    np.testing.assert_allclose(periodic_integral(values), 2.3 * 2 * jnp.pi, rtol=2e-14)
    np.testing.assert_allclose(coefficients[frequency == 0], 2.3, rtol=2e-14)
    np.testing.assert_allclose(jax.jit(differentiate)(values), differentiate(values), rtol=2e-14)


def test_periodic_grid_validation():
    with pytest.raises(ValueError, match="n >= 2"):
        periodic_grid(1)
    with pytest.raises(ValueError, match="n >= 2"):
        differentiation_matrix(1)
    with pytest.raises(ValueError, match="one-dimensional"):
        fourier_interpolate(jnp.ones((2, 3)), 0.2)


def test_complex_fourier_interpolation():
    x_grid = periodic_grid(9)
    samples = jnp.exp(2j * x_grid)
    x = jnp.array([0.17, 2.4])

    np.testing.assert_allclose(fourier_interpolate(samples, x), jnp.exp(2j * x), rtol=2e-13)

"""Periodic spectral operators on uniform grids."""

import jax.numpy as jnp
import numpy as np
import pytest

from pyqsc_jax.spectral import (
    differentiate,
    differentiation_matrix,
    fourier_coefficients,
    fourier_interpolate,
    periodic_antiderivative,
    periodic_grid,
    periodic_integral,
)


def test_differentiation_is_spectrally_exact_for_even_and_odd_grids():
    """Differentiation matrix and FFT derivative are exact for resolved trigonometric data."""

    period = 3.7
    fundamental = 2 * jnp.pi / period
    for n in (8, 9):
        x = periodic_grid(n, period=period)
        values = jnp.sin(2 * fundamental * x) + 0.2 * jnp.cos(3 * fundamental * x)
        expected = 2 * fundamental * jnp.cos(2 * fundamental * x) - 0.6 * fundamental * jnp.sin(
            3 * fundamental * x
        )
        derivative = differentiation_matrix(n, period=period) @ values
        np.testing.assert_allclose(derivative, expected, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(differentiate(values, period=period), expected, rtol=2e-13)


def test_fourier_interpolation_is_exact_for_real_and_complex_data():
    """Trigonometric interpolation reproduces resolved real and complex signals off-grid."""

    x = jnp.array([0.12, 1.23, 5.72])
    for n in (8, 9):
        x_grid = periodic_grid(n)
        samples = jnp.sin(2 * x_grid) + 0.3 * jnp.cos(3 * x_grid)
        expected = jnp.sin(2 * x) + 0.3 * jnp.cos(3 * x)
        np.testing.assert_allclose(
            fourier_interpolate(samples, x), expected, rtol=2e-13, atol=2e-13
        )
        np.testing.assert_allclose(
            fourier_interpolate(samples, x_grid), samples, rtol=2e-13, atol=2e-15
        )
    x_grid = periodic_grid(9)
    np.testing.assert_allclose(
        fourier_interpolate(jnp.exp(2j * x_grid), x), jnp.exp(2j * x), rtol=2e-13
    )


def test_periodic_integral_and_mean_coefficient():
    """The trapezoidal integral and zero-frequency coefficient recover the mean exactly."""

    x = periodic_grid(17)
    values = 2.3 + jnp.cos(3 * x) - 0.4 * jnp.sin(5 * x)
    frequency, coefficients = fourier_coefficients(values)

    np.testing.assert_allclose(periodic_integral(values), 2.3 * 2 * jnp.pi, rtol=2e-14)
    np.testing.assert_allclose(coefficients[frequency == 0], 2.3, rtol=2e-14)


def test_periodic_antiderivative_is_exact_with_linear_mean_and_nyquist():
    """Antiderivative keeps the linear mean part and maps the Nyquist mode to zero."""

    period = 2.1
    fundamental = 2 * jnp.pi / period
    for n in (8, 9):
        x = periodic_grid(n, period=period)
        values = 1.7 + jnp.cos(fundamental * x) - 0.4 * jnp.sin(3 * fundamental * x)
        expected = 1.7 * x + jnp.sin(fundamental * x) / fundamental
        expected = expected + 0.4 * (jnp.cos(3 * fundamental * x) - 1) / (3 * fundamental)
        antiderivative = periodic_antiderivative(values, period=period)
        np.testing.assert_allclose(antiderivative, expected, rtol=0, atol=2e-14)
    nyquist = jnp.cos(0.5 * 8 * fundamental * periodic_grid(8, period=period))
    np.testing.assert_allclose(
        periodic_antiderivative(nyquist, period=period), 0.0, rtol=0, atol=2e-15
    )


def test_spectral_input_guards():
    """Too-small grids and non-1D samples are rejected."""

    with pytest.raises(ValueError, match="n >= 2"):
        periodic_grid(1)
    with pytest.raises(ValueError, match="n >= 2"):
        differentiation_matrix(1)
    with pytest.raises(ValueError, match="one-dimensional"):
        fourier_interpolate(jnp.ones((2, 3)), 0.2)
    with pytest.raises(ValueError, match="one-dimensional"):
        periodic_antiderivative(jnp.ones((2, 3)))

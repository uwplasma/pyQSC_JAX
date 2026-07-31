"""Spectral operations on uniform periodic grids."""

from typing import Any

import jax
import jax.numpy as jnp

ArrayLike = Any


def periodic_grid(n: int, *, period: ArrayLike = 2 * jnp.pi) -> jax.Array:
    """Return ``n`` uniform points on ``[0, period)``."""

    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("A periodic grid requires an integer n >= 2.")
    return jnp.arange(n) * (jnp.asarray(period) / n)


def differentiation_matrix(n: int, *, period: ArrayLike = 2 * jnp.pi) -> jax.Array:
    """Return the first-derivative Fourier collocation matrix.

    The even and odd formulas follow the trigonometric differentiation
    matrices of Weideman and Reddy. Rows act on values sampled by
    :func:`periodic_grid`.
    """

    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("A differentiation matrix requires an integer n >= 2.")

    index = jnp.arange(n)
    difference = index[:, None] - index[None, :]
    angle = jnp.pi * difference / n
    sign = jnp.where(jnp.mod(difference, 2) == 0, 1.0, -1.0)
    off_diagonal = difference != 0
    if n % 2 == 0:
        entries = 0.5 * sign / jnp.tan(angle)
    else:
        entries = 0.5 * sign / jnp.sin(angle)
    dimensionless = jnp.where(off_diagonal, entries, 0.0)
    return dimensionless * (2 * jnp.pi / jnp.asarray(period))


def differentiate(values: ArrayLike, *, period: ArrayLike = 2 * jnp.pi) -> jax.Array:
    """Differentiate values along their final periodic axis."""

    values = jnp.asarray(values)
    matrix = differentiation_matrix(values.shape[-1], period=period)
    return jnp.einsum("jk,...k->...j", matrix, values)


def periodic_integral(values: ArrayLike, *, period: ArrayLike = 2 * jnp.pi) -> jax.Array:
    """Integrate uniform periodic samples along their final axis."""

    return jnp.asarray(period) * jnp.mean(jnp.asarray(values), axis=-1)


def fourier_coefficients(values: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Return integer frequencies and complex Fourier coefficients."""

    values = jnp.asarray(values)
    n = values.shape[-1]
    frequency = jnp.fft.fftfreq(n, d=1 / n)
    coefficients = jnp.fft.fft(values, axis=-1) / n
    return frequency, coefficients


def fourier_interpolate(
    values: ArrayLike, x: ArrayLike, *, period: ArrayLike = 2 * jnp.pi
) -> jax.Array:
    """Evaluate the trigonometric interpolant of one-dimensional real samples."""

    values = jnp.asarray(values)
    if values.ndim != 1:
        raise ValueError("fourier_interpolate currently accepts one-dimensional samples.")

    frequency, coefficients = fourier_coefficients(values)
    x = jnp.asarray(x)
    angle = 2 * jnp.pi * x / jnp.asarray(period)
    phase = jnp.exp(1j * angle[..., None] * frequency)
    if values.size % 2 == 0:
        nyquist = values.size // 2
        phase = phase.at[..., nyquist].set(jnp.cos(nyquist * angle))
    result = jnp.sum(coefficients * phase, axis=-1)
    if jnp.issubdtype(values.dtype, jnp.floating):
        return jnp.real(result)
    return result

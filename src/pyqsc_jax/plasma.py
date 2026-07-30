"""Surface-free plasma-current source and free-space field jets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.second_order import MU0

ArrayLike = Any


def _positive_scalar(value: ArrayLike, *, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim:
        raise ValueError(f"{name} must be a scalar.")
    try:
        if bool(array <= 0):
            raise ValueError(f"{name} must be positive.")
    except jax.errors.TracerBoolConversionError:
        pass
    return array


def enclosed_current_from_covariant(
    I2: ArrayLike,
    *,
    formal_radius: ArrayLike,
    chi: int,
) -> jax.Array:
    """Convert covariant ``I2`` to enclosed toroidal current in amperes."""

    radius = _positive_scalar(formal_radius, name="formal_radius")
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1.")
    return 2 * jnp.pi * chi * jnp.asarray(I2) * radius**2 / MU0


def covariant_current_from_enclosed(
    enclosed_current: ArrayLike,
    *,
    formal_radius: ArrayLike,
    chi: int,
) -> jax.Array:
    """Convert enclosed toroidal current in amperes to covariant ``I2``."""

    radius = _positive_scalar(formal_radius, name="formal_radius")
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1.")
    return MU0 * jnp.asarray(enclosed_current) / (2 * jnp.pi * chi * radius**2)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlasmaCurrentSource:
    """Positive-volume current measure through quadratic radial order.

    The vector coefficients follow

    ``W / L = r * w1 + r**2 * (w2_cosine*cos(theta) + w2_sine*sin(theta))``.
    """

    formal_radius: jax.Array
    parallel_current_mu0: jax.Array
    enclosed_toroidal_current: jax.Array
    C2: jax.Array
    beta_1s: jax.Array
    w1: jax.Array
    wstar2_cosine: jax.Array
    wstar2_sine: jax.Array
    w2_cosine: jax.Array
    w2_sine: jax.Array
    axis_length_per_radian: jax.Array
    chi: int = field(metadata={"static": True})


def plasma_current_source(
    solution: NearAxisSolution,
    *,
    formal_radius: ArrayLike,
) -> PlasmaCurrentSource:
    """Construct the exact positive-volume weighted source through ``O(r²)``."""

    if solution.second_order is None:
        raise ValueError("The plasma current source requires a second-order solution.")
    radius = _positive_scalar(formal_radius, name="formal_radius")
    inputs = solution.inputs
    geometry = solution.geometry
    chi = inputs.sG * inputs.spsi
    axis_length_per_radian = geometry.abs_G0_over_B0
    x = solution.X1c
    y = solution.Y1s
    y_sigma = solution.Y1c
    derivative = geometry.d_d_varphi
    current_density_mu0 = 2 * chi * inputs.I2
    C2 = solution.G2 + solution.N_helicity * inputs.I2

    cosine_tangent = -chi * inputs.spsi * inputs.B0 * solution.beta_1s
    cosine_normal = current_density_mu0 * (
        (derivative @ x) / axis_length_per_radian - geometry.torsion * y_sigma
    )
    cosine_binormal = (
        current_density_mu0
        * ((derivative @ y_sigma) / axis_length_per_radian + geometry.torsion * x)
        - 2 * chi * C2 * y / axis_length_per_radian
    )
    sine_normal = (
        -current_density_mu0 * geometry.torsion * y + 2 * chi * C2 * x / axis_length_per_radian
    )
    sine_binormal = (
        current_density_mu0 * (derivative @ y) / axis_length_per_radian
        + 2 * chi * C2 * y_sigma / axis_length_per_radian
    )

    tangent = geometry.tangent_cartesian
    normal = geometry.normal_cartesian
    binormal = geometry.binormal_cartesian
    w1 = current_density_mu0 * tangent
    wstar2_cosine = (
        cosine_tangent * tangent
        + cosine_normal[:, None] * normal
        + cosine_binormal[:, None] * binormal
    )
    wstar2_sine = sine_normal[:, None] * normal + sine_binormal[:, None] * binormal
    w2_cosine = wstar2_cosine - (current_density_mu0 * geometry.curvature * x)[:, None] * tangent
    return PlasmaCurrentSource(
        formal_radius=radius,
        parallel_current_mu0=current_density_mu0,
        enclosed_toroidal_current=enclosed_current_from_covariant(
            inputs.I2,
            formal_radius=radius,
            chi=chi,
        ),
        C2=C2,
        beta_1s=solution.beta_1s,
        w1=w1,
        wstar2_cosine=wstar2_cosine,
        wstar2_sine=wstar2_sine,
        w2_cosine=w2_cosine,
        w2_sine=wstar2_sine,
        axis_length_per_radian=axis_length_per_radian,
        chi=chi,
    )


def evaluate_weighted_current(
    source: PlasmaCurrentSource,
    radial_coordinate: ArrayLike,
    theta: ArrayLike,
) -> jax.Array:
    """Evaluate ``W = chi * mu0 * J_r * J`` at all toroidal samples."""

    radial_coordinate, theta = jnp.broadcast_arrays(
        jnp.asarray(radial_coordinate),
        jnp.asarray(theta),
    )
    radial = radial_coordinate[..., None, None]
    cosine = jnp.cos(theta)[..., None, None]
    sine = jnp.sin(theta)[..., None, None]
    return source.axis_length_per_radian * (
        radial * source.w1 + radial**2 * (cosine * source.w2_cosine + sine * source.w2_sine)
    )

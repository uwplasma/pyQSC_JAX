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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlasmaFieldData:
    """Matched on-axis free-space plasma field and error metadata."""

    field: jax.Array
    regularized_axis_integral: jax.Array
    matched_axis_and_core: jax.Array
    second_order_shape_correction: jax.Array
    core_binormal_constant: jax.Array
    core_normal_constant: jax.Array
    maximum_matching_scale_error: jax.Array
    formal_radius_to_curvature_radius: jax.Array
    estimated_field_remainder: jax.Array
    current_source: PlasmaCurrentSource
    angular_resolution: int = field(metadata={"static": True})


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


def _full_torus_axis_samples(
    solution: NearAxisSolution,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    geometry = solution.geometry
    nfp = solution.inputs.axis.nfp
    cylindrical_period = 2 * jnp.pi / nfp
    boozer_period = 2 * jnp.pi / nfp
    period_index = jnp.arange(nfp)
    full_phi = (geometry.samples.phi[None, :] + period_index[:, None] * cylindrical_period).reshape(
        -1
    )
    full_varphi = (geometry.varphi[None, :] + period_index[:, None] * boozer_period).reshape(-1)
    radius = jnp.tile(geometry.samples.R, nfp)
    height = jnp.tile(geometry.samples.Z, nfp)
    position = jnp.stack(
        (
            radius * jnp.cos(full_phi),
            radius * jnp.sin(full_phi),
            height,
        ),
        axis=-1,
    )
    tangent_cylindrical = jnp.tile(
        geometry.tangent_cylindrical,
        (nfp, 1),
    )
    tangent = jnp.stack(
        (
            tangent_cylindrical[:, 0] * jnp.cos(full_phi)
            - tangent_cylindrical[:, 1] * jnp.sin(full_phi),
            tangent_cylindrical[:, 0] * jnp.sin(full_phi)
            + tangent_cylindrical[:, 1] * jnp.cos(full_phi),
            tangent_cylindrical[:, 2],
        ),
        axis=-1,
    )
    d_phi = cylindrical_period / solution.inputs.nphi
    weights = jnp.tile(geometry.d_varphi_d_phi * d_phi, nfp)
    return full_varphi, position, tangent, weights


def regularized_axis_integral(solution: NearAxisSolution) -> jax.Array:
    """Evaluate the full-torus periodic finite-part integral in Eq. (136)."""

    source_varphi, source_position, source_tangent, weights = _full_torus_axis_samples(solution)
    observation_varphi = solution.varphi
    observation_position = solution.geometry.position_cartesian
    displacement = observation_position[:, None, :] - source_position[None, :, :]
    distance_squared = jnp.sum(displacement**2, axis=-1)
    angle = source_varphi[None, :] - observation_varphi[:, None]
    sine_half = jnp.sin(0.5 * angle)
    coincident = jnp.abs(sine_half) < 16 * jnp.finfo(angle.dtype).eps
    safe_distance_squared = jnp.where(coincident, 1.0, distance_squared)
    safe_sine = jnp.where(coincident, 1.0, jnp.abs(sine_half))
    filament = (
        solution.geometry.abs_G0_over_B0
        * jnp.cross(
            source_tangent[None, :, :],
            displacement,
        )
        / safe_distance_squared[..., None] ** 1.5
    )
    singular_model = (
        solution.geometry.curvature[:, None, None]
        * solution.geometry.binormal_cartesian[:, None, :]
        / (4 * safe_sine[..., None])
    )
    integrand = jnp.where(
        coincident[..., None],
        0.0,
        filament - singular_model,
    )
    return jnp.sum(weights[None, :, None] * integrand, axis=1)


def _second_order_shape_correction(
    solution: NearAxisSolution,
    source: PlasmaCurrentSource,
    *,
    angular_resolution: int,
) -> jax.Array:
    if (
        not isinstance(angular_resolution, int)
        or isinstance(angular_resolution, bool)
        or angular_resolution < 8
    ):
        raise ValueError("angular_resolution must be an integer >= 8.")
    theta = 2 * jnp.pi * jnp.arange(angular_resolution) / angular_resolution
    cosine = jnp.cos(theta)[None, :, None]
    sine = jnp.sin(theta)[None, :, None]
    cosine2 = jnp.cos(2 * theta)[None, :, None]
    sine2 = jnp.sin(2 * theta)[None, :, None]
    geometry = solution.geometry
    tangent = geometry.tangent_cartesian[:, None, :]
    normal = geometry.normal_cartesian[:, None, :]
    binormal = geometry.binormal_cartesian[:, None, :]
    e = (
        solution.X1c[:, None, None] * cosine * normal
        + (solution.Y1s[:, None, None] * sine + solution.Y1c[:, None, None] * cosine) * binormal
    )
    X2 = (
        solution.X20[:, None, None]
        + solution.X2c[:, None, None] * cosine2
        + solution.X2s[:, None, None] * sine2
    )
    Y2 = (
        solution.Y20[:, None, None]
        + solution.Y2c[:, None, None] * cosine2
        + solution.Y2s[:, None, None] * sine2
    )
    Z2 = (
        solution.Z20[:, None, None]
        + solution.Z2c[:, None, None] * cosine2
        + solution.Z2s[:, None, None] * sine2
    )
    xi2 = X2 * normal + Y2 * binormal + Z2 * tangent
    w1 = source.w1[:, None, :]
    wstar2 = source.wstar2_cosine[:, None, :] * cosine + source.wstar2_sine[:, None, :] * sine
    E2 = jnp.sum(e**2, axis=-1, keepdims=True)
    integrand = (jnp.cross(w1, xi2) + jnp.cross(wstar2, e)) / E2 - 2 * jnp.sum(
        e * xi2, axis=-1, keepdims=True
    ) * jnp.cross(w1, e) / E2**2
    return -0.5 * source.formal_radius**2 * jnp.mean(integrand, axis=1)


def matched_plasma_field_kernel(
    solution: NearAxisSolution,
    source: PlasmaCurrentSource,
    regularized_integral: jax.Array,
    *,
    reference_length: ArrayLike,
) -> jax.Array:
    """Combine finite-part and local-core terms at an arbitrary matching length."""

    reference_length = _positive_scalar(
        reference_length,
        name="reference_length",
    )
    geometry = solution.geometry
    axis_scale = geometry.abs_G0_over_B0
    x = solution.X1c
    sigma = solution.Y1c / solution.Y1s
    trace_Q = x**2 + (1 + sigma**2) / x**2
    core_binormal = -0.5 - 0.5 * jnp.log((trace_Q + 2) / 4) + (x**2 + 1) / (trace_Q + 2)
    core_normal = -source.chi * sigma / (trace_Q + 2)
    curvature_binormal = geometry.curvature[:, None] * geometry.binormal_cartesian
    finite_part = regularized_integral - curvature_binormal * jnp.log(
        reference_length / (4 * axis_scale)
    )
    return (
        finite_part
        + curvature_binormal
        * (jnp.log(2 * reference_length / source.formal_radius) + core_binormal[:, None])
        + (geometry.curvature * core_normal)[:, None] * geometry.normal_cartesian
    )


def plasma_field_on_axis(
    solution: NearAxisSolution,
    *,
    formal_radius: ArrayLike,
    angular_resolution: int = 128,
) -> PlasmaFieldData:
    """Evaluate the matched leading on-axis free-space plasma field."""

    source = plasma_current_source(
        solution,
        formal_radius=formal_radius,
    )
    regularized_integral = regularized_axis_integral(solution)
    axis_scale = solution.geometry.abs_G0_over_B0
    matched = matched_plasma_field_kernel(
        solution,
        source,
        regularized_integral,
        reference_length=4 * axis_scale,
    )
    independent_matching_scale = matched_plasma_field_kernel(
        solution,
        source,
        regularized_integral,
        reference_length=7 * axis_scale,
    )
    shape_correction = _second_order_shape_correction(
        solution,
        source,
        angular_resolution=angular_resolution,
    )
    current_prefactor = source.parallel_current_mu0 * source.formal_radius**2 / 4
    field_value = current_prefactor * matched + shape_correction
    x = solution.X1c
    sigma = solution.Y1c / solution.Y1s
    trace_Q = x**2 + (1 + sigma**2) / x**2
    core_binormal = -0.5 - 0.5 * jnp.log((trace_Q + 2) / 4) + (x**2 + 1) / (trace_Q + 2)
    core_normal = -source.chi * sigma / (trace_Q + 2)
    radius_to_curvature = source.formal_radius * jnp.max(solution.geometry.curvature)
    logarithm = jnp.abs(jnp.log(source.formal_radius / solution.geometry.abs_G0_over_B0))
    estimated_remainder = (
        jnp.abs(source.parallel_current_mu0)
        * source.formal_radius**4
        / solution.geometry.abs_G0_over_B0**3
        * (1 + logarithm)
    )
    return PlasmaFieldData(
        field=field_value,
        regularized_axis_integral=regularized_integral,
        matched_axis_and_core=matched,
        second_order_shape_correction=shape_correction,
        core_binormal_constant=core_binormal,
        core_normal_constant=core_normal,
        maximum_matching_scale_error=jnp.max(jnp.abs(matched - independent_matching_scale)),
        formal_radius_to_curvature_radius=radius_to_curvature,
        estimated_field_remainder=estimated_remainder,
        current_source=source,
        angular_resolution=angular_resolution,
    )

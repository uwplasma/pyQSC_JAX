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
    I2: ArrayLike, *, formal_radius: ArrayLike, chi: int
) -> jax.Array:
    """Convert covariant ``I2`` to enclosed toroidal current in amperes."""

    radius = _positive_scalar(formal_radius, name="formal_radius")
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1.")
    return 2 * jnp.pi * chi * jnp.asarray(I2) * radius**2 / MU0


def covariant_current_from_enclosed(
    enclosed_current: ArrayLike, *, formal_radius: ArrayLike, chi: int
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
    """Matched on-axis free-space plasma field and error metadata.

    ``estimated_field_remainder`` is ``max|B_p| (a / L_*)**2 (1 + |log(a / L_*)|)`` with
    ``L_* = |G0| / B0``: an indicator of the omitted relative order, not a certified bound.
    """

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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlasmaGradientData:
    """Local plasma gradient and external vacuum value/gradient target."""

    field: PlasmaFieldData
    gradient: jax.Array
    gradient_frenet: jax.Array
    external_field: jax.Array
    external_gradient: jax.Array
    external_gradient_frenet: jax.Array
    external_gradient_stf: jax.Array
    external_gradient_independent: jax.Array
    maximum_divergence: jax.Array
    maximum_ampere_error: jax.Array
    maximum_external_asymmetry: jax.Array
    maximum_external_trace: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlasmaHessianData:
    """Complete local plasma Hessian and external 3+5+7 vacuum target."""

    field: PlasmaGradientData
    hessian: jax.Array
    hessian_frenet: jax.Array
    external_hessian: jax.Array
    external_hessian_stf: jax.Array
    external_hessian_independent: jax.Array
    maximum_derivative_asymmetry: jax.Array
    maximum_external_symmetry_error: jax.Array
    maximum_external_trace: jax.Array
    estimated_hessian_remainder: jax.Array


def plasma_current_source(
    solution: NearAxisSolution, *, formal_radius: ArrayLike
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
            inputs.I2, formal_radius=radius, chi=chi
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
    source: PlasmaCurrentSource, radial_coordinate: ArrayLike, theta: ArrayLike
) -> jax.Array:
    """Evaluate ``W = chi * mu0 * J_r * J`` at all toroidal samples."""

    radial_coordinate, theta = jnp.broadcast_arrays(
        jnp.asarray(radial_coordinate), jnp.asarray(theta)
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
    position = jnp.stack((radius * jnp.cos(full_phi), radius * jnp.sin(full_phi), height), axis=-1)
    tangent_cylindrical = jnp.tile(geometry.tangent_cylindrical, (nfp, 1))
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
        * jnp.cross(source_tangent[None, :, :], displacement)
        / safe_distance_squared[..., None] ** 1.5
    )
    singular_model = (
        solution.geometry.curvature[:, None, None]
        * solution.geometry.binormal_cartesian[:, None, :]
        / (4 * safe_sine[..., None])
    )
    integrand = jnp.where(coincident[..., None], 0.0, filament - singular_model)
    return jnp.sum(weights[None, :, None] * integrand, axis=1)


def _second_order_shape_correction(
    solution: NearAxisSolution, source: PlasmaCurrentSource, *, angular_resolution: int
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

    reference_length = _positive_scalar(reference_length, name="reference_length")
    geometry = solution.geometry
    axis_scale = geometry.abs_G0_over_B0
    x = solution.X1c
    sigma = solution.sigma
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
    solution: NearAxisSolution, *, formal_radius: ArrayLike, angular_resolution: int = 128
) -> PlasmaFieldData:
    """Evaluate the matched leading on-axis free-space plasma field."""

    source = plasma_current_source(solution, formal_radius=formal_radius)
    regularized_integral = regularized_axis_integral(solution)
    axis_scale = solution.geometry.abs_G0_over_B0
    matched = matched_plasma_field_kernel(
        solution, source, regularized_integral, reference_length=4 * axis_scale
    )
    independent_matching_scale = matched_plasma_field_kernel(
        solution, source, regularized_integral, reference_length=7 * axis_scale
    )
    shape_correction = _second_order_shape_correction(
        solution, source, angular_resolution=angular_resolution
    )
    current_prefactor = source.parallel_current_mu0 * source.formal_radius**2 / 4
    field_value = current_prefactor * matched + shape_correction
    x = solution.X1c
    sigma = solution.sigma
    trace_Q = x**2 + (1 + sigma**2) / x**2
    core_binormal = -0.5 - 0.5 * jnp.log((trace_Q + 2) / 4) + (x**2 + 1) / (trace_Q + 2)
    core_normal = -source.chi * sigma / (trace_Q + 2)
    radius_to_curvature = source.formal_radius * jnp.max(solution.geometry.curvature)
    logarithm = jnp.abs(jnp.log(source.formal_radius / solution.geometry.abs_G0_over_B0))
    # An order-of-magnitude indicator of the omitted relative order a**2 (with its logarithm), not a
    # bound. It scales the whole retained field, so the pressure-driven field at I2 == 0 is covered.
    estimated_remainder = (
        jnp.max(jnp.linalg.norm(field_value, axis=-1))
        * (source.formal_radius / solution.geometry.abs_G0_over_B0) ** 2
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


def project_symmetric_trace_free_rank2(tensor: ArrayLike) -> jax.Array:
    """Project final two axes onto symmetric trace-free rank-two tensors."""

    tensor = jnp.asarray(tensor)
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("A rank-two Cartesian tensor must end in shape (3, 3).")
    symmetric = 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))
    trace = jnp.trace(symmetric, axis1=-2, axis2=-1)
    return symmetric - trace[..., None, None] * jnp.eye(3, dtype=tensor.dtype) / 3


def pack_symmetric_trace_free_rank2(tensor: ArrayLike) -> jax.Array:
    """Pack an STF matrix as ``(xx, yy, xy, xz, yz)``."""

    tensor = project_symmetric_trace_free_rank2(tensor)
    return jnp.stack(
        (
            tensor[..., 0, 0],
            tensor[..., 1, 1],
            tensor[..., 0, 1],
            tensor[..., 0, 2],
            tensor[..., 1, 2],
        ),
        axis=-1,
    )


def unpack_symmetric_trace_free_rank2(components: ArrayLike) -> jax.Array:
    """Unpack ``(xx, yy, xy, xz, yz)`` into an STF matrix."""

    components = jnp.asarray(components)
    if components.shape[-1:] != (5,):
        raise ValueError("STF rank-two components must end in length 5.")
    xx, yy, xy, xz, yz = jnp.moveaxis(components, -1, 0)
    return jnp.stack(
        (
            jnp.stack((xx, xy, xz), axis=-1),
            jnp.stack((xy, yy, yz), axis=-1),
            jnp.stack((xz, yz, -xx - yy), axis=-1),
        ),
        axis=-2,
    )


def elliptical_channel_gradient(
    x: ArrayLike,
    sigma: ArrayLike,
    *,
    parallel_current_mu0: ArrayLike,
    chi: int,
    frame: ArrayLike | None = None,
) -> jax.Array:
    """Return the leading field-component-first gradient of an elliptical channel."""

    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1.")
    x = jnp.asarray(x)
    sigma = jnp.asarray(sigma)
    parallel_current_mu0 = jnp.asarray(parallel_current_mu0)
    trace_Q = x**2 + (1 + sigma**2) / x**2
    derivative_first = (
        parallel_current_mu0[..., None, None]
        / (trace_Q + 2)[..., None, None]
        * jnp.stack(
            (
                jnp.stack((jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)), axis=-1),
                jnp.stack((jnp.zeros_like(x), chi * sigma, 1 + (1 + sigma**2) / x**2), axis=-1),
                jnp.stack((jnp.zeros_like(x), -(1 + x**2), -chi * sigma), axis=-1),
            ),
            axis=-2,
        )
    )
    field_first_frenet = jnp.swapaxes(derivative_first, -1, -2)
    if frame is None:
        return field_first_frenet
    frame = jnp.asarray(frame)
    if frame.shape[-2:] != (3, 3):
        raise ValueError("frame must end in shape (3, 3).")
    return jnp.einsum("...ai,...ab,...bj->...ij", frame, field_first_frenet, frame)


def zero_current_gradient(
    solution: NearAxisSolution, plasma_field_frenet: ArrayLike, *, formal_radius: ArrayLike
) -> jax.Array:
    """Return the first nonzero, order-``a**2`` plasma gradient of the ``I2 = 0`` branch.

    The uniform-channel gradient is proportional to the on-axis current and vanishes
    here. Pressure-driven current away from the axis still produces a symmetric,
    trace-free gradient, fixed by the second-order shaping and the quadratic current
    harmonic. The result is field-component-first in the ``(t, n, b)`` frame and is
    only valid for a stellarator-symmetric solution with ``I2 = 0``.

    ``plasma_field_frenet`` holds the ``(t, n, b)`` components of the matched on-axis
    field, whose derivative along the axis is the tangential row.
    """

    if solution.second_order is None:
        raise ValueError("The zero-current gradient requires a second-order solution.")
    inputs, geometry, second = solution.inputs, solution.geometry, solution.second_order
    radius = _positive_scalar(formal_radius, name="formal_radius")
    sG, spsi = inputs.sG, inputs.spsi
    chi = sG * spsi
    axis_scale = geometry.abs_G0_over_B0
    pressure = MU0 * inputs.p2 / inputs.B0
    curvature, torsion = geometry.curvature, geometry.torsion
    x = solution.X1c
    sigma = solution.sigma
    d_ds = geometry.d_d_varphi / axis_scale
    x_s, sigma_s = d_ds @ x, d_ds @ sigma
    w = 1 + x**2 + 1j * chi * sigma
    w_s = 2 * x * x_s + 1j * chi * sigma_s

    f_t = 4 * spsi * pressure * axis_scale * inputs.etabar * x / (solution.iotaN * w)
    f_n = 2j * sG * pressure * x**2 / w
    f_b = 2 * sG * pressure * (1 + 1j * chi * sigma) / w
    f_n_s = 2j * sG * pressure * (2 * x * x_s * w - x**2 * w_s) / w**2
    f_b_s = 2 * sG * pressure * (1j * chi * sigma_s * w - (1 + 1j * chi * sigma) * w_s) / w**2
    shape = second.X2c + chi * second.Y2s + 1j * (second.Y2c - chi * second.X2s)
    current_harmonic = 10 * inputs.etabar**2 - 8 * inputs.B2c / inputs.B0
    q_t = (
        x**2
        / (2 * w**2)
        * (
            spsi * pressure * (8 * second.Z2s - axis_scale / solution.iotaN * current_harmonic)
            + 8j * sG * pressure * second.Z2c
            - 4 * f_t * shape
        )
    )
    curved = curvature * x**2 * f_t / (4 * w)
    g_nn = radius**2 * (
        -(q_t + curvature * f_t / 4 + curved).imag
        - f_b_s.real / 2
        - torsion * (f_n.real + f_b.imag) / 2
    )
    g_nb = radius**2 * (
        f_n_s.real / 2 - torsion * f_b.real / 2 + torsion * f_n.imag / 2 - (q_t + curved).real
    )

    b_t, b_n, b_b = jnp.moveaxis(jnp.asarray(plasma_field_frenet), -1, 0)
    g_tt = d_ds @ b_t - curvature * b_n
    g_tn = d_ds @ b_n + curvature * b_t - torsion * b_b
    g_tb = d_ds @ b_b + torsion * b_n
    return jnp.stack(
        (
            jnp.stack((g_tt, g_tn, g_tb), axis=-1),
            jnp.stack((g_tn, g_nn, g_nb), axis=-1),
            jnp.stack((g_tb, g_nb, -g_tt - g_nn), axis=-1),
        ),
        axis=-2,
    )


def plasma_gradient_on_axis(
    solution: NearAxisSolution, *, formal_radius: ArrayLike, angular_resolution: int = 128
) -> PlasmaGradientData:
    """Evaluate the local plasma gradient and subtract it from the total jet.

    For ``I2 != 0`` this is the leading uniform-channel gradient. For ``I2 == 0`` that
    term vanishes and the first nonzero, order-``a**2`` gradient is returned instead.

    The switch at exactly ``I2 == 0`` changes the retained order, so the gradient jumps by
    the order-``a**2`` term as ``I2 -> 0``. That jump is a truncation artefact, not physics:
    hold ``I2`` fixed when optimizing, and do not differentiate with respect to ``I2`` near
    zero. A uniform order-``a**2`` finite-current gradient needs higher-order equilibrium terms.
    """

    plasma_field = plasma_field_on_axis(
        solution, formal_radius=formal_radius, angular_resolution=angular_resolution
    )
    x = solution.X1c
    sigma = solution.sigma
    frame = solution.geometry.frenet_frame
    gradient_frenet = elliptical_channel_gradient(
        x,
        sigma,
        parallel_current_mu0=plasma_field.current_source.parallel_current_mu0,
        chi=plasma_field.current_source.chi,
    )
    # The two branches have different asymptotic orders, so they are selected, not summed.
    gradient_frenet = jnp.where(
        solution.inputs.I2 == 0,
        zero_current_gradient(
            solution,
            jnp.einsum("...ai,...i->...a", frame, plasma_field.field),
            formal_radius=formal_radius,
        ),
        gradient_frenet,
    )
    gradient = jnp.einsum("...ai,...ab,...bj->...ij", frame, gradient_frenet, frame)
    external_field = solution.B_axis - plasma_field.field
    external_gradient = solution.grad_B_axis - gradient
    external_gradient_frenet = jnp.einsum(
        "...ai,...ij,...bj->...ab", frame, external_gradient, frame
    )
    external_gradient_stf = project_symmetric_trace_free_rank2(external_gradient)
    plasma_divergence = jnp.trace(gradient, axis1=-2, axis2=-1)
    ampere = (
        gradient_frenet[:, 2, 1]
        - gradient_frenet[:, 1, 2]
        - plasma_field.current_source.parallel_current_mu0
    )
    external_asymmetry = external_gradient - jnp.swapaxes(external_gradient, -1, -2)
    external_trace = jnp.trace(external_gradient, axis1=-2, axis2=-1)
    return PlasmaGradientData(
        field=plasma_field,
        gradient=gradient,
        gradient_frenet=gradient_frenet,
        external_field=external_field,
        external_gradient=external_gradient,
        external_gradient_frenet=external_gradient_frenet,
        external_gradient_stf=external_gradient_stf,
        external_gradient_independent=pack_symmetric_trace_free_rank2(external_gradient),
        maximum_divergence=jnp.max(jnp.abs(plasma_divergence)),
        maximum_ampere_error=jnp.max(jnp.abs(ampere)),
        maximum_external_asymmetry=jnp.max(jnp.abs(external_asymmetry)),
        maximum_external_trace=jnp.max(jnp.abs(external_trace)),
    )


def _symmetrize_rank3(tensor: ArrayLike) -> jax.Array:
    """Symmetrize the final three axes of a Cartesian tensor."""

    tensor = jnp.asarray(tensor)
    if tensor.shape[-3:] != (3, 3, 3):
        raise ValueError("A rank-three Cartesian tensor must end in shape (3, 3, 3).")
    return (
        tensor
        + jnp.transpose(tensor, (*range(tensor.ndim - 3), -3, -1, -2))
        + jnp.transpose(tensor, (*range(tensor.ndim - 3), -2, -3, -1))
        + jnp.transpose(tensor, (*range(tensor.ndim - 3), -2, -1, -3))
        + jnp.transpose(tensor, (*range(tensor.ndim - 3), -1, -3, -2))
        + jnp.transpose(tensor, (*range(tensor.ndim - 3), -1, -2, -3))
    ) / 6


def project_symmetric_trace_free_rank3(tensor: ArrayLike) -> jax.Array:
    """Project final three axes onto fully symmetric trace-free rank three."""

    tensor = jnp.asarray(tensor)
    symmetric = _symmetrize_rank3(tensor)
    trace = jnp.einsum("...iik->...k", symmetric)
    identity = jnp.eye(3, dtype=tensor.dtype)
    correction = (
        jnp.einsum("ij,...k->...ijk", identity, trace)
        + jnp.einsum("ik,...j->...ijk", identity, trace)
        + jnp.einsum("jk,...i->...ijk", identity, trace)
    ) / 5
    return symmetric - correction


def pack_symmetric_trace_free_rank3(tensor: ArrayLike) -> jax.Array:
    """Pack an STF rank-three tensor as seven Cartesian components."""

    tensor = project_symmetric_trace_free_rank3(tensor)
    return jnp.stack(
        (
            tensor[..., 0, 0, 0],
            tensor[..., 0, 0, 1],
            tensor[..., 0, 0, 2],
            tensor[..., 0, 1, 1],
            tensor[..., 0, 1, 2],
            tensor[..., 1, 1, 1],
            tensor[..., 1, 1, 2],
        ),
        axis=-1,
    )


def _rank3_stf_basis(dtype: jnp.dtype) -> jax.Array:
    basis = jnp.zeros((7, 3, 3, 3), dtype=dtype)

    def set_symmetric(array, basis_index, indices, value):
        from itertools import permutations

        for permutation in set(permutations(indices)):
            array = array.at[basis_index, permutation[0], permutation[1], permutation[2]].set(value)
        return array

    basis = set_symmetric(basis, 0, (0, 0, 0), 1)
    basis = set_symmetric(basis, 0, (0, 2, 2), -1)
    basis = set_symmetric(basis, 1, (0, 0, 1), 1)
    basis = set_symmetric(basis, 1, (1, 2, 2), -1)
    basis = set_symmetric(basis, 2, (0, 0, 2), 1)
    basis = set_symmetric(basis, 2, (2, 2, 2), -1)
    basis = set_symmetric(basis, 3, (0, 1, 1), 1)
    basis = set_symmetric(basis, 3, (0, 2, 2), -1)
    basis = set_symmetric(basis, 4, (0, 1, 2), 1)
    basis = set_symmetric(basis, 5, (1, 1, 1), 1)
    basis = set_symmetric(basis, 5, (1, 2, 2), -1)
    basis = set_symmetric(basis, 6, (1, 1, 2), 1)
    basis = set_symmetric(basis, 6, (2, 2, 2), -1)
    return basis


def unpack_symmetric_trace_free_rank3(components: ArrayLike) -> jax.Array:
    """Unpack ``(xxx,xxy,xxz,xyy,xyz,yyy,yyz)`` into an STF tensor."""

    components = jnp.asarray(components)
    if components.shape[-1:] != (7,):
        raise ValueError("STF rank-three components must end in length 7.")
    return jnp.einsum("...a,aijk->...ijk", components, _rank3_stf_basis(components.dtype))


def _cubic_potential(alpha_n: ArrayLike, alpha_b: ArrayLike, h_c: jax.Array, h_s: jax.Array):
    """Cubic interior logarithmic potential of the affine density ``alpha_n*u + alpha_b*v``.

    Returns the coefficients of ``(u**3, u**2*v, u*v**2, v**3)`` on the last axis, where
    ``u`` and ``v`` are physical normal and binormal distances. Poisson's equation fixes
    two combinations and matching to the decaying exterior potential of the filled
    ellipse fixes the harmonic remainder. ``h_c`` and ``h_s`` are the two components of
    the second angular anisotropy of the ellipse, so nothing is diagonalized and the
    result stays differentiable through a circular section.
    """

    return (
        jnp.stack(
            (
                ((3 - 2 * h_c - h_c**2 + h_s**2) * alpha_n - 2 * h_s * (1 - h_c) * alpha_b) / 12,
                (2 * h_s * (1 + h_c) * alpha_n + (1 - 2 * h_c + h_c**2 - h_s**2) * alpha_b) / 4,
                ((1 + 2 * h_c + h_c**2 - h_s**2) * alpha_n + 2 * h_s * (1 - h_c) * alpha_b) / 4,
                (-2 * h_s * (1 + h_c) * alpha_n + (3 + 2 * h_c - h_c**2 + h_s**2) * alpha_b) / 12,
            ),
            axis=-1,
        )
        * jnp.pi
    )


def _transverse_plasma_hessian(
    solution: NearAxisSolution, source: PlasmaCurrentSource
) -> jax.Array:
    """Leading transverse plasma Hessian ``H[sample, alpha, beta, gamma]``.

    ``alpha`` and ``beta`` are normal/binormal displacement directions and ``gamma`` is
    the ``(t, n, b)`` field component. The cubic vector potential has three sources: the
    affine weighted current, the quadratic deformation of the section, and curvature.
    The last two are proportional to the on-axis current.
    """

    second, geometry = solution.second_order, solution.geometry
    chi, current, curvature = source.chi, source.parallel_current_mu0, geometry.curvature
    x = solution.X1c
    sigma = solution.sigma
    denominator = (1 + x**2) ** 2 + sigma**2
    h_c = (x**4 - 1 - sigma**2) / denominator
    h_s = -2 * chi * sigma * x**2 / denominator

    # Affine current c*cos + s*sin in physical distances, using q1 = u/x, q2 = v/y - sigma*u/x.
    frame = geometry.frenet_frame
    cosine = jnp.einsum("...ai,...i->...a", frame, source.wstar2_cosine)
    sine = jnp.einsum("...ai,...i->...a", frame, source.wstar2_sine)
    alpha_n = (cosine - sigma[:, None] * sine) / x[:, None]
    alpha_b = chi * x[:, None] * sine
    potential = -_cubic_potential(alpha_n, alpha_b, h_c[:, None], h_s[:, None]) / (2 * jnp.pi)

    shape_n = 2 / x**2 * (
        (1 + sigma**2) * second.X20 + (1 - sigma**2) * second.X2c - 2 * sigma * second.X2s
    ) + 2 * chi * (second.Y2s - sigma * second.Y20 + sigma * second.Y2c)
    shape_b = 2 * chi * (second.X2s - sigma * second.X20 + sigma * second.X2c) + 2 * x**2 * (
        second.Y20 - second.Y2c
    )
    first = second.X2c + sigma * second.X2s - chi * x**2 * second.Y2s
    other = chi * second.X2s - chi * sigma * second.X2c + x**2 * second.Y2c
    cubic_cosine = (1 + x**2) ** 3 - 3 * (1 + x**2) * sigma**2
    cubic_sine = 3 * (1 + x**2) ** 2 * sigma - sigma**3
    boundary = -4 * jnp.pi * x**2 / (3 * denominator**3)
    boundary_c = boundary * (first * cubic_cosine - chi * other * cubic_sine)
    boundary_s = boundary * (chi * first * cubic_sine + other * cubic_cosine)
    boundary_cubic = jnp.stack((boundary_c, 3 * boundary_s, -3 * boundary_c, -boundary_s), axis=-1)

    quadratic = (
        jnp.stack((1 + x**2 + sigma**2, -2 * chi * sigma * x**2, x**2 * (1 + x**2)), axis=-1)
        / denominator[:, None]
    )
    observation_cubic = jnp.pi * jnp.concatenate((quadratic, jnp.zeros_like(x)[:, None]), axis=-1)
    ones, zeros = jnp.ones_like(x), jnp.zeros_like(x)
    tangent_correction = (current * curvature / (4 * jnp.pi))[:, None] * (
        _cubic_potential(ones, zeros, h_c, h_s) - observation_cubic
    ) + current / (2 * jnp.pi) * (_cubic_potential(shape_n, shape_b, h_c, h_s) - boundary_cubic)
    a_t = potential[:, 0] + tangent_correction
    a_n, a_b = potential[:, 1], potential[:, 2]
    m = -current / 2 * quadratic  # Quadratic potential of the uniform current.

    # B_t = d_u A_b - d_v A_n,  B_n = d_v A_t,  B_b = kappa*A_t - d_u A_t, differentiated twice.
    nn = jnp.stack(
        (6 * a_b[:, 0] - 2 * a_n[:, 1], 2 * a_t[:, 1], 2 * curvature * m[:, 0] - 6 * a_t[:, 0]),
        axis=-1,
    )
    nb = jnp.stack(
        (2 * a_b[:, 1] - 2 * a_n[:, 2], 2 * a_t[:, 2], curvature * m[:, 1] - 2 * a_t[:, 1]), axis=-1
    )
    bb = jnp.stack(
        (2 * a_b[:, 2] - 6 * a_n[:, 3], 6 * a_t[:, 3], 2 * curvature * m[:, 2] - 2 * a_t[:, 2]),
        axis=-1,
    )
    return jnp.stack((jnp.stack((nn, nb), axis=1), jnp.stack((nb, bb), axis=1)), axis=1)


def _plasma_hessian_frenet(
    solution: NearAxisSolution, gradient_frenet: jax.Array, source: PlasmaCurrentSource
) -> jax.Array:
    transverse = _transverse_plasma_hessian(solution, source)
    frenet_derivative_first = jnp.zeros((solution.inputs.nphi, 3, 3, 3), dtype=transverse.dtype)
    frenet_derivative_first = frenet_derivative_first.at[:, 1:, 1:, :].set(transverse)

    derivative_first_gradient = jnp.swapaxes(gradient_frenet, -1, -2)
    derivative_along_axis = (
        jnp.einsum("nm,mab->nab", solution.geometry.d_d_varphi, derivative_first_gradient)
        / solution.geometry.abs_G0_over_B0
    )
    zeros = jnp.zeros_like(solution.geometry.curvature)
    connection = jnp.stack(
        (
            jnp.stack((zeros, solution.geometry.curvature, zeros), axis=-1),
            jnp.stack((-solution.geometry.curvature, zeros, solution.geometry.torsion), axis=-1),
            jnp.stack((zeros, -solution.geometry.torsion, zeros), axis=-1),
        ),
        axis=-2,
    )
    tangential = (
        derivative_along_axis
        - jnp.einsum("...ad,...dg->...ag", connection, derivative_first_gradient)
        - jnp.einsum("...ad,...gd->...ag", derivative_first_gradient, connection)
    )
    frenet_derivative_first = frenet_derivative_first.at[:, 0, :, :].set(tangential)
    frenet_derivative_first = frenet_derivative_first.at[:, :, 0, :].set(tangential)
    return jnp.transpose(frenet_derivative_first, (0, 3, 1, 2))


def plasma_hessian_on_axis(
    solution: NearAxisSolution, *, formal_radius: ArrayLike, angular_resolution: int = 128
) -> PlasmaHessianData:
    """Evaluate the complete local plasma Hessian and external vacuum jet."""

    gradient = plasma_gradient_on_axis(
        solution, formal_radius=formal_radius, angular_resolution=angular_resolution
    )
    # The tangential rows differentiate the LEADING gradient. Using the order-a**2
    # zero-current gradient here would mix asymptotic orders within one tensor and
    # break the symmetry of the external Hessian at that order.
    source = gradient.field.current_source
    leading_gradient_frenet = elliptical_channel_gradient(
        solution.X1c,
        solution.sigma,
        parallel_current_mu0=source.parallel_current_mu0,
        chi=source.chi,
    )
    hessian_frenet = _plasma_hessian_frenet(solution, leading_gradient_frenet, source)
    frame = solution.geometry.frenet_frame
    hessian = jnp.einsum("...gi,...gab,...aj,...bk->...ijk", frame, hessian_frenet, frame, frame)
    external_hessian = solution.grad_grad_B_axis - hessian
    external_symmetric = _symmetrize_rank3(external_hessian)
    external_stf = project_symmetric_trace_free_rank3(external_hessian)
    external_trace = jnp.einsum("...iik->...k", external_hessian)
    external_symmetry = external_hessian - external_symmetric
    derivative_asymmetry = hessian - jnp.swapaxes(hessian, -1, -2)
    radius = gradient.field.current_source.formal_radius
    axis_scale = solution.geometry.abs_G0_over_B0
    logarithm = jnp.abs(jnp.log(radius / axis_scale))
    remainder = jnp.max(jnp.abs(hessian)) * (radius / axis_scale) ** 2 * (1 + logarithm)
    return PlasmaHessianData(
        field=gradient,
        hessian=hessian,
        hessian_frenet=hessian_frenet,
        external_hessian=external_hessian,
        external_hessian_stf=external_stf,
        external_hessian_independent=pack_symmetric_trace_free_rank3(external_hessian),
        maximum_derivative_asymmetry=jnp.max(jnp.abs(derivative_asymmetry)),
        maximum_external_symmetry_error=jnp.max(jnp.abs(external_symmetry)),
        maximum_external_trace=jnp.max(jnp.abs(external_trace)),
        estimated_hessian_remainder=remainder,
    )

"""Sampled magnetic-axis geometry and Frenet frames."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from pyqsc_jax.axis import Axis, AxisSamples, evaluate_axis
from pyqsc_jax.models import GeometryDiagnostics
from pyqsc_jax.spectral import differentiation_matrix, periodic_grid

ArrayLike = Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AxisGeometry:
    """Immutable sampled geometry over one magnetic field period.

    Cylindrical vector components use the final-axis order ``(R, phi, Z)``.
    Cartesian vector components use ``(x, y, z)``.
    """

    axis: Axis
    samples: AxisSamples
    position_cartesian: jax.Array
    d_r_d_phi_cylindrical: jax.Array
    d2_r_d_phi2_cylindrical: jax.Array
    d3_r_d_phi3_cylindrical: jax.Array
    d_l_d_phi: jax.Array
    axis_length: jax.Array
    tangent_cylindrical: jax.Array
    normal_cylindrical: jax.Array
    binormal_cylindrical: jax.Array
    tangent_cartesian: jax.Array
    normal_cartesian: jax.Array
    binormal_cartesian: jax.Array
    curvature: jax.Array
    torsion: jax.Array
    frame_helicity: jax.Array
    varphi: jax.Array
    d_varphi_d_phi: jax.Array
    d_d_phi: jax.Array
    d_d_varphi: jax.Array
    abs_G0_over_B0: jax.Array
    diagnostics: GeometryDiagnostics


def cylindrical_vector_to_cartesian(vector: ArrayLike, phi: ArrayLike) -> jax.Array:
    """Convert vectors from cylindrical ``(R, phi, Z)`` to Cartesian basis."""

    vector = jnp.asarray(vector)
    phi = jnp.asarray(phi)
    cosine = jnp.cos(phi)
    sine = jnp.sin(phi)
    return jnp.stack(
        (
            vector[..., 0] * cosine - vector[..., 1] * sine,
            vector[..., 0] * sine + vector[..., 1] * cosine,
            vector[..., 2],
        ),
        axis=-1,
    )


def _frame_helicity(normal_cylindrical: jax.Array) -> jax.Array:
    x_positive = normal_cylindrical[:, 0] >= 0
    z_positive = normal_cylindrical[:, 2] >= 0
    quadrant = jnp.where(
        x_positive & z_positive,
        1,
        jnp.where(~x_positive & z_positive, 2, jnp.where(~x_positive & ~z_positive, 3, 4)),
    )
    next_quadrant = jnp.roll(quadrant, -1)
    increment = jnp.where(
        (quadrant == 4) & (next_quadrant == 1),
        1,
        jnp.where(
            (quadrant == 1) & (next_quadrant == 4),
            -1,
            next_quadrant - quadrant,
        ),
    )
    return jnp.rint(jnp.sum(increment) / 4).astype(jnp.int32)


def compute_axis_geometry(
    axis: Axis,
    *,
    nphi: int = 61,
    speed_tolerance: ArrayLike = 1e-12,
    curvature_tolerance: ArrayLike = 1e-10,
    radius_tolerance: ArrayLike = 1e-10,
) -> AxisGeometry:
    """Evaluate axis geometry and Frenet validity over one field period."""

    if not isinstance(nphi, int) or isinstance(nphi, bool) or nphi < 3:
        raise ValueError("nphi must be an integer >= 3.")

    period = 2 * jnp.pi / axis.nfp
    phi = periodic_grid(nphi, period=period)
    samples = evaluate_axis(axis, phi)
    cosine = jnp.cos(phi)
    sine = jnp.sin(phi)
    position_cartesian = jnp.stack((samples.R * cosine, samples.R * sine, samples.Z), axis=-1)

    d_r = jnp.stack((samples.d_R_d_phi, samples.R, samples.d_Z_d_phi), axis=-1)
    d2_r = jnp.stack(
        (
            samples.d2_R_d_phi2 - samples.R,
            2 * samples.d_R_d_phi,
            samples.d2_Z_d_phi2,
        ),
        axis=-1,
    )
    d3_r = jnp.stack(
        (
            samples.d3_R_d_phi3 - 3 * samples.d_R_d_phi,
            3 * samples.d2_R_d_phi2 - samples.R,
            samples.d3_Z_d_phi3,
        ),
        axis=-1,
    )

    d_l_d_phi = jnp.linalg.norm(d_r, axis=-1)
    d2_l_d_phi2 = jnp.sum(d_r * d2_r, axis=-1) / d_l_d_phi
    tangent = d_r / d_l_d_phi[:, None]
    d_tangent_d_l = (-d_r * d2_l_d_phi2[:, None] / d_l_d_phi[:, None] + d2_r) / d_l_d_phi[
        :, None
    ] ** 2
    curvature = jnp.linalg.norm(d_tangent_d_l, axis=-1)
    pointwise_frenet_valid = (d_l_d_phi > speed_tolerance) & (curvature > curvature_tolerance)
    normal = jnp.where(
        pointwise_frenet_valid[:, None],
        d_tangent_d_l / curvature[:, None],
        jnp.nan,
    )
    binormal = jnp.cross(tangent, normal)

    cross_first_second = jnp.cross(d_r, d2_r)
    torsion = jnp.where(
        pointwise_frenet_valid,
        jnp.sum(d_r * jnp.cross(d2_r, d3_r), axis=-1) / jnp.sum(cross_first_second**2, axis=-1),
        jnp.nan,
    )

    d_phi = period / nphi
    axis_length = jnp.sum(d_l_d_phi) * d_phi * axis.nfp
    B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
    d_d_phi = differentiation_matrix(nphi, period=period)
    d_d_varphi = d_d_phi / d_varphi_d_phi[:, None]
    varphi = jnp.concatenate(
        (jnp.zeros(1, dtype=d_l_d_phi.dtype), jnp.cumsum(d_l_d_phi[:-1] + d_l_d_phi[1:]))
    )
    varphi = varphi * (0.5 * d_phi * 2 * jnp.pi / axis_length)

    tangent_cartesian = cylindrical_vector_to_cartesian(tangent, phi)
    normal_cartesian = cylindrical_vector_to_cartesian(normal, phi)
    binormal_cartesian = cylindrical_vector_to_cartesian(binormal, phi)
    frame = jnp.stack((tangent_cartesian, normal_cartesian, binormal_cartesian), axis=-2)
    gram = frame @ jnp.swapaxes(frame, -1, -2)
    identity = jnp.eye(3, dtype=frame.dtype)
    orthogonality_error = jnp.max(jnp.abs(gram - identity))
    determinant = jnp.linalg.det(frame)
    minimum_speed = jnp.min(d_l_d_phi)
    minimum_curvature = jnp.min(curvature)
    minimum_radius = jnp.min(samples.R)
    diagnostics = GeometryDiagnostics(
        minimum_speed=minimum_speed,
        minimum_curvature=minimum_curvature,
        minimum_cylindrical_radius=minimum_radius,
        maximum_frame_orthogonality_error=orthogonality_error,
        minimum_frame_determinant=jnp.min(determinant),
        frenet_valid=jnp.all(pointwise_frenet_valid) & jnp.isfinite(orthogonality_error),
        cylindrical_coordinates_valid=minimum_radius > radius_tolerance,
    )

    return AxisGeometry(
        axis=axis,
        samples=samples,
        position_cartesian=position_cartesian,
        d_r_d_phi_cylindrical=d_r,
        d2_r_d_phi2_cylindrical=d2_r,
        d3_r_d_phi3_cylindrical=d3_r,
        d_l_d_phi=d_l_d_phi,
        axis_length=axis_length,
        tangent_cylindrical=tangent,
        normal_cylindrical=normal,
        binormal_cylindrical=binormal,
        tangent_cartesian=tangent_cartesian,
        normal_cartesian=normal_cartesian,
        binormal_cartesian=binormal_cartesian,
        curvature=curvature,
        torsion=torsion,
        frame_helicity=_frame_helicity(normal),
        varphi=varphi,
        d_varphi_d_phi=d_varphi_d_phi,
        d_d_phi=d_d_phi,
        d_d_varphi=d_d_varphi,
        abs_G0_over_B0=abs_G0_over_B0,
        diagnostics=diagnostics,
    )

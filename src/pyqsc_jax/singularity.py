"""Regular-coordinate singular-radius diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pyqsc_jax.field import (
    _differentiate_cylindrical_vector,
    _regular_map_vectors,
    _to_cartesian,
)
from pyqsc_jax.models import NearAxisSolution, SingularityDiagnostics


def _determinant_coefficients(
    solution: NearAxisSolution,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    geometry = solution.geometry
    length_scale = geometry.abs_G0_over_B0
    d1, d2, h11, h12, h22 = _regular_map_vectors(solution)
    derivative = lambda vector: _differentiate_cylindrical_vector(vector, solution)  # noqa: E731
    cartesian = lambda vector: _to_cartesian(vector, solution)  # noqa: E731
    arguments = (
        cartesian(length_scale * geometry.tangent_cylindrical),
        cartesian(d1),
        cartesian(d2),
        cartesian(derivative(d1)),
        cartesian(derivative(d2)),
        cartesian(h11),
        cartesian(h12),
        cartesian(h22),
        cartesian(derivative(h11)),
        cartesian(derivative(h12)),
        cartesian(derivative(h22)),
    )
    (
        axis_tangent,
        d1_value,
        d2_value,
        d1_prime,
        d2_prime,
        h11_value,
        h12_value,
        h22_value,
        h11_prime,
        h12_prime,
        h22_prime,
    ) = arguments

    determinant = lambda first, second, third: jnp.einsum(  # noqa: E731
        "ni,ni->n", first, jnp.cross(second, third)
    )
    g0 = determinant(axis_tangent, d1_value, d2_value)
    g1c = (
        determinant(d1_prime, d1_value, d2_value)
        + determinant(axis_tangent, h11_value, d2_value)
        + determinant(axis_tangent, d1_value, h12_value)
    )
    g1s = (
        determinant(d2_prime, d1_value, d2_value)
        + determinant(axis_tangent, h12_value, d2_value)
        + determinant(axis_tangent, d1_value, h22_value)
    )
    q1_squared = (
        0.5 * determinant(h11_prime, d1_value, d2_value)
        + determinant(d1_prime, h11_value, d2_value)
        + determinant(d1_prime, d1_value, h12_value)
        + determinant(axis_tangent, h11_value, h12_value)
    )
    q2_squared = (
        0.5 * determinant(h22_prime, d1_value, d2_value)
        + determinant(d2_prime, h12_value, d2_value)
        + determinant(d2_prime, d1_value, h22_value)
        + determinant(axis_tangent, h12_value, h22_value)
    )
    q1_q2 = (
        determinant(h12_prime, d1_value, d2_value)
        + determinant(d1_prime, h12_value, d2_value)
        + determinant(d2_prime, h11_value, d2_value)
        + determinant(d1_prime, d1_value, h22_value)
        + determinant(d2_prime, d1_value, h12_value)
        + determinant(axis_tangent, h11_value, h22_value)
        + determinant(axis_tangent, h12_value, h12_value)
    )
    hessian_11 = 2 * q1_squared
    hessian_22 = 2 * q2_squared
    g20 = (hessian_11 + hessian_22) / 4
    g2c = (hessian_11 - hessian_22) / 4
    g2s = q1_q2 / 2
    return g0, g1c, g1s, g20, g2s, g2c


def _positive_quadratic_root(
    constant: jax.Array,
    linear: jax.Array,
    quadratic: jax.Array,
) -> jax.Array:
    tolerance = 100 * jnp.finfo(quadratic.dtype).eps
    discriminant = linear**2 - 4 * quadratic * constant
    square_root = jnp.sqrt(jnp.maximum(discriminant, 0))
    denominator = 2 * quadratic
    root_minus = (-linear - square_root) / denominator
    root_plus = (-linear + square_root) / denominator
    infinity = jnp.asarray(jnp.inf, dtype=quadratic.dtype)
    root_minus = jnp.where((discriminant >= 0) & (root_minus > 0), root_minus, infinity)
    root_plus = jnp.where((discriminant >= 0) & (root_plus > 0), root_plus, infinity)
    quadratic_root = jnp.minimum(root_minus, root_plus)
    linear_root = -constant / linear
    linear_root = jnp.where(linear_root > 0, linear_root, infinity)
    return jnp.where(jnp.abs(quadratic) <= tolerance, linear_root, quadratic_root)


def singularity_diagnostics(
    solution: NearAxisSolution,
    *,
    angular_resolution: int = 256,
    newton_iterations: int = 8,
) -> SingularityDiagnostics:
    """Locate the first singularity of the quadratic near-axis map.

    A uniform angular scan enumerates all positive roots of the quadratic
    Jacobian approximation. A vectorized Newton solve then refines the
    simultaneous conditions ``det(X) = 0`` and ``d det(X) / d theta = 0``.
    """

    if solution.second_order is None:
        raise ValueError("A second-order solution is required for singular-radius diagnostics.")
    if not isinstance(angular_resolution, int) or angular_resolution < 8:
        raise ValueError("angular_resolution must be an integer >= 8.")
    if not isinstance(newton_iterations, int) or newton_iterations < 0:
        raise ValueError("newton_iterations must be a nonnegative integer.")

    g0, g1c, g1s, g20, g2s, g2c = _determinant_coefficients(solution)
    theta_grid = jnp.arange(angular_resolution, dtype=g0.dtype) * (2 * jnp.pi / angular_resolution)
    sine = jnp.sin(theta_grid)
    cosine = jnp.cos(theta_grid)
    sine_2 = jnp.sin(2 * theta_grid)
    cosine_2 = jnp.cos(2 * theta_grid)
    linear = g1c[:, None] * cosine + g1s[:, None] * sine
    quadratic = g20[:, None] + g2s[:, None] * sine_2 + g2c[:, None] * cosine_2
    root_grid = _positive_quadratic_root(g0[:, None], linear, quadratic)
    minimum_indices = jnp.argmin(root_grid, axis=1)
    initial_radius = jnp.take_along_axis(root_grid, minimum_indices[:, None], axis=1)[:, 0]
    initial_theta = theta_grid[minimum_indices]
    finite_seed = jnp.isfinite(initial_radius)
    radius = jnp.where(finite_seed, initial_radius, 1)
    theta = initial_theta

    def newton_step(_, state):
        radius, theta = state
        sine = jnp.sin(theta)
        cosine = jnp.cos(theta)
        sine_2 = jnp.sin(2 * theta)
        cosine_2 = jnp.cos(2 * theta)
        linear = g1c * cosine + g1s * sine
        linear_prime = -g1c * sine + g1s * cosine
        linear_second = -linear
        quadratic = g20 + g2s * sine_2 + g2c * cosine_2
        quadratic_prime = 2 * g2s * cosine_2 - 2 * g2c * sine_2
        quadratic_second = -4 * (g2s * sine_2 + g2c * cosine_2)
        residual_0 = g0 + radius * linear + radius**2 * quadratic
        residual_1 = radius * linear_prime + radius**2 * quadratic_prime
        jacobian_00 = linear + 2 * radius * quadratic
        jacobian_01 = residual_1
        jacobian_10 = linear_prime + 2 * radius * quadratic_prime
        jacobian_11 = radius * linear_second + radius**2 * quadratic_second
        determinant = jacobian_00 * jacobian_11 - jacobian_01 * jacobian_10
        safe_determinant = jnp.where(
            jnp.abs(determinant) > jnp.finfo(radius.dtype).eps,
            determinant,
            jnp.inf,
        )
        delta_radius = (-residual_0 * jacobian_11 + jacobian_01 * residual_1) / safe_determinant
        delta_theta = (-jacobian_00 * residual_1 + residual_0 * jacobian_10) / safe_determinant
        candidate_radius = radius + delta_radius
        radius = jnp.where(candidate_radius > 0, candidate_radius, radius / 2)
        return radius, theta + delta_theta

    radius, theta = jax.lax.fori_loop(
        0,
        newton_iterations,
        newton_step,
        (radius, theta),
    )
    sine = jnp.sin(theta)
    cosine = jnp.cos(theta)
    sine_2 = jnp.sin(2 * theta)
    cosine_2 = jnp.cos(2 * theta)
    linear = g1c * cosine + g1s * sine
    linear_prime = -g1c * sine + g1s * cosine
    quadratic = g20 + g2s * sine_2 + g2c * cosine_2
    quadratic_prime = 2 * g2s * cosine_2 - 2 * g2c * sine_2
    residual_0 = g0 + radius * linear + radius**2 * quadratic
    residual_1 = radius * linear_prime + radius**2 * quadratic_prime
    residual_norm = jnp.sqrt(residual_0**2 + residual_1**2)
    valid = finite_seed & jnp.isfinite(radius) & jnp.isfinite(residual_norm)
    radius = jnp.where(valid, radius, jnp.inf)
    residual_norm = jnp.where(valid, residual_norm, jnp.inf)
    return SingularityDiagnostics(
        r_singularity=jnp.min(radius),
        r_singularity_vs_varphi=radius,
        inv_r_singularity_vs_varphi=1 / radius,
        theta_singularity_vs_varphi=jnp.mod(theta, 2 * jnp.pi),
        residual_norm_vs_varphi=residual_norm,
        maximum_residual_norm=jnp.max(jnp.where(valid, residual_norm, 0)),
        g0=g0,
        g1c=g1c,
        g1s=g1s,
        g20=g20,
        g2s=g2s,
        g2c=g2c,
        angular_resolution=angular_resolution,
        newton_iterations=newton_iterations,
    )

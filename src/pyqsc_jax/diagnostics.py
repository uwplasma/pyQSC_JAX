"""On-axis diagnostics of a near-axis solution.

Total field jet through second derivatives, the regular-coordinate singular
radius, leading-order Mercier terms, and B20 diagnostics with the analytic
elimination of the affine ``B2c`` input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from pyqsc_jax.geometry import cylindrical_vector_to_cartesian
from pyqsc_jax.models import FieldJet, MercierDiagnostics, NearAxisSolution, SingularityDiagnostics
from pyqsc_jax.second_order import MU0, solve_second_order


def _differentiate_cylindrical_vector(vector: jax.Array, solution: NearAxisSolution) -> jax.Array:
    """Differentiate a vector with respect to Boozer ``varphi``.

    Cylindrical components are periodic over one field period, while fixed
    Cartesian components generally are not. The two connection terms below
    account for rotation of the cylindrical basis.
    """

    derivative = solution.geometry.d_d_varphi @ vector
    d_phi_d_varphi = 1 / solution.geometry.d_varphi_d_phi
    derivative = derivative.at[:, 0].add(-d_phi_d_varphi * vector[:, 1])
    return derivative.at[:, 1].add(d_phi_d_varphi * vector[:, 0])


def _to_cartesian(vector: jax.Array, solution: NearAxisSolution) -> jax.Array:
    return cylindrical_vector_to_cartesian(vector, solution.phi)


def _regular_map_vectors(
    solution: NearAxisSolution,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``d1, d2, h11, h12, h22`` in periodic cylindrical components."""

    r2 = solution.second_order
    if r2 is None:
        raise ValueError("A second-order solution is required for the regular coordinate map.")
    geometry = solution.geometry
    tangent = geometry.tangent_cylindrical
    normal = geometry.normal_cylindrical
    binormal = geometry.binormal_cylindrical
    d1 = solution.X1c[:, None] * normal + solution.Y1c[:, None] * binormal
    d2 = solution.Y1s[:, None] * binormal
    h11 = (
        2 * (r2.X20 + r2.X2c)[:, None] * normal
        + 2 * (r2.Y20 + r2.Y2c)[:, None] * binormal
        + 2 * (r2.Z20 + r2.Z2c)[:, None] * tangent
    )
    h12 = (
        2 * r2.X2s[:, None] * normal
        + 2 * r2.Y2s[:, None] * binormal
        + 2 * r2.Z2s[:, None] * tangent
    )
    h22 = (
        2 * (r2.X20 - r2.X2c)[:, None] * normal
        + 2 * (r2.Y20 - r2.Y2c)[:, None] * binormal
        + 2 * (r2.Z20 - r2.Z2c)[:, None] * tangent
    )
    return d1, d2, h11, h12, h22


def total_field_jet(solution: NearAxisSolution) -> FieldJet:
    """Compute ``B``, ``grad(B)``, and ``grad(grad(B))`` on the axis.

    This is the regular-coordinate chain rule in equations (55)--(83) of the
    surface-free plasma/coil derivation. It requires the complete second-order
    near-axis solution but no finite-radius surface.
    """

    r2 = solution.second_order
    if r2 is None:
        raise ValueError("A second-order solution is required for the field Hessian.")

    geometry = solution.geometry
    inputs = solution.inputs
    length_scale = geometry.abs_G0_over_B0
    tangent = geometry.tangent_cylindrical
    d1, d2, h11, h12, h22 = _regular_map_vectors(solution)

    derivative = lambda vector: _differentiate_cylindrical_vector(vector, solution)  # noqa: E731
    V0 = length_scale * tangent
    V1 = derivative(d1) + solution.iotaN * d2
    V2 = derivative(d2) - solution.iotaN * d1
    V11 = derivative(h11) + 2 * solution.iotaN * h12
    V12 = derivative(h12) + solution.iotaN * (h22 - h11)
    V22 = derivative(h22) - 2 * solution.iotaN * h12

    p0 = inputs.B0**2 / solution.G0
    p1 = 2 * inputs.B0**2 * inputs.etabar / solution.G0
    flux_term = inputs.B0**2 * (r2.G2 + solution.iota * inputs.I2) / solution.G0**2
    C11 = (
        inputs.B0**2 * inputs.etabar**2 + 2 * inputs.B0 * (r2.B20 + inputs.B2c)
    ) / solution.G0 - flux_term
    C22 = 2 * inputs.B0 * (r2.B20 - inputs.B2c) / solution.G0 - flux_term

    field_coordinate_gradient_cylindrical = jnp.stack(
        (p0 * derivative(V0), p1 * V0 + p0 * V1, p0 * V2), axis=-1
    )
    field_coordinate_hessian_cylindrical = jnp.zeros(
        (inputs.nphi, 3, 3, 3), dtype=field_coordinate_gradient_cylindrical.dtype
    )
    coordinate_hessian_values = {
        (0, 0): p0 * derivative(derivative(V0)),
        (0, 1): p1 * derivative(V0) + p0 * derivative(V1),
        (0, 2): p0 * derivative(V2),
        (1, 1): 2 * C11[:, None] * V0 + 2 * p1 * V1 + p0 * V11,
        (1, 2): p1 * V2 + p0 * V12,
        (2, 2): 2 * C22[:, None] * V0 + p0 * V22,
    }
    for (first, second), value in coordinate_hessian_values.items():
        field_coordinate_hessian_cylindrical = field_coordinate_hessian_cylindrical.at[
            :, :, first, second
        ].set(value)
        field_coordinate_hessian_cylindrical = field_coordinate_hessian_cylindrical.at[
            :, :, second, first
        ].set(value)

    field_coordinate_gradient = jnp.stack(
        [
            _to_cartesian(field_coordinate_gradient_cylindrical[:, :, index], solution)
            for index in range(3)
        ],
        axis=-1,
    )
    field_coordinate_hessian = jnp.stack(
        [
            jnp.stack(
                [
                    _to_cartesian(
                        field_coordinate_hessian_cylindrical[:, :, first, second], solution
                    )
                    for second in range(3)
                ],
                axis=-1,
            )
            for first in range(3)
        ],
        axis=-1,
    )

    coordinate_jacobian = jnp.stack(
        (
            length_scale * geometry.tangent_cartesian,
            _to_cartesian(d1, solution),
            _to_cartesian(d2, solution),
        ),
        axis=-1,
    )
    coordinate_hessian = jnp.zeros_like(field_coordinate_hessian)
    map_hessian_values = {
        (0, 0): length_scale**2 * solution.curvature[:, None] * geometry.normal_cartesian,
        (0, 1): _to_cartesian(derivative(d1), solution),
        (0, 2): _to_cartesian(derivative(d2), solution),
        (1, 1): _to_cartesian(h11, solution),
        (1, 2): _to_cartesian(h12, solution),
        (2, 2): _to_cartesian(h22, solution),
    }
    for (first, second), value in map_hessian_values.items():
        coordinate_hessian = coordinate_hessian.at[:, :, first, second].set(value)
        coordinate_hessian = coordinate_hessian.at[:, :, second, first].set(value)

    inverse_coordinate_jacobian = jnp.linalg.inv(coordinate_jacobian)
    gradient = jnp.einsum("nia,naj->nij", field_coordinate_gradient, inverse_coordinate_jacobian)
    hessian = jnp.einsum(
        "niab,naj,nbk->nijk",
        field_coordinate_hessian,
        inverse_coordinate_jacobian,
        inverse_coordinate_jacobian,
    ) - jnp.einsum(
        "nia,nal,nlcb,ncj,nbk->nijk",
        field_coordinate_gradient,
        inverse_coordinate_jacobian,
        coordinate_hessian,
        inverse_coordinate_jacobian,
        inverse_coordinate_jacobian,
    )
    field = p0 * _to_cartesian(V0, solution)

    frenet_basis = jnp.stack(
        (geometry.normal_cartesian, geometry.binormal_cartesian, geometry.tangent_cartesian),
        axis=-1,
    )
    field_first_frenet = jnp.einsum(
        "nia,nijk,njb,nkc->nabc", frenet_basis, hessian, frenet_basis, frenet_basis
    )
    hessian_frenet = jnp.transpose(field_first_frenet, (0, 2, 3, 1))

    norm_squared = jnp.sum(hessian**2, axis=(1, 2, 3))
    inverse_scale_vs_varphi = jnp.sqrt(jnp.sqrt(norm_squared) / (4 * inputs.B0))
    coordinate_determinant = jnp.linalg.det(coordinate_jacobian)
    divergence = jnp.trace(gradient, axis1=1, axis2=2)
    divergence_gradient = jnp.einsum("niik->nk", hessian)
    return FieldJet(
        field=field,
        gradient=gradient,
        hessian=hessian,
        hessian_frenet=hessian_frenet,
        coordinate_jacobian=coordinate_jacobian,
        inverse_coordinate_jacobian=inverse_coordinate_jacobian,
        coordinate_hessian=coordinate_hessian,
        minimum_absolute_coordinate_jacobian=jnp.min(jnp.abs(coordinate_determinant)),
        maximum_field_error=jnp.max(jnp.abs(field - solution.B_axis)),
        maximum_gradient_error=jnp.max(jnp.abs(gradient - solution.grad_B_axis)),
        maximum_divergence=jnp.max(jnp.abs(divergence)),
        maximum_derivative_asymmetry=jnp.max(jnp.abs(hessian - jnp.swapaxes(hessian, 2, 3))),
        maximum_divergence_gradient=jnp.max(jnp.abs(divergence_gradient)),
        grad_grad_B_inverse_scale_length_vs_varphi=inverse_scale_vs_varphi,
        L_grad_grad_B=1 / inverse_scale_vs_varphi,
        grad_grad_B_inverse_scale_length=jnp.max(inverse_scale_vs_varphi),
    )


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
    constant: jax.Array, linear: jax.Array, quadratic: jax.Array
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
    solution: NearAxisSolution, *, angular_resolution: int = 256, newton_iterations: int = 8
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
    # The angular scan only seeds Newton. Its branchy root formula (sqrt of a
    # clipped discriminant, where-selected infinities) has NaN derivatives, so
    # derivatives must come from the refined det = 0, d(det)/dtheta = 0 equations.
    radius = jax.lax.stop_gradient(jnp.where(finite_seed, initial_radius, 1))
    theta = jax.lax.stop_gradient(initial_theta)

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
            jnp.abs(determinant) > jnp.finfo(radius.dtype).eps, determinant, jnp.inf
        )
        delta_radius = (-residual_0 * jacobian_11 + jacobian_01 * residual_1) / safe_determinant
        delta_theta = (-jacobian_00 * residual_1 + residual_0 * jacobian_10) / safe_determinant
        candidate_radius = radius + delta_radius
        radius = jnp.where(candidate_radius > 0, candidate_radius, radius / 2)
        return radius, theta + delta_theta

    radius, theta = jax.lax.fori_loop(0, newton_iterations, newton_step, (radius, theta))
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


def mercier_diagnostics(solution: NearAxisSolution) -> MercierDiagnostics:
    """Compute the leading magnetic-well and Mercier terms.

    The normalization and signs follow the standard pyQSC implementation.
    """

    r2 = solution.second_order
    if r2 is None:
        raise ValueError("A second-order solution is required for Mercier diagnostics.")

    inputs = solution.inputs
    geometry = solution.geometry
    etabar_squared = inputs.etabar**2
    curvature_squared = solution.curvature**2
    numerator = (
        etabar_squared**2
        + curvature_squared**2 * solution.sigma**2
        + etabar_squared * curvature_squared
    )
    denominator = (
        etabar_squared**2
        + curvature_squared**2 * (1 + solution.sigma**2)
        + 2 * etabar_squared * curvature_squared
    )
    integrand = geometry.d_l_d_phi * numerator / denominator
    weighted_integral = (
        jnp.sum(integrand)
        * (2 * jnp.pi / (inputs.axis.nfp * inputs.nphi))
        * inputs.axis.nfp
        * 2
        * jnp.pi
        / geometry.axis_length
    )
    DGeod_times_r2 = (
        -2
        * MU0**2
        * inputs.p2**2
        * solution.G0**4
        * etabar_squared
        / (jnp.pi**3 * inputs.B0**10 * solution.iotaN**2)
        * weighted_integral
    )
    d2_volume_d_psi2 = (
        4
        * jnp.pi**2
        * jnp.abs(solution.G0)
        / inputs.B0**3
        * (
            3 * etabar_squared
            - 4 * r2.B20_mean / inputs.B0
            + 2 * (r2.G2 + solution.iota * inputs.I2) / solution.G0
        )
    )
    DWell_times_r2 = (
        MU0
        * inputs.p2
        * jnp.abs(solution.G0)
        / (8 * jnp.pi**4 * inputs.B0**3)
        * (d2_volume_d_psi2 - 8 * jnp.pi**2 * MU0 * inputs.p2 * jnp.abs(solution.G0) / inputs.B0**5)
    )
    return MercierDiagnostics(
        d2_volume_d_psi2=d2_volume_d_psi2,
        DGeod_times_r2=DGeod_times_r2,
        DWell_times_r2=DWell_times_r2,
        DMerc_times_r2=DWell_times_r2 + DGeod_times_r2,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class B20Diagnostics:
    """Dense-grid and toroidal-spectrum measures of nonconstant B20."""

    weighted_mean: jax.Array
    anomaly: jax.Array
    weighted_l2: jax.Array
    smooth_maximum: jax.Array
    grid_maximum: jax.Array
    peak_to_peak: jax.Array
    fourier_modes: jax.Array
    fourier_coefficients: jax.Array
    nonzero_fourier_norm: jax.Array
    nonzero_fourier_l1: jax.Array
    fourier_tail_ratio: jax.Array
    smooth_maximum_power: int


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class B2cOptimizationResult:
    """Exact weighted-L2 optimum for the affine B2c dependence."""

    solution: NearAxisSolution
    diagnostics: B20Diagnostics
    B2c_optimal: jax.Array
    affine_intercept: jax.Array
    affine_response: jax.Array
    projected_response_norm_squared: jax.Array
    affine_reconstruction_error: jax.Array
    degenerate: jax.Array


def b20_diagnostics(
    solution: NearAxisSolution, *, smooth_maximum_power: int = 16
) -> B20Diagnostics:
    """Evaluate normalized dense-grid and nonzero-mode B20 diagnostics."""

    if solution.second_order is None:
        raise ValueError("A second-order solution is required for B20 diagnostics.")
    if (
        not isinstance(smooth_maximum_power, int)
        or isinstance(smooth_maximum_power, bool)
        or smooth_maximum_power < 2
    ):
        raise ValueError("smooth_maximum_power must be an integer >= 2.")
    weights = solution.geometry.d_l_d_phi
    weight_sum = jnp.sum(weights)
    B20 = solution.B20
    B0 = solution.inputs.B0
    weighted_mean = jnp.sum(weights * B20) / weight_sum
    anomaly = B20 - weighted_mean
    normalized = anomaly / B0
    weighted_l2 = jnp.sqrt(jnp.sum(weights * normalized**2) / weight_sum)
    smooth_maximum = (
        jnp.sum(weights * jnp.abs(normalized) ** smooth_maximum_power) / weight_sum
    ) ** (1 / smooth_maximum_power)
    grid_maximum = jnp.max(jnp.abs(normalized))
    peak_to_peak = (jnp.max(B20) - jnp.min(B20)) / B0

    maximum_mode = (solution.inputs.nphi - 1) // 2
    modes = jnp.arange(1, maximum_mode + 1)
    phase = jnp.exp(-1j * modes[:, None] * solution.inputs.axis.nfp * solution.varphi[None, :])
    coefficients = jnp.sum(weights[None, :] * anomaly[None, :] * phase, axis=1) / weight_sum
    coefficient_power = jnp.abs(coefficients / B0) ** 2
    nonzero_fourier_norm = jnp.sqrt(2 * jnp.sum(coefficient_power))
    nonzero_fourier_l1 = 2 * jnp.sum(jnp.abs(coefficients / B0))
    tail_start = maximum_mode * 3 // 4
    tiny = jnp.finfo(B20.dtype).tiny
    fourier_tail_ratio = jnp.sqrt(
        jnp.sum(coefficient_power[tail_start:]) / jnp.maximum(jnp.sum(coefficient_power), tiny)
    )
    return B20Diagnostics(
        weighted_mean=weighted_mean,
        anomaly=anomaly,
        weighted_l2=weighted_l2,
        smooth_maximum=smooth_maximum,
        grid_maximum=grid_maximum,
        peak_to_peak=peak_to_peak,
        fourier_modes=modes,
        fourier_coefficients=coefficients,
        nonzero_fourier_norm=nonzero_fourier_norm,
        nonzero_fourier_l1=nonzero_fourier_l1,
        fourier_tail_ratio=fourier_tail_ratio,
        smooth_maximum_power=smooth_maximum_power,
    )


def _first_order_with_B2c(solution: NearAxisSolution, B2c: jax.Array) -> NearAxisSolution:
    inputs = replace(solution.inputs, B2c=B2c)
    return replace(
        solution,
        inputs=inputs,
        second_order=None,
        mercier=None,
        field_jet=None,
        singularity=None,
        third_order=None,
    )


def _affine_B20(solution: NearAxisSolution) -> tuple[jax.Array, jax.Array]:
    if solution.second_order is None:
        raise ValueError("A second-order solution is required to eliminate B2c.")
    zero = solve_second_order(
        _first_order_with_B2c(solution, jnp.zeros_like(solution.inputs.B2c)),
        attach_diagnostics=False,
    )
    one = solve_second_order(
        _first_order_with_B2c(solution, jnp.ones_like(solution.inputs.B2c)),
        attach_diagnostics=False,
    )
    return zero.B20, one.B20 - zero.B20


def optimal_B2c_value(
    solution: NearAxisSolution, *, degeneracy_tolerance: float = 1e-24
) -> jax.Array:
    """Return the exact B2c minimizing the nonconstant weighted-L2 B20."""

    if degeneracy_tolerance < 0:
        raise ValueError("degeneracy_tolerance must be nonnegative.")
    intercept, response = _affine_B20(solution)
    weights = solution.geometry.d_l_d_phi
    weight_sum = jnp.sum(weights)
    projected_intercept = intercept - jnp.sum(weights * intercept) / weight_sum
    projected_response = response - jnp.sum(weights * response) / weight_sum
    denominator = jnp.sum(weights * projected_response**2)
    numerator = jnp.sum(weights * projected_intercept * projected_response)
    return jnp.where(
        denominator > degeneracy_tolerance, -numerator / denominator, solution.inputs.B2c
    )


def optimize_B2c(
    solution: NearAxisSolution,
    *,
    smooth_maximum_power: int = 16,
    degeneracy_tolerance: float = 1e-24,
) -> B2cOptimizationResult:
    """Eliminate B2c analytically and return the fully recomputed solution."""

    intercept, response = _affine_B20(solution)
    weights = solution.geometry.d_l_d_phi
    weight_sum = jnp.sum(weights)
    projected_intercept = intercept - jnp.sum(weights * intercept) / weight_sum
    projected_response = response - jnp.sum(weights * response) / weight_sum
    denominator = jnp.sum(weights * projected_response**2)
    numerator = jnp.sum(weights * projected_intercept * projected_response)
    degenerate = denominator <= degeneracy_tolerance
    optimum = jnp.where(degenerate, solution.inputs.B2c, -numerator / denominator)
    optimal = solve_second_order(_first_order_with_B2c(solution, optimum))
    if solution.inputs.order == 3:
        from pyqsc_jax.third_order import solve_third_order

        optimal = solve_third_order(optimal)
    reconstruction_error = jnp.max(jnp.abs(optimal.B20 - (intercept + optimum * response)))
    return B2cOptimizationResult(
        solution=optimal,
        diagnostics=b20_diagnostics(optimal, smooth_maximum_power=smooth_maximum_power),
        B2c_optimal=optimum,
        affine_intercept=intercept,
        affine_response=response,
        projected_response_norm_squared=denominator,
        affine_reconstruction_error=reconstruction_error,
        degenerate=degenerate,
    )

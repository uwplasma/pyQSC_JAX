"""Dense B20 diagnostics and exact elimination of the affine B2c input."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from pyqsc_jax.first_order import solve
from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.second_order import solve_second_order


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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class B20ResolutionVerification:
    """Independent fixed-parameter B20 checks on successively finer grids."""

    resolutions: jax.Array
    weighted_l2: jax.Array
    smooth_maximum: jax.Array
    grid_maximum: jax.Array
    peak_to_peak: jax.Array
    nonzero_fourier_l1: jax.Array
    fourier_tail_ratio: jax.Array
    relative_weighted_l2_change: jax.Array
    relative_grid_maximum_change: jax.Array


def b20_diagnostics(
    solution: NearAxisSolution,
    *,
    smooth_maximum_power: int = 16,
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
        shear=None,
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
    solution: NearAxisSolution,
    *,
    degeneracy_tolerance: float = 1e-24,
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
        denominator > degeneracy_tolerance,
        -numerator / denominator,
        solution.inputs.B2c,
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
    if solution.shear is not None:
        from pyqsc_jax.shear import solve_magnetic_shear

        optimal = solve_magnetic_shear(optimal, B31c=solution.shear.B31c)
    reconstruction_error = jnp.max(jnp.abs(optimal.B20 - (intercept + optimum * response)))
    return B2cOptimizationResult(
        solution=optimal,
        diagnostics=b20_diagnostics(
            optimal,
            smooth_maximum_power=smooth_maximum_power,
        ),
        B2c_optimal=optimum,
        affine_intercept=intercept,
        affine_response=response,
        projected_response_norm_squared=denominator,
        affine_reconstruction_error=reconstruction_error,
        degenerate=degenerate,
    )


def verify_B20_resolution(
    solution: NearAxisSolution,
    *,
    multipliers: tuple[int, ...] = (1, 2, 4),
    smooth_maximum_power: int = 16,
) -> B20ResolutionVerification:
    """Recompute one fixed physical candidate at independent resolutions."""

    if not multipliers or any(
        not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 1
        for multiplier in multipliers
    ):
        raise ValueError("multipliers must be a nonempty tuple of positive integers.")
    inputs = solution.inputs
    resolutions = tuple(multiplier * (inputs.nphi - 1) + 1 for multiplier in multipliers)
    diagnostics = []
    for nphi in resolutions:
        candidate = solve(
            axis=inputs.axis,
            etabar=inputs.etabar,
            B0=inputs.B0,
            sigma0=inputs.sigma0,
            I2=inputs.I2,
            p2=inputs.p2,
            B2c=inputs.B2c,
            B2s=inputs.B2s,
            nphi=nphi,
            order="r2",
            sG=inputs.sG,
            spsi=inputs.spsi,
        )
        diagnostics.append(b20_diagnostics(candidate, smooth_maximum_power=smooth_maximum_power))
    weighted_l2 = jnp.stack(tuple(item.weighted_l2 for item in diagnostics))
    smooth_maximum = jnp.stack(tuple(item.smooth_maximum for item in diagnostics))
    grid_maximum = jnp.stack(tuple(item.grid_maximum for item in diagnostics))
    peak_to_peak = jnp.stack(tuple(item.peak_to_peak for item in diagnostics))
    nonzero_fourier_l1 = jnp.stack(tuple(item.nonzero_fourier_l1 for item in diagnostics))
    tail_ratio = jnp.stack(tuple(item.fourier_tail_ratio for item in diagnostics))
    tiny = jnp.finfo(weighted_l2.dtype).tiny
    relative_l2_change = jnp.concatenate(
        (
            jnp.zeros(1, dtype=weighted_l2.dtype),
            jnp.abs(jnp.diff(weighted_l2)) / jnp.maximum(weighted_l2[:-1], tiny),
        )
    )
    relative_maximum_change = jnp.concatenate(
        (
            jnp.zeros(1, dtype=grid_maximum.dtype),
            jnp.abs(jnp.diff(grid_maximum)) / jnp.maximum(grid_maximum[:-1], tiny),
        )
    )
    return B20ResolutionVerification(
        resolutions=jnp.asarray(resolutions),
        weighted_l2=weighted_l2,
        smooth_maximum=smooth_maximum,
        grid_maximum=grid_maximum,
        peak_to_peak=peak_to_peak,
        nonzero_fourier_l1=nonzero_fourier_l1,
        fourier_tail_ratio=tail_ratio,
        relative_weighted_l2_change=relative_l2_change,
        relative_grid_maximum_change=relative_maximum_change,
    )

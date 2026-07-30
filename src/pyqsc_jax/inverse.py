"""Branch-local inverse solves for a prescribed rotational transform."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp

from pyqsc_jax.first_order import sigma_equation, sigma_residual
from pyqsc_jax.geometry import AxisGeometry
from pyqsc_jax.models import InverseSolveDiagnostics, NearAxisInputs, RootSolveReport
from pyqsc_jax.solvers import DEFAULT_ROOT_OPTIONS, RootSolveOptions, implicit_dense_root

ArrayLike = Any


def parameter_response_derivative(
    inputs: NearAxisInputs,
    geometry: AxisGeometry,
    sigma: jax.Array,
    iota: jax.Array,
    *,
    parameter: str,
) -> jax.Array:
    """Return the local forward derivative d(iota)/d(parameter)."""

    if parameter not in ("etabar", "I2"):
        raise ValueError("parameter must be 'etabar' or 'I2'.")
    parameter_value = getattr(inputs, parameter)
    forward_state = jnp.asarray(sigma).at[0].set(iota)
    forward_jacobian = jax.jacfwd(
        lambda candidate: sigma_residual(
            candidate,
            inputs=inputs,
            geometry=geometry,
        )
    )(forward_state)
    parameter_derivative = jax.jacfwd(
        lambda value: sigma_residual(
            forward_state,
            inputs=replace(inputs, **{parameter: value}),
            geometry=geometry,
        )
    )(parameter_value)
    return jnp.linalg.solve(forward_jacobian, -parameter_derivative)[0]


def solve_target_iota(
    inputs: NearAxisInputs,
    geometry: AxisGeometry,
    *,
    target_iota: ArrayLike,
    root_options: RootSolveOptions = DEFAULT_ROOT_OPTIONS,
    fold_tolerance: float = 1e-8,
) -> tuple[
    NearAxisInputs,
    jax.Array,
    jax.Array,
    RootSolveReport,
    InverseSolveDiagnostics,
]:
    """Solve the sigma equation at fixed iota for etabar or I2."""

    target_iota = jnp.asarray(target_iota)
    if target_iota.ndim != 0:
        raise ValueError("iota must be a scalar.")
    if inputs.solve_for not in ("etabar", "I2"):
        raise ValueError("Target-iota mode requires solve_for='etabar' or solve_for='I2'.")

    initial_state = jnp.full((inputs.nphi,), inputs.sigma0)
    if inputs.solve_for == "etabar":
        parameter_sign = jnp.where(inputs.etabar < 0, -1.0, 1.0)
        safe_magnitude = jnp.maximum(jnp.abs(inputs.etabar), jnp.finfo(inputs.etabar.dtype).tiny)
        initial_state = initial_state.at[0].set(jnp.log(safe_magnitude))

        def parameter_from_state(state):
            return parameter_sign * jnp.exp(state[0])

        def inputs_from_parameter(parameter):
            return replace(inputs, etabar=parameter)

    else:
        parameter_sign = jnp.sign(inputs.I2)
        initial_state = initial_state.at[0].set(inputs.I2)

        def parameter_from_state(state):
            return state[0]

        def inputs_from_parameter(parameter):
            return replace(inputs, I2=parameter)

    def residual(state):
        parameter = parameter_from_state(state)
        local_inputs = inputs_from_parameter(parameter)
        sigma = state.at[0].set(inputs.sigma0)
        return sigma_equation(
            sigma,
            target_iota,
            inputs=local_inputs,
            geometry=geometry,
        )

    state, report = implicit_dense_root(
        residual,
        initial_state,
        options=root_options,
    )
    solved_parameter = parameter_from_state(state)
    solved_inputs = inputs_from_parameter(solved_parameter)
    sigma = state.at[0].set(inputs.sigma0)

    response_derivative = parameter_response_derivative(
        solved_inputs,
        geometry,
        sigma,
        target_iota,
        parameter=inputs.solve_for,
    )
    absolute_response = jnp.abs(response_derivative)
    branch_fold = ~jnp.isfinite(response_derivative) | (absolute_response <= fold_tolerance)
    diagnostics = InverseSolveDiagnostics(
        target_iota=target_iota,
        achieved_iota=target_iota,
        solved_value=solved_parameter,
        response_derivative=response_derivative,
        absolute_response_derivative=absolute_response,
        fold_tolerance=jnp.asarray(fold_tolerance),
        branch_fold=branch_fold,
        parameter_sign=jnp.sign(solved_parameter),
        parameter=inputs.solve_for,
    )
    return solved_inputs, sigma, target_iota, report, diagnostics

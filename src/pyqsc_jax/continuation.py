"""Pseudo-arclength continuation of first-order near-axis branches."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp

from pyqsc_jax.axis import Axis
from pyqsc_jax.first_order import first_order_solution, sigma_equation, solve
from pyqsc_jax.inverse import parameter_response_derivative
from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.solvers import DEFAULT_ROOT_OPTIONS, RootSolveOptions, implicit_dense_root

ArrayLike = Any


@dataclass(frozen=True)
class ContinuationResult:
    """A variable-length sequence of corrected pseudo-arclength points."""

    solutions: tuple[NearAxisSolution, ...]
    tangents: jax.Array
    response_derivative: jax.Array
    fold_detected: jax.Array
    requested_points: int
    status: str

    @property
    def etabar(self) -> jax.Array:
        """Etabar values along the corrected branch."""

        return jnp.stack(tuple(solution.inputs.etabar for solution in self.solutions))

    @property
    def iota(self) -> jax.Array:
        """Rotational-transform values along the corrected branch."""

        return jnp.stack(tuple(solution.iota for solution in self.solutions))

    @property
    def sigma(self) -> jax.Array:
        """Periodic sigma samples along the corrected branch."""

        return jnp.stack(tuple(solution.sigma for solution in self.solutions))

    @property
    def complete(self) -> bool:
        """Whether all requested points converged."""

        return self.status == "complete" and len(self.solutions) == self.requested_points


def continue_etabar_branch(
    *,
    axis: Axis,
    etabar_start: ArrayLike,
    etabar_next: ArrayLike,
    num_points: int = 25,
    step_size: float | None = None,
    B0: ArrayLike = 1.0,
    sigma0: ArrayLike = 0.0,
    I2: ArrayLike = 0.0,
    nphi: int = 61,
    sG: int = 1,
    spsi: int = 1,
    root_options: RootSolveOptions = DEFAULT_ROOT_OPTIONS,
    fold_tolerance: float = 1e-4,
) -> ContinuationResult:
    """Trace a fixed-sign etabar branch through folds in iota(etabar)."""

    if not isinstance(num_points, int) or isinstance(num_points, bool) or num_points < 2:
        raise ValueError("num_points must be an integer >= 2.")
    if step_size is not None and step_size <= 0:
        raise ValueError("step_size must be positive.")
    if fold_tolerance < 0:
        raise ValueError("fold_tolerance must be nonnegative.")

    etabar_start = jnp.asarray(etabar_start)
    etabar_next = jnp.asarray(etabar_next)
    if etabar_start.ndim != 0 or etabar_next.ndim != 0:
        raise ValueError("etabar_start and etabar_next must be scalars.")
    sign = jnp.where(etabar_start < 0, -1.0, 1.0)
    if bool(etabar_start == 0) or bool(etabar_next == 0):
        raise ValueError("Continuation seeds must have nonzero etabar.")
    if bool(jnp.sign(etabar_start) != jnp.sign(etabar_next)):
        raise ValueError("Continuation seeds must have the same etabar sign.")

    common = {
        "axis": axis,
        "B0": B0,
        "sigma0": sigma0,
        "I2": I2,
        "nphi": nphi,
        "order": "r1",
        "sG": sG,
        "spsi": spsi,
    }
    first = solve(etabar=etabar_start, **common)
    second = solve(etabar=etabar_next, **common)
    if not bool(first.root_report.converged) or not bool(second.root_report.converged):
        raise ValueError("Both initial continuation points must converge.")

    solutions = [first, second]
    pair0 = jnp.asarray((first.inputs.etabar, first.iota))
    pair1 = jnp.asarray((second.inputs.etabar, second.iota))
    secant = pair1 - pair0
    secant_norm = jnp.linalg.norm(secant)
    if not bool(jnp.isfinite(secant_norm)) or bool(secant_norm == 0):
        raise ValueError("Initial continuation points must be finite and distinct.")
    tangent = secant / secant_norm
    arclength_step = float(secant_norm) if step_size is None else step_size
    tangents = [tangent, tangent]
    responses = [
        parameter_response_derivative(
            first.inputs,
            first.geometry,
            first.sigma,
            first.iota,
            parameter="etabar",
        ),
        parameter_response_derivative(
            second.inputs,
            second.geometry,
            second.sigma,
            second.iota,
            parameter="etabar",
        ),
    ]
    fold_flags = [
        jnp.abs(responses[0]) <= fold_tolerance,
        (jnp.abs(responses[1]) <= fold_tolerance) | (responses[0] * responses[1] < 0),
    ]
    status = "complete"

    for _ in range(2, num_points):
        previous = solutions[-1]
        before_previous = solutions[-2]
        previous_pair = jnp.asarray((previous.inputs.etabar, previous.iota))
        predicted_pair = previous_pair + arclength_step * tangent
        predicted_etabar = predicted_pair[0]
        if bool(sign * predicted_etabar <= 0):
            status = "branch_zero_crossing"
            break

        sigma_scale = arclength_step / jnp.linalg.norm(
            previous_pair - jnp.asarray((before_previous.inputs.etabar, before_previous.iota))
        )
        predicted_sigma = previous.sigma + sigma_scale * (previous.sigma - before_previous.sigma)
        initial_state = jnp.concatenate(
            (
                jnp.log(jnp.abs(predicted_etabar))[None],
                predicted_pair[1:],
                predicted_sigma[1:],
            )
        )
        base_inputs = previous.inputs

        def residual(
            state,
            base_inputs=base_inputs,
            geometry=previous.geometry,
            predicted_pair=predicted_pair,
            tangent=tangent,
        ):
            etabar = sign * jnp.exp(state[0])
            iota = state[1]
            sigma = jnp.concatenate((base_inputs.sigma0[None], state[2:]))
            local_inputs = replace(base_inputs, etabar=etabar)
            sigma_part = sigma_equation(
                sigma,
                iota,
                inputs=local_inputs,
                geometry=geometry,
            )
            arclength_part = jnp.dot(
                jnp.asarray((etabar, iota)) - predicted_pair,
                tangent,
            )
            return jnp.concatenate((sigma_part, arclength_part[None]))

        state, report = implicit_dense_root(
            residual,
            initial_state,
            options=root_options,
        )
        corrected_etabar = sign * jnp.exp(state[0])
        corrected_iota = state[1]
        corrected_sigma = jnp.concatenate((base_inputs.sigma0[None], state[2:]))
        corrected_inputs = replace(base_inputs, etabar=corrected_etabar)
        corrected = first_order_solution(
            corrected_inputs,
            previous.geometry,
            corrected_sigma,
            corrected_iota,
            report,
        )
        if not bool(report.converged):
            status = "solver_failure"
            break

        corrected_pair = jnp.asarray((corrected_etabar, corrected_iota))
        new_tangent = corrected_pair - previous_pair
        new_tangent = new_tangent / jnp.linalg.norm(new_tangent)
        new_tangent = jnp.where(jnp.dot(new_tangent, tangent) < 0, -new_tangent, new_tangent)
        response = parameter_response_derivative(
            corrected.inputs,
            corrected.geometry,
            corrected.sigma,
            corrected.iota,
            parameter="etabar",
        )
        fold = (jnp.abs(response) <= fold_tolerance) | (responses[-1] * response < 0)
        solutions.append(corrected)
        tangents.append(new_tangent)
        responses.append(response)
        fold_flags.append(fold)
        tangent = new_tangent

    return ContinuationResult(
        solutions=tuple(solutions),
        tangents=jnp.stack(tuple(tangents)),
        response_derivative=jnp.stack(tuple(responses)),
        fold_detected=jnp.stack(tuple(fold_flags)),
        requested_points=num_points,
        status=status,
    )

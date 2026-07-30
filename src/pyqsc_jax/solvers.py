"""pyQSC_JAX-specific nonlinear solver policy and reports."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from solvax import linear_solve, root_solve

from pyqsc_jax.models import LinearSolveReport, RootSolveReport

ArrayLike = Any


@dataclass(frozen=True)
class RootSolveOptions:
    """Tolerance and globalization policy for dense Newton root solves."""

    atol: float = 1e-13
    rtol: float = 1e-13
    step_tolerance: float = 1e-13
    max_steps: int = 20
    max_backtracking_steps: int = 12

    def __post_init__(self) -> None:
        if self.atol < 0 or self.rtol < 0 or self.step_tolerance < 0:
            raise ValueError("Root tolerances must be nonnegative.")
        if self.max_steps < 0 or self.max_backtracking_steps < 0:
            raise ValueError("Root iteration limits must be nonnegative.")


DEFAULT_ROOT_OPTIONS = RootSolveOptions()


def _infinity_norm(value: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(value))


def dense_newton_root(
    residual_function: Callable[[jax.Array], jax.Array],
    initial_guess: ArrayLike,
    *,
    options: RootSolveOptions = DEFAULT_ROOT_OPTIONS,
) -> tuple[jax.Array, RootSolveReport]:
    """Solve a small dense nonlinear system with damped Newton updates."""

    initial_guess = jnp.asarray(initial_guess)
    initial_residual = residual_function(initial_guess)
    initial_residual_norm = _infinity_norm(initial_residual)
    tolerance = jnp.maximum(options.atol, options.rtol * initial_residual_norm)
    dtype = initial_guess.dtype

    initial_state = (
        initial_guess,
        initial_residual,
        initial_residual_norm,
        jnp.asarray(jnp.inf, dtype=dtype),
        jnp.int32(0),
        jnp.int32(0),
    )

    def continue_iteration(state):
        x, residual, residual_norm, step_norm, iterations, _ = state
        finite = (
            jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(residual)) & jnp.isfinite(residual_norm)
        )
        step_threshold = options.step_tolerance * (1 + _infinity_norm(x))
        return (
            (residual_norm > tolerance)
            & (iterations < options.max_steps)
            & (step_norm > step_threshold)
            & finite
        )

    def newton_step(state):
        x, residual, residual_norm, _, iterations, total_backtracking = state
        jacobian = jax.jacfwd(residual_function)(x)
        if initial_guess.ndim == 0:
            full_step = -residual / jacobian
        else:
            full_step = jnp.linalg.solve(jacobian, -residual)

        def candidate(damping):
            candidate_x = x + damping * full_step
            candidate_residual = residual_function(candidate_x)
            candidate_norm = _infinity_norm(candidate_residual)
            return candidate_x, candidate_residual, candidate_norm

        damping0 = jnp.asarray(1.0, dtype=dtype)
        candidate_x0, candidate_residual0, candidate_norm0 = candidate(damping0)
        line_state0 = (
            damping0,
            candidate_x0,
            candidate_residual0,
            candidate_norm0,
            jnp.int32(0),
        )

        def continue_backtracking(line_state):
            _, _, candidate_residual, candidate_norm, backtracking = line_state
            candidate_finite = jnp.all(jnp.isfinite(candidate_residual)) & jnp.isfinite(
                candidate_norm
            )
            return ((candidate_norm >= residual_norm) | ~candidate_finite) & (
                backtracking < options.max_backtracking_steps
            )

        def backtrack(line_state):
            damping, _, _, _, backtracking = line_state
            damping = 0.5 * damping
            candidate_x, candidate_residual, candidate_norm = candidate(damping)
            return (
                damping,
                candidate_x,
                candidate_residual,
                candidate_norm,
                backtracking + 1,
            )

        damping, candidate_x, candidate_residual, candidate_norm, backtracking = jax.lax.while_loop(
            continue_backtracking, backtrack, line_state0
        )
        step_norm = _infinity_norm(damping * full_step)
        return (
            candidate_x,
            candidate_residual,
            candidate_norm,
            step_norm,
            iterations + 1,
            total_backtracking + backtracking,
        )

    x, residual, residual_norm, step_norm, iterations, backtracking_steps = jax.lax.while_loop(
        continue_iteration, newton_step, initial_state
    )
    finite = (
        jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(residual)) & jnp.isfinite(residual_norm)
    )
    converged = finite & (residual_norm <= tolerance)
    step_threshold = options.step_tolerance * (1 + _infinity_norm(x))
    stagnated = finite & ~converged & (step_norm <= step_threshold)
    final_jacobian = jax.jacfwd(residual_function)(x)
    if initial_guess.ndim == 0:
        jacobian_condition_number = jnp.where(final_jacobian == 0, jnp.inf, 1.0)
    else:
        jacobian_condition_number = jnp.linalg.cond(final_jacobian)
    report = RootSolveReport(
        initial_residual_norm=initial_residual_norm,
        residual_norm=residual_norm,
        tolerance=tolerance,
        step_norm=step_norm,
        iterations=iterations,
        backtracking_steps=backtracking_steps,
        jacobian_condition_number=jacobian_condition_number,
        converged=converged,
        finite=finite,
        stagnated=stagnated,
    )
    return x, report


def implicit_dense_root(
    residual_function: Callable[[jax.Array], jax.Array],
    initial_guess: ArrayLike,
    *,
    options: RootSolveOptions = DEFAULT_ROOT_OPTIONS,
) -> tuple[jax.Array, RootSolveReport]:
    """Solve once with dense Newton and differentiate the converged equation.

    The Newton candidate is stopped before it is supplied to SOLVAX's custom
    root. Consequently JVPs and VJPs use the implicit function theorem rather
    than differentiating the iteration history.
    """

    candidate, report = dense_newton_root(
        residual_function,
        initial_guess,
        options=options,
    )
    candidate = jax.lax.stop_gradient(candidate)
    root = root_solve(
        residual_function,
        candidate,
        lambda _function, supplied_candidate: supplied_candidate,
    )
    return root, report


def implicit_dense_linear_solve(
    matrix: ArrayLike,
    right_hand_side: ArrayLike,
    *,
    residual_tolerance: float = 1e-11,
    condition_limit: float = 1e12,
) -> tuple[jax.Array, LinearSolveReport]:
    """Solve a dense system with implicit JVP/VJP rules and diagnostics."""

    if residual_tolerance < 0:
        raise ValueError("residual_tolerance must be nonnegative.")
    if condition_limit <= 0:
        raise ValueError("condition_limit must be positive.")

    matrix = jnp.asarray(matrix)
    right_hand_side = jnp.asarray(right_hand_side)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if right_hand_side.ndim != 1 or right_hand_side.shape[0] != matrix.shape[0]:
        raise ValueError("right_hand_side must match the matrix dimension.")

    matvec = lambda value: matrix @ value  # noqa: E731
    transpose_matvec = lambda value: matrix.T @ value  # noqa: E731
    primal_solver = lambda _operator, value: jnp.linalg.solve(matrix, value)  # noqa: E731
    transpose_solver = lambda _operator, value: jnp.linalg.solve(  # noqa: E731
        matrix.T,
        value,
    )
    solution = linear_solve(
        matvec,
        right_hand_side,
        primal_solver,
        transpose_matvec=transpose_matvec,
        transpose_solver=transpose_solver,
    )
    residual = matrix @ solution - right_hand_side
    residual_norm = _infinity_norm(residual)
    right_hand_side_norm = _infinity_norm(right_hand_side)
    relative_residual_norm = residual_norm / jnp.maximum(
        right_hand_side_norm,
        jnp.finfo(right_hand_side.dtype).tiny,
    )
    condition_number = jnp.linalg.cond(matrix)
    finite = (
        jnp.all(jnp.isfinite(solution))
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(condition_number)
    )
    converged = finite & (relative_residual_norm <= residual_tolerance)
    well_conditioned = finite & (condition_number <= condition_limit)
    report = LinearSolveReport(
        residual_norm=residual_norm,
        relative_residual_norm=relative_residual_norm,
        matrix_condition_number=condition_number,
        finite=finite,
        converged=converged,
        well_conditioned=well_conditioned,
    )
    return solution, report

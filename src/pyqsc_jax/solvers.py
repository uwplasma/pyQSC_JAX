"""pyQSC_JAX-specific nonlinear solver policy and reports."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from solvax import linear_solve, root_solve

from pyqsc_jax.models import LinearSolveReport, RootSolveReport

ArrayLike = Any


@dataclass(frozen=True)
class RootSolveOptions:
    """Tolerance and globalization policy for dense Newton root solves.

    ``exact_condition_number`` additionally reports the exact 2-norm condition
    number of the final Jacobian (an O(n^3) SVD); by default only the O(n^2)
    1-norm estimate is computed.
    """

    atol: float = 1e-13
    rtol: float = 1e-13
    step_tolerance: float = 1e-13
    max_steps: int = 20
    max_backtracking_steps: int = 12
    exact_condition_number: bool = False

    def __post_init__(self) -> None:
        if self.atol < 0 or self.rtol < 0 or self.step_tolerance < 0:
            raise ValueError("Root tolerances must be nonnegative.")
        if self.max_steps < 0 or self.max_backtracking_steps < 0:
            raise ValueError("Root iteration limits must be nonnegative.")


DEFAULT_ROOT_OPTIONS = RootSolveOptions()


def _infinity_norm(value: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(value))


def _lu_condition_estimate(matrix: jax.Array, factors, iterations: int = 5) -> jax.Array:
    """Hager-Higham estimate of the 1-norm condition number from LU factors.

    Uses ``2 * iterations`` pairs of triangular solves, O(n^2) each, instead of
    the O(n^3) SVD behind ``jnp.linalg.cond``. The result is a lower bound on
    ``kappa_1(A) = |A|_1 |A^-1|_1``, in practice within a small factor of it.
    """

    n = matrix.shape[0]
    dtype = matrix.dtype

    def body(_, state):
        x, estimate = state
        y = jsl.lu_solve(factors, x)
        estimate = jnp.maximum(estimate, jnp.sum(jnp.abs(y)))
        z = jsl.lu_solve(factors, jnp.where(y >= 0, 1.0, -1.0).astype(dtype), trans=1)
        x = jax.nn.one_hot(jnp.argmax(jnp.abs(z)), n, dtype=dtype)
        return x, estimate

    x0 = jnp.full((n,), 1.0 / n, dtype=dtype)
    _, inverse_norm = jax.lax.fori_loop(0, iterations, body, (x0, jnp.zeros((), dtype)))
    matrix_norm = jnp.max(jnp.sum(jnp.abs(matrix), axis=0))
    return jnp.where(jnp.isfinite(inverse_norm), matrix_norm * inverse_norm, jnp.inf)


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
        jnp.asarray(False),
    )

    def continue_iteration(state):
        x, residual, residual_norm, step_norm, iterations, _, line_search_failed = state
        finite = (
            jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(residual)) & jnp.isfinite(residual_norm)
        )
        step_threshold = options.step_tolerance * (1 + _infinity_norm(x))
        return (
            (residual_norm > tolerance)
            & (iterations < options.max_steps)
            & (step_norm > step_threshold)
            & finite
            & ~line_search_failed
        )

    def newton_step(state):
        x, residual, residual_norm, _, iterations, total_backtracking, _ = state
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

        def acceptable(candidate_residual, candidate_norm):
            finite = jnp.all(jnp.isfinite(candidate_residual)) & jnp.isfinite(candidate_norm)
            return finite & (candidate_norm < residual_norm)

        damping0 = jnp.asarray(1.0, dtype=dtype)
        candidate_x0, candidate_residual0, candidate_norm0 = candidate(damping0)
        line_state0 = (damping0, candidate_x0, candidate_residual0, candidate_norm0, jnp.int32(0))

        def continue_backtracking(line_state):
            _, _, candidate_residual, candidate_norm, backtracking = line_state
            return ~acceptable(candidate_residual, candidate_norm) & (
                backtracking < options.max_backtracking_steps
            )

        def backtrack(line_state):
            damping, _, _, _, backtracking = line_state
            damping = 0.5 * damping
            candidate_x, candidate_residual, candidate_norm = candidate(damping)
            return (damping, candidate_x, candidate_residual, candidate_norm, backtracking + 1)

        damping, candidate_x, candidate_residual, candidate_norm, backtracking = jax.lax.while_loop(
            continue_backtracking, backtrack, line_state0
        )
        # Only a step that decreases the residual is accepted, so the returned iterate is
        # always the best one seen. A failed line search keeps the current iterate and stops.
        accepted = acceptable(candidate_residual, candidate_norm)
        return (
            jnp.where(accepted, candidate_x, x),
            jnp.where(accepted, candidate_residual, residual),
            jnp.where(accepted, candidate_norm, residual_norm),
            jnp.where(accepted, _infinity_norm(damping * full_step), 0.0),
            iterations + 1,
            total_backtracking + backtracking,
            ~accepted,
        )

    (x, residual, residual_norm, step_norm, iterations, backtracking_steps, line_search_failed) = (
        jax.lax.while_loop(continue_iteration, newton_step, initial_state)
    )
    finite = (
        jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(residual)) & jnp.isfinite(residual_norm)
    )
    converged = finite & (residual_norm <= tolerance)
    line_search_failed = line_search_failed & ~converged
    step_threshold = options.step_tolerance * (1 + _infinity_norm(x))
    stagnated = finite & ~converged & ~line_search_failed & (step_norm <= step_threshold)
    final_jacobian = jax.lax.stop_gradient(jax.jacfwd(residual_function)(x))
    jacobian_condition_number = None
    if initial_guess.ndim == 0:
        jacobian_condition_estimate = jnp.where(final_jacobian == 0, jnp.inf, 1.0)
    else:
        jacobian_condition_estimate = _lu_condition_estimate(
            final_jacobian, jsl.lu_factor(final_jacobian)
        )
        if options.exact_condition_number:
            jacobian_condition_number = jnp.linalg.cond(final_jacobian)
    report = RootSolveReport(
        initial_residual_norm=initial_residual_norm,
        residual_norm=residual_norm,
        tolerance=tolerance,
        step_norm=step_norm,
        iterations=iterations,
        backtracking_steps=backtracking_steps,
        jacobian_condition_estimate=jacobian_condition_estimate,
        jacobian_condition_number=jacobian_condition_number,
        converged=converged,
        finite=finite,
        stagnated=stagnated,
        line_search_failed=line_search_failed,
    )
    return x, report


@jax.custom_jvp
def _gate_tangent(value: jax.Array, valid: jax.Array) -> jax.Array:
    """Identity on values; tangents are multiplied by NaN when ``valid`` is false."""

    return value


@_gate_tangent.defjvp
def _gate_tangent_jvp(primals, tangents):
    value, valid = primals
    tangent, _ = tangents
    scale = jnp.where(valid, 1.0, jnp.nan).astype(value.dtype)
    return value, tangent * scale


def implicit_dense_root(
    residual_function: Callable[[jax.Array], jax.Array],
    initial_guess: ArrayLike,
    *,
    options: RootSolveOptions = DEFAULT_ROOT_OPTIONS,
) -> tuple[jax.Array, RootSolveReport]:
    """Solve once with dense Newton and differentiate the converged equation.

    The Newton candidate is stopped before it is supplied to SOLVAX's custom
    root, so JVPs and VJPs use the implicit function theorem rather than the
    iteration history. The implicit derivative is valid only at a converged
    root with a regular (finite-condition) Jacobian; otherwise every tangent
    and cotangent of the returned root is NaN, so a failed solve can never
    silently supply a usable gradient. Check ``report.converged`` on the
    primal path.
    """

    candidate, report = dense_newton_root(residual_function, initial_guess, options=options)
    candidate = jax.lax.stop_gradient(candidate)
    root = root_solve(
        residual_function, candidate, lambda _function, supplied_candidate: supplied_candidate
    )
    regular = report.converged & jnp.isfinite(report.jacobian_condition_estimate)
    return _gate_tangent(root, jax.lax.stop_gradient(regular)), report


def implicit_dense_linear_solve(
    matrix: ArrayLike,
    right_hand_side: ArrayLike,
    *,
    residual_tolerance: float = 1e-11,
    condition_limit: float = 1e12,
    exact_condition_number: bool = False,
) -> tuple[jax.Array, LinearSolveReport]:
    """Solve a dense system with implicit JVP/VJP rules and diagnostics.

    One LU factorization serves the primal solve, the transposed (adjoint)
    solve of reverse mode, and the 1-norm condition estimate used for
    ``well_conditioned``; the exact 2-norm condition number (an SVD) is
    computed only with ``exact_condition_number=True``.
    Derivatives are attached only when the solve converged (finite solution,
    relative residual within ``residual_tolerance``); otherwise tangents and
    cotangents of the solution are NaN.
    """

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

    factors = jsl.lu_factor(jax.lax.stop_gradient(matrix))
    matvec = lambda value: matrix @ value  # noqa: E731
    transpose_matvec = lambda value: matrix.T @ value  # noqa: E731
    primal_solver = lambda _operator, value: jsl.lu_solve(factors, value)  # noqa: E731
    transpose_solver = lambda _operator, value: jsl.lu_solve(factors, value, trans=1)  # noqa: E731
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
        right_hand_side_norm, jnp.finfo(right_hand_side.dtype).tiny
    )
    condition_estimate = _lu_condition_estimate(jax.lax.stop_gradient(matrix), factors)
    condition_number = (
        jnp.linalg.cond(jax.lax.stop_gradient(matrix)) if exact_condition_number else None
    )
    finite = (
        jnp.all(jnp.isfinite(solution))
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(condition_estimate)
    )
    converged = finite & (relative_residual_norm <= residual_tolerance)
    well_conditioned = finite & (condition_estimate <= condition_limit)
    solution = _gate_tangent(solution, jax.lax.stop_gradient(converged))
    report = LinearSolveReport(
        residual_norm=residual_norm,
        relative_residual_norm=relative_residual_norm,
        condition_estimate=condition_estimate,
        condition_number=condition_number,
        finite=finite,
        converged=converged,
        well_conditioned=well_conditioned,
    )
    return solution, report

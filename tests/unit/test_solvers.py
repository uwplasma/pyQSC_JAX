"""Dense Newton and implicit linear solves: convergence, failure semantics, derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyqsc_jax.solvers import (
    RootSolveOptions,
    dense_newton_root,
    implicit_dense_linear_solve,
    implicit_dense_root,
)


def test_newton_converges_with_backtracking_and_reports_it():
    """Newton finds sqrt(2) cleanly and needs backtracking for x**3 = 1 from x = 0.1."""

    root, report = dense_newton_root(lambda x: x**2 - 2.0, jnp.asarray(1.0))
    np.testing.assert_allclose(root, jnp.sqrt(2.0), rtol=2e-13)
    assert bool(report.converged) and bool(report.finite) and not bool(report.stagnated)
    assert 0 < int(report.iterations) < 20
    assert report.residual_norm <= report.tolerance
    np.testing.assert_allclose(report.jacobian_condition_estimate, 1.0)
    assert report.jacobian_condition_number is None

    root, report = dense_newton_root(lambda x: x**3 - 1.0, jnp.asarray(0.1))
    np.testing.assert_allclose(root, 1.0, rtol=2e-12)
    assert bool(report.converged) and not bool(report.line_search_failed)
    assert int(report.backtracking_steps) > 0


def test_newton_failures_are_reported_not_hidden():
    """An iteration cap and a rootless residual both return a finite, unconverged report."""

    root, report = dense_newton_root(
        lambda x: x**2 - 2.0, jnp.asarray(1.0), options=RootSolveOptions(max_steps=0)
    )
    np.testing.assert_allclose(root, 1.0)
    assert not bool(report.converged) and bool(report.finite)
    assert int(report.iterations) == 0

    # x**2 + 1 has no real root: the first step lands on the minimum x = 0 where J = 0.
    root, report = dense_newton_root(lambda x: x**2 + 1.0, jnp.asarray(1.0))
    np.testing.assert_allclose(root, 0.0, atol=1e-15)
    np.testing.assert_allclose(report.residual_norm, 1.0)
    assert bool(report.line_search_failed)
    assert not bool(report.converged) and not bool(report.stagnated)


def test_implicit_root_derivative_uses_the_converged_equation():
    """d sqrt(p)/dp = 1/(2 sqrt(p)) through the implicit-function theorem, under jit."""

    def root(parameter):
        return implicit_dense_root(lambda x: x**2 - parameter, jnp.asarray(1.0))[0]

    np.testing.assert_allclose(jax.jit(root)(2.0), jnp.sqrt(2.0), rtol=2e-13)
    np.testing.assert_allclose(jax.grad(root)(2.0), 1 / (2 * jnp.sqrt(2.0)), rtol=2e-12)


def test_failed_root_solve_poisons_implicit_derivative():
    """An unconverged solve keeps its primal value but returns NaN grad and JVP."""

    def unconverged(parameter):
        return implicit_dense_root(
            lambda x: x**2 - parameter, jnp.asarray(1.0), options=RootSolveOptions(max_steps=1)
        )[0]

    assert np.isnan(jax.grad(unconverged)(2.0))
    assert np.isnan(jax.jit(jax.grad(unconverged))(2.0))
    assert np.isnan(jax.jvp(unconverged, (2.0,), (1.0,))[1])
    assert np.isfinite(unconverged(2.0))


def test_implicit_linear_solve_value_and_derivatives():
    """Linear solve matches numpy; jacfwd == jacrev == central difference."""

    base = jnp.asarray([[3.0, 0.4, -0.2], [0.1, 2.5, 0.3], [-0.5, 0.2, 1.8]])
    right_hand_side = jnp.asarray([1.0, -2.0, 0.5])
    solution, report = implicit_dense_linear_solve(base, right_hand_side)
    np.testing.assert_allclose(solution, jnp.linalg.solve(base, right_hand_side))
    assert bool(report.converged) and bool(report.finite) and bool(report.well_conditioned)
    assert report.relative_residual_norm < 1e-14

    def solve_for(parameter):
        matrix = base + parameter * jnp.eye(3)
        return implicit_dense_linear_solve(matrix, right_hand_side * (1 + parameter))[0]

    forward = jax.jacfwd(solve_for)(0.3)
    reverse = jax.jacrev(solve_for)(0.3)
    step = 1e-6
    finite_difference = (solve_for(0.3 + step) - solve_for(0.3 - step)) / (2 * step)
    np.testing.assert_allclose(forward, reverse, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(forward, finite_difference, rtol=1e-8)


def test_singular_linear_solve_poisons_derivative():
    """The derivative through an exactly singular matrix is NaN, not a silent number."""

    def objective(parameter):
        matrix = jnp.asarray([[parameter, 1.0], [1.0, 1.0]])
        return jnp.sum(implicit_dense_linear_solve(matrix, jnp.asarray([1.0, 2.0]))[0])

    assert np.isfinite(jax.grad(objective)(2.0))
    assert np.isnan(jax.grad(objective)(1.0))


def test_condition_estimate_brackets_the_exact_condition_number():
    """The 1-norm estimate lies within a factor 10 below the exact value for kappa ~ 1e8."""

    rng = np.random.default_rng(0)
    q1, _ = np.linalg.qr(rng.normal(size=(40, 40)))
    q2, _ = np.linalg.qr(rng.normal(size=(40, 40)))
    matrix = jnp.asarray(q1 @ np.diag(np.logspace(0, -8, 40)) @ q2)
    right_hand_side = jnp.ones(40)

    _, report = implicit_dense_linear_solve(matrix, right_hand_side, exact_condition_number=True)
    exact_one_norm = np.linalg.cond(np.asarray(matrix), 1)
    assert exact_one_norm / 10 <= report.condition_estimate <= exact_one_norm * (1 + 1e-10)
    np.testing.assert_allclose(report.condition_number, np.linalg.cond(np.asarray(matrix)))
    _, cheap = implicit_dense_linear_solve(matrix, right_hand_side)
    assert cheap.condition_number is None

    options = RootSolveOptions(exact_condition_number=True, rtol=1e-10)
    _, root_report = dense_newton_root(
        lambda x: (matrix + jnp.diag(x**2)) @ x - right_hand_side, jnp.zeros(40), options=options
    )
    assert root_report.jacobian_condition_number is not None
    assert np.isfinite(root_report.jacobian_condition_estimate)


def test_solver_option_and_shape_guards():
    """Negative tolerances, non-square matrices and mismatched right-hand sides raise."""

    for kwargs in (
        {"atol": -1.0},
        {"rtol": -1.0},
        {"step_tolerance": -1.0},
        {"max_steps": -1},
        {"max_backtracking_steps": -1},
    ):
        with pytest.raises(ValueError):
            RootSolveOptions(**kwargs)
    for matrix, right_hand_side, message in (
        (jnp.ones(2), jnp.ones(2), "square"),
        (jnp.ones((2, 3)), jnp.ones(2), "square"),
        (jnp.eye(2), jnp.ones((2, 1)), "right_hand_side"),
        (jnp.eye(2), jnp.ones(3), "right_hand_side"),
    ):
        with pytest.raises(ValueError, match=message):
            implicit_dense_linear_solve(matrix, right_hand_side)
    for kwargs, message in (
        ({"residual_tolerance": -1.0}, "residual_tolerance"),
        ({"condition_limit": 0.0}, "condition_limit"),
    ):
        with pytest.raises(ValueError, match=message):
            implicit_dense_linear_solve(jnp.eye(2), jnp.ones(2), **kwargs)

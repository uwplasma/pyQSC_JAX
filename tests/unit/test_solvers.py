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


def test_dense_newton_converges_with_report():
    root, report = dense_newton_root(lambda x: x**2 - 2.0, jnp.asarray(1.0))

    np.testing.assert_allclose(root, jnp.sqrt(2.0), rtol=2e-13)
    assert bool(report.converged)
    assert bool(report.finite)
    assert not bool(report.stagnated)
    assert 0 < int(report.iterations) < 20
    assert report.residual_norm <= report.tolerance
    np.testing.assert_allclose(report.jacobian_condition_number, 1.0)


def test_backtracking_is_exercised():
    root, report = dense_newton_root(lambda x: x**3 - 1.0, jnp.asarray(0.1))

    np.testing.assert_allclose(root, 1.0, rtol=2e-12)
    assert bool(report.converged)
    assert int(report.backtracking_steps) > 0


def test_iteration_limit_returns_failure_report():
    options = RootSolveOptions(max_steps=0)
    root, report = dense_newton_root(lambda x: x**2 - 2.0, jnp.asarray(1.0), options=options)

    np.testing.assert_allclose(root, 1.0)
    assert not bool(report.converged)
    assert bool(report.finite)
    assert int(report.iterations) == 0


def test_implicit_root_gradient_uses_converged_equation():
    def root(parameter):
        solved, _ = implicit_dense_root(
            lambda x: x**2 - parameter,
            jnp.asarray(1.0),
        )
        return solved

    parameter = 2.0
    np.testing.assert_allclose(root(parameter), jnp.sqrt(parameter), rtol=2e-13)
    np.testing.assert_allclose(
        jax.grad(root)(parameter),
        1 / (2 * jnp.sqrt(parameter)),
        rtol=2e-12,
    )
    np.testing.assert_allclose(jax.jit(root)(parameter), root(parameter), rtol=2e-13)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"atol": -1.0},
        {"rtol": -1.0},
        {"step_tolerance": -1.0},
        {"max_steps": -1},
        {"max_backtracking_steps": -1},
    ],
)
def test_root_options_validation(kwargs):
    with pytest.raises(ValueError):
        RootSolveOptions(**kwargs)


def test_implicit_dense_linear_solve_and_gradient():
    right_hand_side = jnp.asarray([1.0, -0.5])

    def objective(parameter):
        matrix = jnp.asarray([[parameter, 0.2], [-0.4, 1.7]])
        solution, _ = implicit_dense_linear_solve(matrix, right_hand_side)
        return jnp.sum(solution**2)

    parameter = 2.0
    matrix = jnp.asarray([[parameter, 0.2], [-0.4, 1.7]])
    solution, report = implicit_dense_linear_solve(matrix, right_hand_side)
    np.testing.assert_allclose(solution, jnp.linalg.solve(matrix, right_hand_side))
    assert bool(report.converged)
    assert bool(report.finite)
    assert bool(report.well_conditioned)
    assert report.relative_residual_norm < 1e-14

    step = 2e-5
    finite_difference = (objective(parameter + step) - objective(parameter - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(objective)(parameter), finite_difference, rtol=2e-8)


@pytest.mark.parametrize(
    "matrix, right_hand_side, message",
    [
        (jnp.ones(2), jnp.ones(2), "square"),
        (jnp.ones((2, 3)), jnp.ones(2), "square"),
        (jnp.eye(2), jnp.ones((2, 1)), "right_hand_side"),
        (jnp.eye(2), jnp.ones(3), "right_hand_side"),
    ],
)
def test_linear_solve_shape_validation(matrix, right_hand_side, message):
    with pytest.raises(ValueError, match=message):
        implicit_dense_linear_solve(matrix, right_hand_side)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"residual_tolerance": -1.0}, "residual_tolerance"),
        ({"condition_limit": 0.0}, "condition_limit"),
    ],
)
def test_linear_solve_policy_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        implicit_dense_linear_solve(jnp.eye(2), jnp.ones(2), **kwargs)

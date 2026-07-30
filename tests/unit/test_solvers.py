import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyqsc_jax.solvers import (
    RootSolveOptions,
    dense_newton_root,
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

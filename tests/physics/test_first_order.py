import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.first_order import sigma_residual
from pyqsc_jax.models import NearAxisInputs


def standard_solution(nphi=31, **kwargs):
    parameters = {
        "axis": qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3),
        "etabar": -0.9,
        "nphi": nphi,
    }
    parameters.update(kwargs)
    return qsc.solve(**parameters)


def test_sigma_residual_and_report():
    solution = standard_solution()
    state = solution.sigma.at[0].set(solution.iota)
    residual = sigma_residual(
        state,
        inputs=solution.inputs,
        geometry=solution.geometry,
    )

    assert bool(solution.root_report.converged)
    assert bool(solution.root_report.finite)
    assert solution.root_report.residual_norm < 2e-13
    np.testing.assert_allclose(jnp.max(jnp.abs(residual)), solution.root_report.residual_norm)
    np.testing.assert_allclose(solution.sigma[0], solution.inputs.sigma0)


def test_vacuum_field_gradient_satisfies_maxwell_identities_spectrally():
    solution = standard_solution(nphi=61)
    gradient = solution.grad_B_axis
    divergence = jnp.trace(gradient, axis1=-2, axis2=-1)
    antisymmetric = gradient - jnp.swapaxes(gradient, -1, -2)

    np.testing.assert_allclose(jnp.linalg.norm(solution.B_axis, axis=-1), solution.inputs.B0)
    assert jnp.max(jnp.abs(divergence)) < 2e-8
    assert jnp.max(jnp.abs(antisymmetric)) < 2e-8
    assert jnp.all(solution.L_grad_B > 0)


def test_first_order_shapes_and_coefficients():
    solution = standard_solution()
    assert solution.axis is solution.inputs.axis
    assert solution.B_axis.shape == (31, 3)
    assert solution.grad_B_axis.shape == (31, 3, 3)
    assert solution.sigma.shape == (31,)
    np.testing.assert_allclose(solution.X1s, 0.0)
    np.testing.assert_allclose(
        solution.X1c * solution.curvature,
        solution.inputs.etabar,
        rtol=2e-13,
    )
    area_jacobian = solution.X1c * solution.Y1s - solution.X1s * solution.Y1c
    np.testing.assert_allclose(
        area_jacobian,
        solution.inputs.sG * solution.inputs.spsi,
        rtol=2e-13,
    )


@pytest.mark.parametrize("order", [2, "r2", 3, "r3"])
def test_higher_order_request_is_explicitly_rejected(order):
    with pytest.raises(NotImplementedError, match="Second- and third-order"):
        standard_solution(order=order)


def test_invalid_order_and_inverse_mode_are_explicitly_rejected():
    with pytest.raises(ValueError, match="order must be"):
        standard_solution(order="fourth")
    with pytest.raises(NotImplementedError, match="Inverse"):
        standard_solution(solve_for="etabar")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"nphi": 2}, "nphi"),
        ({"nphi": True}, "nphi"),
        ({"order": 4}, "order"),
        ({"sG": 0}, "sG"),
        ({"spsi": 0}, "spsi"),
        ({"solve_for": "pressure"}, "solve_for"),
        ({"etabar": jnp.ones(2)}, "etabar"),
    ],
)
def test_input_model_validation(overrides, message):
    parameters = {
        "axis": qsc.Axis(rc=[1.0], zs=[0.0]),
        "etabar": -0.9,
    }
    parameters.update(overrides)
    with pytest.raises(ValueError, match=message):
        NearAxisInputs(**parameters)

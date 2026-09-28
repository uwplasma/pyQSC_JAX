import jax
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
    residual = sigma_residual(state, inputs=solution.inputs, geometry=solution.geometry)

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
        solution.X1c * solution.curvature, solution.inputs.etabar, rtol=2e-13
    )
    area_jacobian = solution.X1c * solution.Y1s - solution.X1s * solution.Y1c
    np.testing.assert_allclose(area_jacobian, solution.inputs.sG * solution.inputs.spsi, rtol=2e-13)


def test_invalid_order_is_explicitly_rejected():
    with pytest.raises(ValueError, match="order must be"):
        standard_solution(order="fourth")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"nphi": 2}, "nphi"),
        ({"nphi": True}, "nphi"),
        ({"order": 4}, "order"),
        ({"sG": 0}, "sG"),
        ({"spsi": 0}, "spsi"),
        ({"etabar": jnp.ones(2)}, "etabar"),
    ],
)
def test_input_model_validation(overrides, message):
    parameters = {"axis": qsc.Axis(rc=[1.0], zs=[0.0]), "etabar": -0.9}
    parameters.update(overrides)
    with pytest.raises(ValueError, match=message):
        NearAxisInputs(**parameters)


def test_diagnostics_are_lazy_unless_requested():
    kwargs = dict(rc=[1.0, 0.09], zs=[0.0, -0.09], nfp=2, etabar=0.95, order="r2", nphi=31)
    plain = qsc.Qsc(**kwargs)
    assert plain.mercier is None and plain.field_jet is None and plain.singularity is None
    assert plain.second_order.linear_report.condition_number is None
    assert plain.root_report.jacobian_condition_number is None

    stored = qsc.Qsc(**kwargs, diagnostics=True)
    assert stored.singularity is not None and stored.field_jet is not None
    assert stored.second_order.linear_report.condition_number is not None
    assert stored.root_report.jacobian_condition_number is not None
    np.testing.assert_allclose(plain.r_singularity, stored.r_singularity, rtol=1e-12)
    np.testing.assert_allclose(
        plain.grad_grad_B_axis, stored.grad_grad_B_axis, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(plain.DMerc_times_r2, stored.DMerc_times_r2, rtol=1e-12)
    attached = plain.with_diagnostics()
    np.testing.assert_allclose(attached.singularity.r_singularity, stored.r_singularity)
    with pytest.raises(AttributeError, match="singular"):
        _ = qsc.Qsc(**{**kwargs, "order": "r1"}).r_singularity


def test_solve_does_not_retrace_for_new_values_or_input_types():
    from pyqsc_jax.first_order import _solve

    axis = qsc.Axis(rc=[1.0, 0.09], zs=[0.0, -0.09], nfp=2)
    qsc.solve(axis=axis, etabar=0.95, nphi=23)
    size = _solve._cache_size()
    qsc.solve(axis=axis, etabar=jnp.asarray(0.9), nphi=23)
    qsc.solve(axis=axis, etabar=1, B0=np.float64(1.1), nphi=23)
    assert _solve._cache_size() == size


def test_solution_pytree_supports_ahead_of_time_lowering():
    def iota(etabar):
        return qsc.solve(axis=qsc.Axis(rc=[1.0, 0.09], zs=[0.0, -0.09], nfp=2), etabar=etabar)

    lowered = jax.jit(iota).lower(0.95)
    assert lowered.compile()(0.95).iota.shape == ()

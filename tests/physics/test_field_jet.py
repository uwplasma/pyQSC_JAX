import jax
import numpy as np
import pytest

import pyqsc_jax as qsc


def _vacuum_qa(nphi=61):
    return qsc.Qsc(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=0.64,
        B2c=-0.00322,
        nphi=nphi,
        order="r2",
    )


def _finite_pressure_current(nphi=61):
    return qsc.Qsc(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=0.9,
        B2c=-0.7,
        p2=-600000.0,
        nphi=nphi,
        order="r2",
    )


@pytest.mark.parametrize("factory", [_vacuum_qa, _finite_pressure_current])
def test_total_field_jet_satisfies_chain_rule_and_maxwell_identities(factory):
    solution = factory()
    jet = solution.field_jet

    assert jet is not None
    assert jet.field.shape == (61, 3)
    assert jet.gradient.shape == (61, 3, 3)
    assert jet.hessian.shape == (61, 3, 3, 3)
    assert jet.hessian_frenet.shape == (61, 3, 3, 3)
    assert jet.coordinate_jacobian.shape == (61, 3, 3)
    assert jet.coordinate_hessian.shape == (61, 3, 3, 3)
    assert jet.minimum_absolute_coordinate_jacobian > 0.9
    assert jet.maximum_field_error < 5e-15
    assert jet.maximum_gradient_error < 2e-8
    assert jet.maximum_divergence < 3e-8
    assert jet.maximum_derivative_asymmetry < 2e-13
    assert jet.maximum_divergence_gradient < 5e-5
    np.testing.assert_allclose(solution.grad_grad_B_axis, jet.hessian)
    np.testing.assert_allclose(solution.grad_grad_B, jet.hessian_frenet)


def test_vacuum_field_hessian_is_fully_symmetric_and_trace_free():
    solution = _vacuum_qa()
    hessian = solution.field_jet.hessian

    np.testing.assert_allclose(hessian, np.swapaxes(hessian, 1, 2), rtol=0, atol=7e-7)
    np.testing.assert_allclose(hessian, np.swapaxes(hessian, 1, 3), rtol=0, atol=7e-7)
    np.testing.assert_allclose(
        np.einsum("niik->nk", hessian),
        0,
        rtol=0,
        atol=7e-7,
    )


@pytest.mark.parametrize(
    "factory, expected_components, expected_inverse_scale",
    [
        (
            _vacuum_qa,
            {
                (0, 0, 0): [2.5058735043974697e-11, 1.3780320235299797, -3.0796390010619596],
                (0, 2, 1): [2.5345018639059774e-14, -0.9042329447022055, 0.04480778724398558],
                (1, 2, 2): [-1.0583277338937587, -1.0542622720207702, 1.4815487851166125],
            },
            2.8316255897272793,
        ),
        (
            _finite_pressure_current,
            {
                (0, 0, 0): [-1.0984223013409288e-11, -0.4764076511218376, 1.041130887765426],
                (0, 2, 1): [-1.044435577486018e-14, -0.4447191502214616, 0.220359403292567],
                (1, 2, 2): [1.2467274928318781, 1.4565818008885116, -0.5095116177456592],
            },
            2.031147679042703,
        ),
    ],
)
def test_field_hessian_matches_upstream_pyqsc(
    factory,
    expected_components,
    expected_inverse_scale,
):
    solution = factory()
    indices = [0, 15, 30]

    for component, expected in expected_components.items():
        np.testing.assert_allclose(
            np.asarray(solution.grad_grad_B)[indices, *component],
            expected,
            rtol=3e-6,
            atol=2e-6,
        )
    np.testing.assert_allclose(
        solution.grad_grad_B_inverse_scale_length,
        expected_inverse_scale,
        rtol=3e-7,
    )


def test_field_hessian_resolution_convergence():
    medium = _vacuum_qa(nphi=61)
    fine = _vacuum_qa(nphi=91)

    np.testing.assert_allclose(
        np.asarray(medium.grad_grad_B)[0],
        np.asarray(fine.grad_grad_B)[0],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        medium.grad_grad_B_inverse_scale_length_vs_varphi[0],
        fine.grad_grad_B_inverse_scale_length_vs_varphi[0],
        rtol=2e-6,
    )


def test_field_hessian_supports_jit_and_jvp():
    def component(etabar):
        solution = qsc.Qsc(
            rc=[1.0, 0.155, 0.0102],
            zs=[0.0, 0.154, 0.0111],
            nfp=2,
            etabar=etabar,
            B2c=-0.00322,
            nphi=31,
            order="r2",
        )
        return solution.grad_grad_B_axis[7, 0, 1, 2]

    eager_value = component(0.64)
    compiled_value = jax.jit(component)(0.64)
    value, tangent = jax.jvp(component, (0.64,), (1.0,))

    np.testing.assert_allclose(compiled_value, eager_value, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(value, eager_value, rtol=3e-12, atol=3e-12)
    assert np.isfinite(tangent)

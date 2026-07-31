from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.near_axis import near_axis

# Frozen BSD-2-Clause pyQSC reference samples at the audited upstream commit.
CASES = (
    (
        "qa",
        {},
        {
            "X3c1": [0.06254885838364596, 1.1028068624859182, 0.9898672691339223],
            "Y3c1": [0.0, 1.9611076246147592, 0.06254912908105097],
            "Y3s1": [0.263992436104027, 3.1822277655813185, 0.6000377597469493],
            "B0_order_a_squared_to_cancel": [
                0.257001365727628,
                3.7466692503729218,
                1.5413730743953158,
            ],
        },
    ),
    (
        "finite_pressure_current",
        {
            "rc": [1.0, 0.09],
            "zs": [0.0, -0.09],
            "nfp": 2,
            "etabar": 0.95,
            "I2": 0.9,
            "B2c": -0.7,
            "p2": -600000.0,
        },
        {
            "X3c1": [-0.7763858841222269, -0.2167234757709553, -0.4037215275576512],
            "Y3c1": [0.0, 0.13599120989448577, 0.02104082225854222],
            "Y3s1": [-1.2142017571662096, -0.27932779079566616, -0.18341957155186067],
            "B0_order_a_squared_to_cancel": [
                -1.9418435614778733,
                -0.4920849106382204,
                -0.5442441717107536,
            ],
        },
    ),
    (
        "qh",
        {
            "rc": [1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
            "zs": [0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
            "nfp": 4,
            "etabar": 1.569,
            "B2c": 0.1348,
        },
        {
            "X3c1": [-0.05934821467751999, -0.002180735261069853, -0.006774272974061107],
            "Y3c1": [0.0, 0.002774228833885528, 0.00030670030088582816],
            "Y3s1": [-0.15982340894062852, -0.002817420091098044, -0.002404111069876869],
            "B0_order_a_squared_to_cancel": [
                -0.1947843318576025,
                -0.0049574377809037395,
                -0.008071209239579269,
            ],
        },
    ),
)


def standard_solution(nphi=31, **kwargs):
    parameters = {
        "rc": [1.0, 0.155, 0.0102],
        "zs": [0.0, 0.154, 0.0111],
        "nfp": 2,
        "etabar": 0.64,
        "B2c": -0.00322,
        "nphi": nphi,
    }
    parameters.update(kwargs)
    return qsc.Qsc(**parameters)


@pytest.mark.parametrize(("_name", "kwargs", "reference"), CASES)
def test_third_order_matches_upstream_pyqsc(_name, kwargs, reference):
    solution = standard_solution(order="r3", nphi=61, **kwargs)
    indices = np.asarray([0, 15, 30])

    for name, expected in reference.items():
        np.testing.assert_allclose(
            np.asarray(getattr(solution, name))[indices],
            expected,
            rtol=3.0e-9,
            atol=5.0e-10,
        )


def test_third_order_structure_and_independent_constraints():
    solution = standard_solution(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=0.9,
        B2c=-0.7,
        p2=-600000.0,
        order="r3",
        nphi=61,
    )
    third = solution.third_order
    assert third is not None

    np.testing.assert_allclose(third.X3c1, solution.X1c * third.flux_constraint_coefficient)
    np.testing.assert_allclose(third.Y3c1, solution.Y1c * third.flux_constraint_coefficient)
    np.testing.assert_allclose(third.Y3s1, solution.Y1s * third.flux_constraint_coefficient)
    np.testing.assert_allclose(
        third.B0_order_a_squared_to_cancel,
        2 * solution.inputs.B0 * third.flux_constraint_coefficient,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    assert float(jnp.max(jnp.abs(third.flux_constraint_residual))) < 2.0e-13
    assert float(jnp.max(jnp.abs(third.consistency_error))) < 2.0e-10

    for name in (
        "X3s1",
        "Z3s1",
        "Z3c1",
        "X3s3",
        "X3c3",
        "Y3s3",
        "Y3c3",
        "Z3s3",
        "Z3c3",
    ):
        np.testing.assert_array_equal(getattr(third, name), jnp.zeros(solution.inputs.nphi))

    derivative = solution.geometry.d_d_varphi
    np.testing.assert_allclose(third.d_X3c1_d_varphi, derivative @ third.X3c1, atol=2.0e-12)
    np.testing.assert_allclose(third.d_Y3s1_d_varphi, derivative @ third.Y3s1, atol=2.0e-12)
    np.testing.assert_allclose(third.d_Y3c1_d_varphi, derivative @ third.Y3c1, atol=2.0e-12)


def test_qh_untwisting_and_compatibility_surface_include_r3():
    r3 = qsc.Qsc(
        rc=[1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
        zs=[0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
        nfp=4,
        etabar=1.569,
        B2c=0.1348,
        order="r3",
        nphi=61,
    )

    assert np.linalg.norm(np.asarray(r3.X3s1_untwisted)) > 0
    adapter = near_axis(
        rc=[1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
        zs=[0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
        nfp=4,
        etabar=1.569,
        B2c=0.1348,
        nphi=61,
        order="r3",
    )
    r = 0.02
    theta = 0.37
    x2, y2, z2 = near_axis(
        rc=[1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
        zs=[0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
        nfp=4,
        etabar=1.569,
        B2c=0.1348,
        nphi=61,
        order="r2",
    )._frenet_displacements(r, theta)
    x3, y3, z3 = adapter._frenet_displacements(r, theta)
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    cosine3 = jnp.cos(3 * theta)
    sine3 = jnp.sin(3 * theta)

    np.testing.assert_allclose(
        x3 - x2,
        r**3
        * (
            r3.X3c1_untwisted * cosine
            + r3.X3s1_untwisted * sine
            + r3.X3c3_untwisted * cosine3
            + r3.X3s3_untwisted * sine3
        ),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        y3 - y2,
        r**3
        * (
            r3.Y3c1_untwisted * cosine
            + r3.Y3s1_untwisted * sine
            + r3.Y3c3_untwisted * cosine3
            + r3.Y3s3_untwisted * sine3
        ),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(z3, z2, atol=2.0e-15)


def test_third_order_is_jittable_and_differentiable():
    def coefficient(etabar):
        return standard_solution(order="r3", nphi=31, etabar=etabar).X3c1[7]

    value = coefficient(jnp.asarray(-0.9))
    jitted = jax.jit(coefficient)(jnp.asarray(-0.9))
    tangent = jax.jvp(coefficient, (jnp.asarray(-0.9),), (jnp.asarray(1.0),))[1]
    step = 2.0e-5
    finite_difference = (coefficient(-0.9 + step) - coefficient(-0.9 - step)) / (2 * step)

    np.testing.assert_allclose(jitted, value, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(tangent, finite_difference, rtol=3.0e-5, atol=3.0e-7)


def test_third_order_requires_second_order_and_valid_adapter_order():
    r1 = standard_solution(order="r1")
    with pytest.raises(ValueError, match="second-order"):
        qsc.solve_third_order(r1)
    with pytest.raises(AttributeError, match="Lower-order"):
        _ = r1.X3c1
    with pytest.raises(ValueError, match="order must be"):
        near_axis(rc=[1.0], zs=[0.0], etabar=-0.9, order="r4")

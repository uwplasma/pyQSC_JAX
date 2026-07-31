from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.second_order import MU0


def finite_current_solution(*, I2=0.9, p2=-600000.0, nphi=31):
    return qsc.Qsc(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=I2,
        p2=p2,
        B2c=-0.7,
        nphi=nphi,
        order="r2",
    )


def test_covariant_and_enclosed_current_conversions_round_trip():
    I2 = 0.9
    radius = 0.08
    chi = -1
    current = qsc.enclosed_current_from_covariant(
        I2,
        formal_radius=radius,
        chi=chi,
    )

    np.testing.assert_allclose(current, 2 * np.pi * chi * I2 * radius**2 / MU0)
    np.testing.assert_allclose(
        qsc.covariant_current_from_enclosed(
            current,
            formal_radius=radius,
            chi=chi,
        ),
        I2,
    )
    derivative = jax.grad(
        lambda a: qsc.enclosed_current_from_covariant(
            I2,
            formal_radius=a,
            chi=chi,
        )
    )(radius)
    np.testing.assert_allclose(derivative, 4 * np.pi * chi * I2 * radius / MU0)


def test_positive_volume_source_matches_manuscript_coefficients():
    solution = finite_current_solution()
    radius = 0.07
    source = qsc.plasma_current_source(solution, formal_radius=radius)
    chi = solution.inputs.sG * solution.inputs.spsi
    expected_j = 2 * chi * solution.inputs.I2
    expected_C2 = solution.G2 + solution.N_helicity * solution.inputs.I2

    np.testing.assert_allclose(source.parallel_current_mu0, expected_j)
    np.testing.assert_allclose(source.C2, expected_C2)
    np.testing.assert_allclose(
        source.enclosed_toroidal_current,
        np.pi * radius**2 * expected_j / MU0,
    )
    np.testing.assert_allclose(
        source.w1,
        expected_j * solution.geometry.tangent_cartesian,
    )
    np.testing.assert_allclose(
        source.w2_cosine,
        source.wstar2_cosine
        - (expected_j * solution.geometry.curvature * solution.X1c)[:, None]
        * solution.geometry.tangent_cartesian,
    )
    np.testing.assert_allclose(source.w2_sine, source.wstar2_sine)
    assert source.chi == chi


def test_weighted_source_evaluation_has_regular_radial_power_and_batch_shape():
    solution = finite_current_solution(nphi=15)
    source = qsc.plasma_current_source(solution, formal_radius=0.05)
    radial = jnp.asarray([0.0, 0.01])
    theta = jnp.asarray([0.2, 0.7])
    weighted = qsc.evaluate_weighted_current(source, radial, theta)

    assert weighted.shape == (2, 15, 3)
    np.testing.assert_allclose(weighted[0], 0.0)
    direct = source.axis_length_per_radian * (
        radial[1] * source.w1
        + radial[1] ** 2 * (np.cos(theta[1]) * source.w2_cosine + np.sin(theta[1]) * source.w2_sine)
    )
    np.testing.assert_allclose(weighted[1], direct)
    scalar = jax.jit(qsc.evaluate_weighted_current)(source, 0.01, 0.3)
    assert scalar.shape == (15, 3)


def test_pressure_and_parallel_current_paths_remain_well_defined_at_I2_zero():
    pressure_only = finite_current_solution(I2=0.0)
    source = qsc.plasma_current_source(pressure_only, formal_radius=0.06)

    np.testing.assert_allclose(source.parallel_current_mu0, 0.0)
    np.testing.assert_allclose(source.w1, 0.0)
    assert np.max(np.abs(source.wstar2_cosine)) > 0
    assert np.max(np.abs(source.wstar2_sine)) > 0
    assert np.all(np.isfinite(source.wstar2_cosine))

    vacuum = finite_current_solution(I2=0.0, p2=0.0)
    vacuum_source = qsc.plasma_current_source(vacuum, formal_radius=0.06)
    np.testing.assert_allclose(vacuum_source.w1, 0.0)
    np.testing.assert_allclose(vacuum_source.wstar2_cosine, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(vacuum_source.wstar2_sine, 0.0, atol=1.0e-14)


def test_current_source_guards():
    with pytest.raises(ValueError, match="second-order"):
        qsc.plasma_current_source(
            qsc.Qsc(
                rc=[1.0, 0.045],
                zs=[0.0, -0.045],
                nfp=3,
                etabar=-0.9,
                order="r1",
            ),
            formal_radius=0.1,
        )
    for value in (0.0, -0.1):
        with pytest.raises(ValueError, match="positive"):
            qsc.enclosed_current_from_covariant(
                1.0,
                formal_radius=value,
                chi=1,
            )
    with pytest.raises(ValueError, match="scalar"):
        qsc.covariant_current_from_enclosed(
            1.0,
            formal_radius=[0.1],
            chi=1,
        )
    for function in (
        qsc.enclosed_current_from_covariant,
        qsc.covariant_current_from_enclosed,
    ):
        with pytest.raises(ValueError, match="chi"):
            if function is qsc.enclosed_current_from_covariant:
                function(1.0, formal_radius=0.1, chi=0)
            else:
                function(1.0, formal_radius=0.1, chi=0)

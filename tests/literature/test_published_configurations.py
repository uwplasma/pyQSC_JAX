"""Published configurations and classical closed-form limits.

- Landreman, Sengupta & Plunk, J. Plasma Phys. 85, 905850103 (2019), section 5.1 (O(r) QA).
- Landreman & Sengupta, J. Plasma Phys. 85, 815850601 (2019), sections 5.1, 5.2, 5.5 (O(r^2)).
  Reference scalars are pyQSC's reproduction of those configurations (commit cd75359).
- Thin circular current ring (Shafranov): the local-induction field carries log(8 R / a).
- Straight and curved finite conductors: closed-form interior field gradients.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
import pyqsc_jax.plasma as qp
from fixtures import solve_configuration
from pyqsc_jax.plasma import _transverse_plasma_hessian

SECOND_ORDER = {
    "qa": {
        "iota": -0.42047335182825857,
        "beta_1s": 0.0,
        "G2": 0.0,
        "B20_mean": 0.16212431504754432,
        "B20_residual": 0.13498174214090364,
        "B20_variation": 0.3740338506893984,
    },
    "qh": {
        "iota": -1.1441369511851485,
        "beta_1s": 0.0,
        "G2": 0.0,
        "B20_mean": 1.312575397053029,
        "B20_residual": 0.032260200125823604,
        "B20_variation": 0.11724965804270937,
    },
    "finite_pressure_current": {
        "iota": 0.9596981597369478,
        "beta_1s": 3.0336182742616837,
        "G2": -0.09758153451238105,
        "B20_mean": 1.8129933830726364,
        "B20_residual": 0.22096470852551833,
        "B20_variation": 0.6235580990378657,
    },
}


def test_landreman_sengupta_plunk_2019_first_order_qa():
    """LSP 2019 sec. 5.1 (nfp = 3 QA, etabar = -0.9): iota = 0.41830690943."""

    solution = solve_configuration("r1_qa", nphi=31, order="r1")
    np.testing.assert_allclose(solution.iota, 0.41830690943386617, rtol=2e-13)


@pytest.mark.parametrize("name", ["qa", "qh", "finite_pressure_current"])
def test_landreman_sengupta_2019_second_order_scalars(name):
    """LS 2019 sec. 5.1 / 5.2 / 5.5: iota, beta_1s, G2 and B20 mean/residual/variation."""

    solution = solve_configuration(name, nphi=31)
    for key, value in SECOND_ORDER[name].items():
        np.testing.assert_allclose(getattr(solution, key), value, rtol=3e-12, atol=3e-12)


def test_circular_axis_recovers_thin_ring_local_induction():
    """Planar circle: the regularized integral vanishes and the binormal kernel is log(8/a)."""

    solution = qsc.Qsc(rc=[1.0], zs=[0.0], nfp=1, etabar=1.0, I2=0.1, nphi=31, order="r2")
    radius = 0.05
    plasma = qsc.plasma_field_on_axis(solution, formal_radius=radius)
    binormal_kernel = np.sum(
        np.asarray(plasma.matched_axis_and_core) * np.asarray(solution.geometry.binormal_cartesian),
        axis=-1,
    )

    assert np.max(np.abs(qsc.regularized_axis_integral(solution))) < 1.0e-13
    np.testing.assert_allclose(plasma.core_binormal_constant, 0.0)
    np.testing.assert_allclose(plasma.core_normal_constant, 0.0)
    np.testing.assert_allclose(binormal_kernel, np.log(8 / radius), rtol=2.0e-13)


def test_straight_elliptical_channel_gradient_is_closed_form():
    """Uniform current in a straight channel: div B = 0 and curl B = mu0 j (circle and ellipse)."""

    j = 1.7
    gradient = qp.elliptical_channel_gradient(1.0, 0.0, parallel_current_mu0=j, chi=1)
    np.testing.assert_allclose(gradient, [[0, 0, 0], [0, 0, -j / 2], [0, j / 2, 0]])

    j = -0.8
    gradient = qp.elliptical_channel_gradient(1.6, -0.35, parallel_current_mu0=j, chi=-1)
    np.testing.assert_allclose(np.trace(gradient), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(gradient[2, 1] - gradient[1, 2], j)
    assert gradient[1, 1] == -gradient[2, 2]
    assert not np.isclose(gradient[2, 1], -gradient[1, 2])


def test_circular_curved_channel_recovers_finite_conductor_hessian():
    """Curved circular conductor, Eq. (215): Hessian entries are -j kappa / 8 and -3 j kappa / 8."""

    solution = qsc.Qsc(rc=[1.0], zs=[0.0], nfp=1, etabar=1.0, I2=0.1, nphi=31, order="r2")
    zeros = jnp.zeros_like(solution.X20)
    shapes = ("X20", "X2c", "X2s", "Y20", "Y2c", "Y2s", "Z20", "Z2c", "Z2s")
    circular = replace(
        solution, second_order=replace(solution.second_order, **dict.fromkeys(shapes, zeros))
    )
    source = qsc.plasma_current_source(solution, formal_radius=0.1)
    zero_vector = jnp.zeros_like(source.wstar2_cosine)
    curvature_only = replace(source, wstar2_cosine=zero_vector, wstar2_sine=zero_vector)

    transverse = _transverse_plasma_hessian(circular, curvature_only)
    coefficient = source.parallel_current_mu0 * solution.geometry.curvature / 8
    expected = jnp.zeros_like(transverse)
    expected = expected.at[:, 0, 0, 2].set(-coefficient)
    expected = expected.at[:, 0, 1, 1].set(-coefficient)
    expected = expected.at[:, 1, 0, 1].set(-coefficient)
    expected = expected.at[:, 1, 1, 2].set(-3 * coefficient)
    np.testing.assert_allclose(transverse, expected, atol=2.0e-15)

import numpy as np

import pyqsc_jax as qsc
from pyqsc_jax.near_axis import near_axis


def test_standard_solution_matches_legacy_and_upstream_reference():
    parameters = {
        "rc": [1.0, 0.045],
        "zs": [0.0, -0.045],
        "nfp": 3,
        "etabar": -0.9,
        "nphi": 31,
    }
    solution = qsc.Qsc(**parameters)
    legacy = near_axis(**parameters)

    np.testing.assert_allclose(solution.iota, 0.41830690943386617, rtol=2e-13)
    np.testing.assert_allclose(solution.iota, legacy.iota, rtol=2e-13)
    np.testing.assert_allclose(solution.sigma, legacy.sigma, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(solution.B_axis, legacy.B_axis.T, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        solution.grad_B_axis,
        legacy.grad_B_axis.T,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(solution.L_grad_B, legacy.L_grad_B, rtol=2e-13)


def test_finite_current_and_sigma0_match_upstream_reference():
    solution = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=31,
        I2=0.3,
        B0=1.2,
        sigma0=0.1,
    )

    np.testing.assert_allclose(solution.iota, 0.5981040523602934, rtol=2e-13)
    np.testing.assert_allclose(solution.iotaN, 0.5981040523602934, rtol=2e-13)
    np.testing.assert_allclose(solution.G0, 1.2108964178133113, rtol=2e-13)
    np.testing.assert_allclose(np.max(np.abs(solution.sigma)), 1.1527807285268685, rtol=2e-13)
    np.testing.assert_allclose(solution.mean_elongation, 2.300022135471462, rtol=2e-13)


def test_qh_topology_matches_upstream_reference():
    solution = qsc.Qsc(
        rc=[1.0, 0.265],
        zs=[0.0, -0.21],
        nfp=4,
        etabar=-0.9,
        nphi=31,
    )

    assert int(solution.helicity) == -1
    np.testing.assert_allclose(solution.iota, 3.0598175213150203, rtol=2e-13)
    np.testing.assert_allclose(solution.iotaN, -0.9401824786849797, rtol=2e-13)
    np.testing.assert_allclose(solution.G0, 1.3885479374943943, rtol=2e-13)
    np.testing.assert_allclose(np.max(np.abs(solution.sigma)), 0.23608475507759125, rtol=2e-13)

import numpy as np
import pytest

import pyqsc_jax as qsc


def test_mercier_diagnostics_match_upstream_finite_pressure_reference():
    solution = qsc.Qsc(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=0.9,
        B2c=-0.7,
        p2=-600000.0,
        nphi=61,
        order="r2",
    )

    expected = {
        "d2_volume_d_psi2": -121.81059604927127,
        "DGeod_times_r2": -0.1210731174740698,
        "DWell_times_r2": 0.06028523900360924,
        "DMerc_times_r2": -0.06078787847046056,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(getattr(solution, name), value, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(getattr(solution.mercier, name), value, rtol=3e-12, atol=3e-12)


def test_vacuum_mercier_pressure_terms_vanish():
    solution = qsc.Qsc(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=0.64,
        B2c=-0.00322,
        nphi=61,
        order="r2",
    )

    np.testing.assert_allclose(solution.d2_volume_d_psi2, 23.98845128286806, rtol=3e-12)
    assert solution.DGeod_times_r2 == 0
    assert solution.DWell_times_r2 == 0
    assert solution.DMerc_times_r2 == 0


def test_second_order_diagnostics_reject_first_order_solution():
    solution = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=31,
    )

    with pytest.raises(ValueError, match="second-order"):
        qsc.mercier_diagnostics(solution)
    with pytest.raises(ValueError, match="second-order"):
        qsc.total_field_jet(solution)
    with pytest.raises(AttributeError, match="Mercier"):
        _ = solution.DMerc_times_r2
    with pytest.raises(AttributeError, match="second-derivative"):
        _ = solution.grad_grad_B_axis

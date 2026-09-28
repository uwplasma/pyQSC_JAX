"""Frozen reference arrays.

Unless stated otherwise the values were generated with landreman/pyQSC (BSD-2-Clause) at
audited commit cd75359ea47548d5db7ccb458c100085c04ba1bc and are compared at the sample
indices listed in each test.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyqsc_jax as qsc
from fixtures import solve_configuration
from pyqsc_jax import Axis
from pyqsc_jax.geometry import compute_axis_geometry

QH_R1 = dict(rc=[1.0, 0.265], zs=[0.0, -0.21], nfp=4, etabar=-0.9)


def test_first_order_axis_and_sigma_match_pyqsc():
    """r1 QA axis length, curvature/torsion extrema and max |sigma| match pyQSC."""

    solution = solve_configuration("r1_qa", nphi=31, order="r1")
    np.testing.assert_allclose(solution.iota, 0.41830690943386617, rtol=2e-13)
    np.testing.assert_allclose(solution.geometry.axis_length, 6.340238817434161, rtol=2e-13)
    np.testing.assert_allclose(
        [np.min(solution.curvature), np.max(solution.curvature)],
        [0.5956163516982903, 1.3060121594235536],
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        [np.min(solution.torsion), np.max(solution.torsion)],
        [-2.272197039914583, 0.6057564303422814],
        rtol=2e-13,
    )
    np.testing.assert_allclose(np.max(np.abs(solution.sigma)), 0.9920706836908618, rtol=2e-13)


def test_finite_current_and_sigma0_match_pyqsc():
    """I2, B0 != 1 and sigma0 != 0: iota, G0, sigma, elongation and grad B samples match."""

    solution = solve_configuration("r1_qa", nphi=31, order="r1", I2=0.3, B0=1.2, sigma0=0.1)

    np.testing.assert_allclose(solution.iota, 0.5981040523602934, rtol=2e-13)
    np.testing.assert_allclose(solution.iotaN, 0.5981040523602934, rtol=2e-13)
    np.testing.assert_allclose(solution.G0, 1.2108964178133113, rtol=2e-13)
    np.testing.assert_allclose(np.max(np.abs(solution.sigma)), 1.1527807285268685, rtol=2e-13)
    np.testing.assert_allclose(solution.mean_elongation, 2.300022135471462, rtol=2e-13)
    expected_gradient_components = {
        (0, 1): [-1.4189028123671708, -0.024458076547534513, 0.31866877632846324],
        (1, 0): [-1.4958375512360236, -0.03652380306707692, 0.40219561346636035],
        (0, 2): [1.248855714330292, 0.24413995843927835, -0.7230617247040809],
        (2, 0): [0.6533238467899136, -0.2490021462405346, -1.0300929188685888],
    }
    for component, expected in expected_gradient_components.items():
        np.testing.assert_allclose(
            np.asarray(solution.grad_B_axis)[[0, 7, 15], *component],
            expected,
            rtol=3e-12,
            atol=3e-12,
        )


def test_qh_helicity_and_iota_match_pyqsc():
    """A helical axis has frame helicity -1 and iotaN = iota - N with pyQSC's sign convention."""

    solution = qsc.Qsc(**QH_R1, nphi=31)
    assert int(solution.helicity) == -1
    np.testing.assert_allclose(solution.iota, 3.0598175213150203, rtol=2e-13)
    np.testing.assert_allclose(solution.iotaN, -0.9401824786849797, rtol=2e-13)
    np.testing.assert_allclose(solution.G0, 1.3885479374943943, rtol=2e-13)
    np.testing.assert_allclose(np.max(np.abs(solution.sigma)), 0.23608475507759125, rtol=2e-13)


def test_asymmetric_axis_geometry_matches_independent_fortran_reference():
    """Curvature, torsion and varphi of a non-symmetric axis match an independent Fortran code."""

    axis = Axis(
        rc=[1.3, 0.3, 0.01, -0.001],
        zs=[0.0, 0.4, -0.02, -0.003],
        rs=[0.0, -0.1, -0.03, 0.002],
        zc=[0.3, 0.2, 0.04, 0.004],
        nfp=5,
    )
    geometry = compute_axis_geometry(axis, nphi=15)
    curvature = [
        2.10743037699653, 2.33190181686696, 1.83273654023051, 1.81062232906827,
        2.28640008392347, 1.76919841474321, 0.919988560478029, 0.741327470169023,
        1.37147330126897, 2.64680884158075, 3.39786486424852, 2.47005615416209,
        1.50865425515356, 1.18136509189105, 1.42042418970102,
    ]  # fmt: skip
    torsion = [
        -0.167822738386845, -0.0785778346620885, -1.02205137493593, -2.05213528002946,
        -0.964613202459108, -0.593496282035916, -2.15852857178204, -3.72911055219339,
        -1.9330792779459, -1.53882290974916, -1.42156496444929, -1.11381642382793,
        -0.92608309386204, -0.868339812017432, -0.57696266498748,
    ]  # fmt: skip
    varphi = [
        0.0, 0.084185130335249, 0.160931495903817, 0.232881563535092, 0.300551168190665,
        0.368933497012765, 0.444686439112853, 0.528001290336008, 0.612254611059372,
        0.691096975269652, 0.765820243301147, 0.846373713025902, 0.941973362938683,
        1.05053459351092, 1.15941650366667,
    ]  # fmt: skip

    np.testing.assert_allclose(geometry.curvature, curvature, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.torsion, torsion, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.varphi, varphi, rtol=2e-13, atol=2e-13)


SECOND_ORDER_SAMPLES = {
    "qa": {
        "X20": [-0.12607391440629268, 1.277313706167806, -1.1433646049834585],
        "Y20": [3.6484581072275835e-14, -0.061041743645613045, -0.28837977142377097],
        "B20": [0.03617846642088235, 0.08753146934212608, 0.3504652075357061],
    },
    "finite_pressure_current": {
        "X20": [1.0359701826468048, 0.9961323682664798, 1.8900857033253329],
        "Y20": [-2.4299426345854533e-15, -0.9136862473826926, -0.30321915705664426],
        "B20": [2.070309301450683, 1.8875840576321603, 1.4467512024128548],
    },
    "qh": {
        "X20": [-0.18543205322365838, 0.7244473635400297, 0.8388295002982797],
        "Y20": [-1.0252483956584066e-14, -0.07290690883420155, -0.012145324323969307],
        "B20": [1.2242278807890326, 1.3246881250407083, 1.3414775388313407],
    },
}

THIRD_ORDER_SAMPLES = {
    "qa": {
        "X3c1": [0.06254885838364596, 1.1028068624859182, 0.9898672691339223],
        "Y3c1": [0.0, 1.9611076246147592, 0.06254912908105097],
        "Y3s1": [0.263992436104027, 3.1822277655813185, 0.6000377597469493],
        "B0_order_a_squared_to_cancel": [0.257001365727628, 3.7466692503729218, 1.5413730743953158],
    },
    "finite_pressure_current": {
        "X3c1": [-0.7763858841222269, -0.2167234757709553, -0.4037215275576512],
        "Y3c1": [0.0, 0.13599120989448577, 0.02104082225854222],
        "Y3s1": [-1.2142017571662096, -0.27932779079566616, -0.18341957155186067],
        "B0_order_a_squared_to_cancel": [
            -1.9418435614778733,
            -0.4920849106382204,
            -0.5442441717107536,
        ],
    },
    "qh": {
        "X3c1": [-0.05934821467751999, -0.002180735261069853, -0.006774272974061107],
        "Y3c1": [0.0, 0.002774228833885528, 0.00030670030088582816],
        "Y3s1": [-0.15982340894062852, -0.002817420091098044, -0.002404111069876869],
        "B0_order_a_squared_to_cancel": [
            -0.1947843318576025,
            -0.0049574377809037395,
            -0.008071209239579269,
        ],
    },
}

SINGULARITY = {
    "qa": (
        0.2257896241404959,
        [0.7304158315686611, 0.2505553226671735, 0.2935268282653716, 0.23909537193609554],
    ),
    "finite_pressure_current": (
        0.22161114270880408,
        [0.7427937841922576, 0.3271954156702964, 0.2659539057552679, 0.3034006922051161],
    ),
    "qh": (
        0.3583276532138262,
        [0.8681431131590702, 0.4034129514765815, 0.3728231585587465, 0.39141407123791455],
    ),
}

FIELD_HESSIAN = {
    "qa": (
        {
            (0, 0, 0): [2.5058735043974697e-11, 1.3780320235299797, -3.0796390010619596],
            (0, 2, 1): [2.5345018639059774e-14, -0.9042329447022055, 0.04480778724398558],
            (1, 2, 2): [-1.0583277338937587, -1.0542622720207702, 1.4815487851166125],
        },
        2.8316255897272793,
    ),
    "finite_pressure_current": (
        {
            (0, 0, 0): [-1.0984223013409288e-11, -0.4764076511218376, 1.041130887765426],
            (0, 2, 1): [-1.044435577486018e-14, -0.4447191502214616, 0.220359403292567],
            (1, 2, 2): [1.2467274928318781, 1.4565818008885116, -0.5095116177456592],
        },
        2.031147679042703,
    ),
}


@pytest.mark.parametrize("name", ["qa", "finite_pressure_current", "qh"])
def test_second_and_third_order_arrays_match_pyqsc(name):
    """X20/Y20/B20 (nphi = 31), O(r^3) shapes and r_singularity (nphi = 61) match pyQSC."""

    r2 = solve_configuration(name, nphi=31)
    for key, expected in SECOND_ORDER_SAMPLES[name].items():
        np.testing.assert_allclose(
            np.asarray(getattr(r2, key))[[0, 7, 15]], expected, rtol=3e-11, atol=3e-11
        )

    r3 = solve_configuration(name, nphi=61, order="r3")
    for key, expected in THIRD_ORDER_SAMPLES[name].items():
        np.testing.assert_allclose(
            np.asarray(getattr(r3, key))[[0, 15, 30]], expected, rtol=3.0e-9, atol=5.0e-10
        )

    minimum, samples = SINGULARITY[name]
    np.testing.assert_allclose(r3.r_singularity, minimum, rtol=0, atol=5e-8)
    np.testing.assert_allclose(
        np.asarray(r3.r_singularity_vs_varphi)[[0, 15, 30, 45]], samples, rtol=0, atol=5e-8
    )
    np.testing.assert_allclose(r3.inv_r_singularity_vs_varphi, 1 / r3.r_singularity_vs_varphi)

    if name in FIELD_HESSIAN:
        components, inverse_scale = FIELD_HESSIAN[name]
        for component, expected in components.items():
            np.testing.assert_allclose(
                np.asarray(r3.grad_grad_B)[[0, 15, 30], *component], expected, rtol=3e-6, atol=2e-6
            )
        np.testing.assert_allclose(r3.grad_grad_B_inverse_scale_length, inverse_scale, rtol=3e-7)


def test_mercier_diagnostics_match_pyqsc():
    """Finite-pressure Mercier terms and the vacuum V'' match pyQSC at nphi = 61."""

    solution = solve_configuration("finite_pressure_current", nphi=61)
    expected = {
        "d2_volume_d_psi2": -121.81059604927127,
        "DGeod_times_r2": -0.1210731174740698,
        "DWell_times_r2": 0.06028523900360924,
        "DMerc_times_r2": -0.06078787847046056,
    }
    diagnostics = qsc.mercier_diagnostics(solution)
    for name, value in expected.items():
        np.testing.assert_allclose(getattr(solution, name), value, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(getattr(diagnostics, name), value, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(
        solve_configuration("qa", nphi=61).d2_volume_d_psi2, 23.98845128286806, rtol=3e-12
    )


def test_B2c_optimum_regression():
    """The closed-form B2c minimizing the B20 residual of LS2019 QA stays at -0.491726836."""

    result = qsc.optimize_B2c(solve_configuration("qa", nphi=31))
    np.testing.assert_allclose(result.B2c_optimal, -0.49172683641534204, rtol=3.0e-12)


@pytest.mark.physics
def test_documented_plasma_field_cases():
    """Documented plasma-field fractions (this package, not pyQSC) stay at their stated values.

    Circular channel: |B_p|/|B| > 30 % everywhere, mean 0.33257, I = 2 pi I2 a^2 / mu0.
    Database 52521 (pressure-only, nonplanar): mean fraction 1.88960e-3 with > 10 % variation.
    """

    solution = solve_configuration("plasma_dominant_channel", nphi=61)
    plasma = qsc.plasma_field_on_axis(solution, formal_radius=0.2, angular_resolution=64)
    fraction = np.linalg.norm(plasma.field, axis=-1) / np.linalg.norm(solution.B_axis, axis=-1)
    assert np.min(fraction) > 0.30
    np.testing.assert_allclose(np.mean(fraction), 0.3325737905856357, rtol=2.0e-12)
    assert 0.2 < float(solution.r_singularity)
    np.testing.assert_allclose(
        plasma.current_source.enclosed_toroidal_current, 840000.0, rtol=2e-13
    )

    solution = solve_configuration("plasma_stellarator", nphi=121)
    plasma = qsc.plasma_field_on_axis(solution, formal_radius=0.15, angular_resolution=128)
    plasma_norm = np.linalg.norm(np.asarray(plasma.field), axis=-1)
    fraction = plasma_norm / np.linalg.norm(np.asarray(solution.B_axis), axis=-1)
    assert float(solution.inputs.I2) == 0.0
    assert np.sqrt(np.mean(np.asarray(solution.torsion) ** 2)) > 0.5
    assert np.ptp(plasma_norm) / np.mean(plasma_norm) > 0.1
    assert 0.15 / float(solution.r_singularity) < 0.4
    np.testing.assert_allclose(np.mean(fraction), 0.0018895978020422664, rtol=3.0e-12)

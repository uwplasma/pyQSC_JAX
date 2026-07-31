import numpy as np

from pyqsc_jax import Axis
from pyqsc_jax.geometry import compute_axis_geometry
from pyqsc_jax.near_axis import near_axis


def test_geometry_matches_legacy_first_order_baseline():
    parameters = {
        "rc": [1.0, 0.045],
        "zs": [0.0, -0.045],
        "nfp": 3,
        "nphi": 31,
    }
    geometry = compute_axis_geometry(
        Axis(rc=parameters["rc"], zs=parameters["zs"], nfp=parameters["nfp"]),
        nphi=parameters["nphi"],
    )
    legacy = near_axis(etabar=-0.9, **parameters)

    np.testing.assert_allclose(geometry.samples.R, legacy.R0, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.samples.Z, legacy.Z0, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.axis_length, legacy.axis_length, rtol=2e-13)
    np.testing.assert_allclose(geometry.curvature, legacy.curvature, rtol=2e-13)
    np.testing.assert_allclose(geometry.torsion, legacy.torsion, rtol=2e-13)
    np.testing.assert_allclose(geometry.varphi, legacy.varphi, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.normal_cylindrical[:, 0], legacy.normal_R, rtol=2e-13)
    np.testing.assert_allclose(geometry.normal_cylindrical[:, 1], legacy.normal_phi, rtol=2e-13)
    np.testing.assert_allclose(geometry.normal_cylindrical[:, 2], legacy.normal_z, rtol=2e-13)


def test_asymmetric_geometry_matches_independent_fortran_reference():
    axis = Axis(
        rc=[1.3, 0.3, 0.01, -0.001],
        zs=[0.0, 0.4, -0.02, -0.003],
        rs=[0.0, -0.1, -0.03, 0.002],
        zc=[0.3, 0.2, 0.04, 0.004],
        nfp=5,
    )
    geometry = compute_axis_geometry(axis, nphi=15)
    curvature = [
        2.10743037699653,
        2.33190181686696,
        1.83273654023051,
        1.81062232906827,
        2.28640008392347,
        1.76919841474321,
        0.919988560478029,
        0.741327470169023,
        1.37147330126897,
        2.64680884158075,
        3.39786486424852,
        2.47005615416209,
        1.50865425515356,
        1.18136509189105,
        1.42042418970102,
    ]
    torsion = [
        -0.167822738386845,
        -0.0785778346620885,
        -1.02205137493593,
        -2.05213528002946,
        -0.964613202459108,
        -0.593496282035916,
        -2.15852857178204,
        -3.72911055219339,
        -1.9330792779459,
        -1.53882290974916,
        -1.42156496444929,
        -1.11381642382793,
        -0.92608309386204,
        -0.868339812017432,
        -0.57696266498748,
    ]
    varphi = [
        0.0,
        0.084185130335249,
        0.160931495903817,
        0.232881563535092,
        0.300551168190665,
        0.368933497012765,
        0.444686439112853,
        0.528001290336008,
        0.612254611059372,
        0.691096975269652,
        0.765820243301147,
        0.846373713025902,
        0.941973362938683,
        1.05053459351092,
        1.15941650366667,
    ]

    np.testing.assert_allclose(geometry.curvature, curvature, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.torsion, torsion, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.varphi, varphi, rtol=2e-13, atol=2e-13)


def test_qh_frame_helicity_matches_upstream_convention():
    geometry = compute_axis_geometry(
        Axis(rc=[1.0, 0.265], zs=[0.0, -0.21], nfp=4),
        nphi=31,
    )

    assert int(geometry.frame_helicity) == -1

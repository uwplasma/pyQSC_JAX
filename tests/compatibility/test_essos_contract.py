import pytest

from pyqsc_jax.near_axis import near_axis


def test_legacy_import_and_essos_on_axis_contract():
    field = near_axis(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        etabar=-0.9,
        nfp=3,
        nphi=31,
    )

    attributes = {
        "B0",
        "nfp",
        "nphi",
        "phi",
        "varphi",
        "R0",
        "Z0",
        "B_axis",
        "grad_B_axis",
        "axis_length",
        "iota",
        "iotaN",
        "curvature",
        "torsion",
        "elongation",
        "L_grad_B",
        "x",
        "dofs",
        "rc",
        "zs",
        "etabar",
        "sigma0",
        "I2",
        "spsi",
        "sG",
    }
    methods = {
        "AbsB",
        "B_covariant",
        "B_contravariant",
        "B_mag",
        "jacobian",
        "get_boundary",
        "Frenet_to_cylindrical",
        "phi_of_theta_varphi",
    }

    assert all(hasattr(field, name) for name in attributes)
    assert all(callable(getattr(field, name)) for name in methods)


@pytest.mark.xfail(
    strict=True,
    reason="Plotting exists only on the unmerged ESSOS bridge branch.",
)
def test_legacy_plot_method_is_available():
    field = near_axis()

    assert callable(field.plot)

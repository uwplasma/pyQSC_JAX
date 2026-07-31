from __future__ import annotations

import numpy as np
import pytest

import pyqsc_jax as qsc


def resolved_volume_biot_savart(
    solution,
    formal_radius,
    *,
    radial_resolution=16,
    angular_resolution=64,
):
    """Independent midpoint/Gauss volume integral at the first axis point."""

    source = qsc.plasma_current_source(
        solution,
        formal_radius=formal_radius,
    )
    geometry = solution.geometry
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(radial_resolution)
    radial = 0.5 * formal_radius * (radial_nodes + 1)
    radial_weights = 0.5 * formal_radius * radial_weights
    theta = 2 * np.pi * np.arange(angular_resolution) / angular_resolution
    theta_weight = 2 * np.pi / angular_resolution
    cosine = np.cos(theta)
    sine = np.sin(theta)
    cosine2 = np.cos(2 * theta)
    sine2 = np.sin(2 * theta)

    tangent = np.asarray(geometry.tangent_cartesian)
    normal = np.asarray(geometry.normal_cartesian)
    binormal = np.asarray(geometry.binormal_cartesian)
    first_order = (
        np.asarray(solution.X1c)[:, None, None] * cosine[None, :, None] * normal[:, None, :]
        + (
            np.asarray(solution.Y1s)[:, None, None] * sine[None, :, None]
            + np.asarray(solution.Y1c)[:, None, None] * cosine[None, :, None]
        )
        * binormal[:, None, :]
    )
    X2 = (
        np.asarray(solution.X20)[:, None]
        + np.asarray(solution.X2c)[:, None] * cosine2
        + np.asarray(solution.X2s)[:, None] * sine2
    )
    Y2 = (
        np.asarray(solution.Y20)[:, None]
        + np.asarray(solution.Y2c)[:, None] * cosine2
        + np.asarray(solution.Y2s)[:, None] * sine2
    )
    Z2 = (
        np.asarray(solution.Z20)[:, None]
        + np.asarray(solution.Z2c)[:, None] * cosine2
        + np.asarray(solution.Z2s)[:, None] * sine2
    )
    second_order = (
        X2[:, :, None] * normal[:, None, :]
        + Y2[:, :, None] * binormal[:, None, :]
        + Z2[:, :, None] * tangent[:, None, :]
    )
    source_position = (
        np.asarray(geometry.position_cartesian)[:, None, None, :]
        + radial[None, :, None, None] * first_order[:, None, :, :]
        + radial[None, :, None, None] ** 2 * second_order[:, None, :, :]
    )
    weighted_current = float(source.axis_length_per_radian) * (
        radial[None, :, None, None] * np.asarray(source.w1)[:, None, None, :]
        + radial[None, :, None, None] ** 2
        * (
            cosine[None, None, :, None] * np.asarray(source.w2_cosine)[:, None, None, :]
            + sine[None, None, :, None] * np.asarray(source.w2_sine)[:, None, None, :]
        )
    )
    displacement = np.asarray(geometry.position_cartesian[0]) - source_position
    kernel = (
        np.cross(weighted_current, displacement)
        / np.linalg.norm(
            displacement,
            axis=-1,
        )[..., None]
        ** 3
    )
    cylindrical_period = 2 * np.pi / solution.inputs.axis.nfp
    d_phi = cylindrical_period / solution.inputs.nphi
    varphi_weights = np.asarray(geometry.d_varphi_d_phi) * d_phi
    return np.sum(
        kernel
        * varphi_weights[:, None, None, None]
        * radial_weights[None, :, None, None]
        * theta_weight,
        axis=(0, 1, 2),
    ) / (4 * np.pi)


@pytest.mark.literature
@pytest.mark.slow
def test_matched_field_converges_to_resolved_volume_current_biot_savart():
    solution = qsc.Qsc(
        rc=[1.0],
        zs=[0.0],
        nfp=1,
        etabar=1.0,
        I2=0.1,
        nphi=301,
        order="r2",
    )
    radii = (0.1, 0.07)
    errors = []
    scaled_errors = []
    for radius in radii:
        direct = resolved_volume_biot_savart(solution, radius)
        asymptotic = np.asarray(
            qsc.plasma_field_on_axis(
                solution,
                formal_radius=radius,
            ).field[0]
        )
        error = np.linalg.norm(direct - asymptotic)
        errors.append(error)
        scaled_errors.append(error / (radius**4 * abs(np.log(radius))))

    assert errors[1] < 0.35 * errors[0]
    assert max(scaled_errors) / min(scaled_errors) < 1.2
    assert errors[0] < 1.0e-5

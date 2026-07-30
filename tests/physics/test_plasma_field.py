from __future__ import annotations

import jax
import numpy as np
import pytest

import pyqsc_jax as qsc


def circular_solution(*, I2=0.1, p2=0.0, nphi=31):
    return qsc.Qsc(
        rc=[1.0],
        zs=[0.0],
        nfp=1,
        etabar=1.0,
        I2=I2,
        p2=p2,
        nphi=nphi,
        order="r2",
    )


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


def test_periodic_regularized_integral_and_circular_local_induction_limit():
    solution = circular_solution()
    radius = 0.05
    integral = qsc.regularized_axis_integral(solution)
    plasma = qsc.plasma_field_on_axis(solution, formal_radius=radius)
    binormal_kernel = np.sum(
        np.asarray(plasma.matched_axis_and_core) * np.asarray(solution.geometry.binormal_cartesian),
        axis=-1,
    )

    assert np.max(np.abs(integral)) < 1.0e-13
    np.testing.assert_allclose(plasma.core_binormal_constant, 0.0)
    np.testing.assert_allclose(plasma.core_normal_constant, 0.0)
    np.testing.assert_allclose(binormal_kernel, np.log(8 / radius), rtol=2.0e-13)
    np.testing.assert_allclose(
        solution.inputs.I2 * radius**2 / 2 * binormal_kernel,
        solution.inputs.I2 * radius**2 / 2 * np.log(8 / radius),
    )


def test_matching_reference_length_cancels_exactly():
    solution = finite_current_solution()
    source = qsc.plasma_current_source(solution, formal_radius=0.06)
    integral = qsc.regularized_axis_integral(solution)
    first = qsc.matched_plasma_field_kernel(
        solution,
        source,
        integral,
        reference_length=2.3,
    )
    second = qsc.matched_plasma_field_kernel(
        solution,
        source,
        integral,
        reference_length=7.1,
    )
    plasma = qsc.plasma_field_on_axis(solution, formal_radius=0.06)

    np.testing.assert_allclose(first, second, rtol=0, atol=2.0e-15)
    assert plasma.maximum_matching_scale_error < 2.0e-15
    np.testing.assert_allclose(plasma.matched_axis_and_core, first, atol=2.0e-15)


def test_shape_correction_and_full_field_converge_in_angular_resolution():
    solution = finite_current_solution(nphi=15)
    coarse = qsc.plasma_field_on_axis(
        solution,
        formal_radius=0.05,
        angular_resolution=64,
    )
    fine = qsc.plasma_field_on_axis(
        solution,
        formal_radius=0.05,
        angular_resolution=128,
    )

    np.testing.assert_allclose(
        coarse.second_order_shape_correction,
        fine.second_order_shape_correction,
        rtol=2.0e-9,
        atol=5.0e-15,
    )
    np.testing.assert_allclose(coarse.field, fine.field, rtol=2.0e-9, atol=5.0e-15)
    assert fine.formal_radius_to_curvature_radius > 0
    assert fine.estimated_field_remainder > 0
    assert fine.angular_resolution == 128


def test_vacuum_and_pressure_only_limits():
    vacuum = qsc.Qsc(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
        etabar=0.64,
        I2=0.0,
        p2=0.0,
        B2c=-0.00322,
        nphi=31,
        order="r2",
    )
    vacuum_field = qsc.plasma_field_on_axis(vacuum, formal_radius=0.05)
    np.testing.assert_allclose(vacuum_field.field, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        vacuum_field.second_order_shape_correction,
        0.0,
        atol=1.0e-14,
    )

    pressure_only = finite_current_solution(I2=0.0)
    pressure_field = qsc.plasma_field_on_axis(
        pressure_only,
        formal_radius=0.05,
    )
    assert np.max(np.abs(pressure_field.field)) > 0
    np.testing.assert_allclose(
        pressure_field.field,
        pressure_field.second_order_shape_correction,
    )


def test_field_is_jittable_and_radius_derivative_matches_finite_difference():
    solution = finite_current_solution(nphi=15)

    def field_component(radius):
        return qsc.plasma_field_on_axis(
            solution,
            formal_radius=radius,
            angular_resolution=16,
        ).field[0, 2]

    radius = 0.05
    value = field_component(radius)
    jitted = jax.jit(field_component)(radius)
    tangent = jax.jvp(field_component, (radius,), (1.0,))[1]
    step = 1.0e-5
    finite_difference = (field_component(radius + step) - field_component(radius - step)) / (
        2 * step
    )

    np.testing.assert_allclose(jitted, value, rtol=2.0e-13)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-7)


def test_regularized_integral_resolution_convergence_for_nonplanar_axis():
    coarse = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=31,
        order="r1",
    )
    fine = qsc.Qsc(
        rc=[1.0, 0.045],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=61,
        order="r1",
    )
    coarse_integral = qsc.regularized_axis_integral(coarse)
    fine_integral = qsc.regularized_axis_integral(fine)

    np.testing.assert_allclose(
        coarse_integral[0],
        fine_integral[0],
        rtol=2.0e-3,
        atol=2.0e-5,
    )
    assert np.all(np.isfinite(coarse_integral))


@pytest.mark.physics
def test_documented_plasma_dominant_case_exceeds_thirty_percent():
    solution = qsc.solve_configuration("plasma_dominant_channel", nphi=61)
    formal_radius = 0.2
    plasma = qsc.plasma_field_on_axis(
        solution,
        formal_radius=formal_radius,
        angular_resolution=64,
    )
    plasma_norm = np.linalg.norm(np.asarray(plasma.field), axis=-1)
    total_norm = np.linalg.norm(np.asarray(solution.B_axis), axis=-1)
    fraction = plasma_norm / total_norm

    assert np.min(fraction) > 0.30
    np.testing.assert_allclose(np.mean(fraction), 0.3325737905856357, rtol=2.0e-12)
    assert formal_radius < float(solution.r_singularity)
    np.testing.assert_allclose(
        plasma.current_source.enclosed_toroidal_current,
        840000.0,
        rtol=2.0e-13,
    )


def test_plasma_field_guards():
    solution = circular_solution()
    with pytest.raises(ValueError, match="angular_resolution"):
        qsc.plasma_field_on_axis(
            solution,
            formal_radius=0.05,
            angular_resolution=7,
        )
    source = qsc.plasma_current_source(solution, formal_radius=0.05)
    with pytest.raises(ValueError, match="reference_length"):
        qsc.matched_plasma_field_kernel(
            solution,
            source,
            qsc.regularized_axis_integral(solution),
            reference_length=0.0,
        )

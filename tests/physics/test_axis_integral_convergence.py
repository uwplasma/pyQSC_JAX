"""Convergence of the Boozer-angle map and the global finite-part axis integral.

The finite-part integral ``regularized_axis_integral`` depends only on the
axis curve: it reads positions, tangents, curvature, binormals and the
arclength angle, never ``sigma``, ``X1c`` or the current weights. Its
convergence under axis-grid refinement is therefore measured here on the bare
geometry, against a reference computed directly on the analytic curve with
composite Gauss--Legendre quadrature in the cylindrical angle, a
fine-spectrum arclength map, and a cancellation-free chord. The reference
shares no code with the grid, the stored angle or the equilibrium solve.

Measured on rc=[1, 0.09], zs=[0, -0.09], nfp=2 (max error over the grid):

=====  ============  ===================  ==========================
nphi   trapezoid     integral, trapezoid  integral, spectral angle
       angle error   angle, no kink term  and kink correction
=====  ============  ===================  ==========================
21     3.25e-4       1.43e-3              2.21e-5
41     8.53e-5       4.99e-4              1.70e-6
91     1.73e-5       1.31e-4              7.21e-8
151    6.29e-6       5.45e-5              9.56e-9
=====  ============  ===================  ==========================

The trapezoidal angle is second order and, inside the subtracted singular
model, alone sets an ``O(h**2)`` floor on the integral; the ``|s|`` kink of
the bounded integrand sets a second, independent ``O(h**2)`` floor. Removing
only one of them leaves the integral second order.
"""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax import Axis
from pyqsc_jax import plasma as plasma_module
from pyqsc_jax.geometry import compute_axis_geometry

RC = (1.0, 0.09)
ZS = (0.0, -0.09)
NFP = 2


class AnalyticAxis:
    """Stellarator-symmetric axis ``R = sum rc cos(n nfp phi)``, ``Z = sum zs sin``."""

    def __init__(self, rc, zs, nfp, *, spectrum=1024):
        self.rc = np.asarray(rc, dtype=float)
        self.zs = np.asarray(zs, dtype=float)
        self.mode = np.arange(len(rc)) * nfp
        phi = 2 * np.pi * np.arange(spectrum) / spectrum
        speed = np.linalg.norm(self.derivatives(phi)[1], axis=-1)
        self.speed_coefficients = np.fft.rfft(speed) / spectrum
        self.spectrum = spectrum
        self.length = 2 * np.pi * self.speed_coefficients[0].real

    def derivatives(self, phi):
        phi = np.asarray(phi, dtype=float)
        angle = self.mode[:, None] * phi[None]
        m = self.mode[:, None]
        rc = self.rc[:, None]
        zs = self.zs[:, None]
        R = np.sum(rc * np.cos(angle), 0)
        dR = np.sum(-m * rc * np.sin(angle), 0)
        d2R = np.sum(-(m**2) * rc * np.cos(angle), 0)
        Z = np.sum(zs * np.sin(angle), 0)
        dZ = np.sum(m * zs * np.cos(angle), 0)
        d2Z = np.sum(-(m**2) * zs * np.sin(angle), 0)
        cosine, sine = np.cos(phi), np.sin(phi)
        position = np.stack((R * cosine, R * sine, Z), -1)
        first = np.stack((dR * cosine - R * sine, dR * sine + R * cosine, dZ), -1)
        second = np.stack(
            (
                d2R * cosine - 2 * dR * sine - R * cosine,
                d2R * sine + 2 * dR * cosine - R * sine,
                d2Z,
            ),
            -1,
        )
        return position, first, second

    def chord(self, phi0, phi):
        """``r(phi0) - r(phi)`` from trigonometric difference identities."""

        half_sum = 0.5 * (phi0 + phi)
        half_difference = 0.5 * (phi0 - phi)
        m = self.mode[:, None]
        delta_R = np.sum(
            -2 * self.rc[:, None] * np.sin(m * half_sum) * np.sin(m * half_difference), 0
        )
        delta_Z = np.sum(
            2 * self.zs[:, None] * np.cos(m * half_sum) * np.sin(m * half_difference), 0
        )
        R = np.sum(self.rc[:, None] * np.cos(m * phi[None]), 0)
        delta_cos = -2 * np.sin(half_sum) * np.sin(half_difference)
        delta_sin = 2 * np.cos(half_sum) * np.sin(half_difference)
        return np.stack(
            (
                delta_R * np.cos(phi0) + R * delta_cos,
                delta_R * np.sin(phi0) + R * delta_sin,
                delta_Z,
            ),
            -1,
        )

    def arclength_angle(self, phi):
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        coefficients = self.speed_coefficients
        k = np.arange(1, len(coefficients))
        weight = np.where(k == self.spectrum // 2, 1.0, 2.0)
        oscillation = np.real(
            coefficients[1:, None] * (np.exp(1j * k[:, None] * phi[None]) - 1) / (1j * k[:, None])
        )
        arclength = coefficients[0].real * phi + np.sum(weight[:, None] * oscillation, 0)
        return 2 * np.pi * arclength / self.length

    def frenet(self, phi0):
        _, first, second = self.derivatives(np.array([phi0]))
        cross = np.cross(first[0], second[0])
        curvature = np.linalg.norm(cross) / np.linalg.norm(first[0]) ** 3
        return curvature, cross / np.linalg.norm(cross)

    def reference_integral(self, phi0, *, panels=4, order=160):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        curvature, binormal = self.frenet(phi0)
        angle0 = self.arclength_angle(phi0)[0]
        edges = phi0 + 2 * np.pi * np.linspace(0.0, 1.0, panels + 1)
        total = np.zeros(3)
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            phi = 0.5 * (start + stop) + 0.5 * (stop - start) * nodes
            _, first, _ = self.derivatives(phi)
            speed = np.linalg.norm(first, axis=-1)
            chord = self.chord(phi0, phi)
            filament = np.cross(first, chord) / np.sum(chord**2, -1)[:, None] ** 1.5
            sine = np.abs(np.sin(0.5 * (self.arclength_angle(phi) - angle0)))
            model = (
                curvature
                * binormal
                * (2 * np.pi / self.length)
                * speed[:, None]
                / (4 * sine[:, None])
            )
            total += 0.5 * (stop - start) * np.sum(weights[:, None] * (filament - model), 0)
        return total


@cache
def axis_only_solution(rc, zs, nfp, nphi):
    """Stand-in exposing only what the global integral reads: axis geometry."""

    axis = Axis(rc=list(rc), zs=list(zs), nfp=nfp)
    return SimpleNamespace(
        geometry=compute_axis_geometry(axis, nphi=nphi),
        inputs=SimpleNamespace(axis=axis, nphi=nphi),
    )


REFERENCE_AXIS = AnalyticAxis(RC, ZS, NFP)
RESOLUTIONS = np.array([21, 41, 81])


@pytest.fixture(scope="module")
def reference_axis():
    return REFERENCE_AXIS


@cache
def reference_on_grid(nphi):
    phi = np.asarray(axis_only_solution(RC, ZS, NFP, nphi).geometry.samples.phi)
    return np.array([REFERENCE_AXIS.reference_integral(value) for value in phi])


def integral_error(nphi):
    solution = axis_only_solution(RC, ZS, NFP, nphi)
    integral = np.asarray(qsc.regularized_axis_integral(solution))
    return np.max(np.abs(integral - reference_on_grid(nphi)))


def observed_order(resolutions, errors):
    return -np.polyfit(np.log(resolutions), np.log(errors), 1)[0]


def test_reference_integral_is_self_converged(reference_axis):
    coarse = reference_axis.reference_integral(0.3, panels=4, order=160)
    fine = reference_axis.reference_integral(0.3, panels=16, order=200)

    np.testing.assert_allclose(coarse, fine, rtol=0, atol=1.0e-10)


def test_trapezoid_angle_is_second_order_and_spectral_angle_is_exact(reference_axis):
    trapezoid_errors = []
    for nphi in RESOLUTIONS:
        solution = axis_only_solution(RC, ZS, NFP, int(nphi))
        geometry = solution.geometry
        exact = reference_axis.arclength_angle(np.asarray(geometry.samples.phi))
        spectral = np.asarray(plasma_module.arclength_boozer_angle(solution))
        trapezoid_errors.append(np.max(np.abs(np.asarray(geometry.varphi) - exact)))

        np.testing.assert_allclose(spectral, exact, rtol=0, atol=5.0e-15)
        assert abs(spectral[0]) < 1.0e-15
        # Unit-mean derivative: the full-period increment is exactly 2*pi/nfp.
        np.testing.assert_allclose(np.mean(geometry.d_varphi_d_phi), 1.0, rtol=0, atol=2.0e-15)
        full_varphi = np.asarray(plasma_module._full_torus_axis_samples(solution)[0])
        np.testing.assert_allclose(
            full_varphi[nphi] - full_varphi[0], 2 * np.pi / NFP, rtol=0, atol=5.0e-15
        )
        assert np.all(np.diff(full_varphi) > 0)

    # The stored pyQSC angle keeps its trapezoidal O(h**2) definition.
    assert trapezoid_errors[1] > 5.0e-5
    np.testing.assert_allclose(
        observed_order(RESOLUTIONS, trapezoid_errors), 2.0, rtol=0, atol=0.05
    )


def test_paired_integrand_has_opposite_one_sided_limits_and_zero_coincident_value(reference_axis):
    phi0 = 0.3
    axis_scale = reference_axis.length / (2 * np.pi)
    # Frenet data from a resolved grid; phi0 is moved onto its nearest node.
    geometry = axis_only_solution(RC, ZS, NFP, 81).geometry
    phi_grid = np.asarray(geometry.samples.phi)
    node = int(np.argmin(np.abs(phi_grid - phi0)))
    phi0 = float(phi_grid[node])
    position0, _, _ = reference_axis.derivatives(np.array([phi0]))
    curvature = float(geometry.curvature[node])
    binormal = np.asarray(geometry.binormal_cartesian[node])
    normal = np.asarray(geometry.normal_cartesian[node])
    torsion = float(geometry.torsion[node])
    d_curvature_d_s = float((geometry.d_d_varphi @ geometry.curvature)[node]) / axis_scale
    jump = axis_scale * (d_curvature_d_s * binormal - curvature * torsion * normal) / 3

    def integrand(epsilon):
        phi = np.array([phi0 + epsilon])
        position, first, _ = reference_axis.derivatives(phi)
        tangent = first / np.linalg.norm(first, axis=-1)[:, None]
        angle = reference_axis.arclength_angle(phi) - reference_axis.arclength_angle(phi0)
        return np.asarray(
            plasma_module.paired_axis_integrand(
                jnp.asarray(position0),
                jnp.asarray([curvature]),
                jnp.asarray(binormal[None]),
                jnp.asarray(angle[None]),
                jnp.asarray(position[None]),
                jnp.asarray(tangent[None]),
                axis_scale,
            )
        )[0, 0]

    assert np.all(integrand(0.0) == 0.0)
    assert np.linalg.norm(jump) > 1.0e-2
    for epsilon in (1.0e-2, 3.0e-3, 1.0e-3):
        plus = integrand(epsilon)
        minus = integrand(-epsilon)
        # One-sided limits are +-jump with an O(|s|) approach.
        assert np.max(np.abs(plus - jump)) < 0.1 * epsilon
        assert np.max(np.abs(minus + jump)) < 0.1 * epsilon
        # The coincident value 0 is the limit of the symmetric mean.
        assert np.max(np.abs(0.5 * (plus + minus))) < 0.1 * epsilon


@pytest.mark.parametrize("nfp, nphi", [(2, 21), (3, 7)])
def test_field_period_replication_matches_single_period_representation(nfp, nphi):
    rc = (1.0,) + (0.0,) * (nfp - 1) + (0.09,)
    zs = (0.0,) + (0.0,) * (nfp - 1) + (-0.09,)
    periodic = axis_only_solution(RC, ZS, nfp, nphi)
    whole = axis_only_solution(rc, zs, 1, nfp * nphi)

    np.testing.assert_allclose(
        whole.geometry.samples.R[:nphi], periodic.geometry.samples.R, rtol=0, atol=1.0e-14
    )
    periodic_integral = np.asarray(qsc.regularized_axis_integral(periodic))
    whole_integral = np.asarray(qsc.regularized_axis_integral(whole))
    for period in range(nfp):
        rotation = 2 * np.pi * period / nfp
        cosine, sine = np.cos(rotation), np.sin(rotation)
        rotated = np.stack(
            (
                cosine * periodic_integral[:, 0] - sine * periodic_integral[:, 1],
                sine * periodic_integral[:, 0] + cosine * periodic_integral[:, 1],
                periodic_integral[:, 2],
            ),
            -1,
        )
        np.testing.assert_allclose(
            whole_integral[period * nphi : (period + 1) * nphi], rotated, rtol=0, atol=2.0e-13
        )


@pytest.mark.parametrize("nphi", [21, 42])
def test_circular_axis_integral_vanishes_for_even_and_odd_grids(nphi):
    solution = axis_only_solution((2.3,), (0.0,), 1, nphi)

    assert np.max(np.abs(np.asarray(qsc.regularized_axis_integral(solution)))) < 1.0e-13


def test_nonplanar_integral_converges_at_fourth_order_to_independent_reference():
    errors = np.array([integral_error(int(nphi)) for nphi in RESOLUTIONS])

    # Measured: 2.2e-5, 1.7e-6, 1.15e-7 (order 3.9).
    assert errors[1] < 3.0e-6
    assert observed_order(RESOLUTIONS, errors) > 3.6


def test_trapezoid_angle_alone_would_limit_the_integral_to_second_order(monkeypatch):
    """With the kink correction kept, the stored trapezoidal angle is the floor.

    Its local ``O(h**3)`` step error, divided by the ``|angle|`` of the singular
    model and accumulated over the torus, gives ``O(h**2 log h)``.
    """

    def trapezoid_angle(solution):
        return solution.geometry.varphi

    monkeypatch.setattr(plasma_module, "arclength_boozer_angle", trapezoid_angle)
    errors = np.array([integral_error(int(nphi)) for nphi in RESOLUTIONS])

    # Measured: 1.6e-3, 5.5e-4, 1.7e-4 (order 1.66).
    assert errors[1] > 1.0e-4
    assert 1.4 < observed_order(RESOLUTIONS, errors) < 2.1


@pytest.mark.slow
def test_global_integral_error_is_separate_from_equilibrium_convergence(reference_axis):
    """Hold the global integral at its reference to expose equilibrium convergence."""

    fields = {}
    for nphi in (31, 61):
        solution = qsc.Qsc(
            rc=list(RC),
            zs=list(ZS),
            nfp=NFP,
            etabar=0.95,
            B2c=-0.7,
            p2=-6.0e5,
            I2=0.6,
            nphi=nphi,
            order="r2",
        )
        plasma = qsc.plasma_field_on_axis(solution, formal_radius=0.05)
        source = plasma.current_source
        prefactor = float(source.parallel_current_mu0 * source.formal_radius**2 / 4)
        integral = np.asarray(plasma.regularized_axis_integral)[0]
        reference = reference_axis.reference_integral(0.0)
        field = np.asarray(plasma.field)[0]
        fields[nphi] = (field, field + prefactor * (reference - integral), prefactor)

    field_31, held_31, prefactor = fields[31]
    _, held_61, _ = fields[61]
    integral_part = np.max(np.abs(field_31 - held_31))
    equilibrium_part = np.max(np.abs(held_31 - held_61))
    # Measured at phi=0, nphi=31: 8e-10 from the integral, 3e-7 from the equilibrium.
    assert integral_part < 5.0e-9
    assert equilibrium_part > 20 * integral_part
    assert np.max(np.abs(held_61 - fields[61][0])) < 5.0e-10
    assert abs(prefactor) > 0

"""Fixed-Cartesian finite differences of the volume Biot-Savart field about the axis.

The reference is the unexpanded volume law over the truncated quadratic near-axis
source of ``test_plasma_volume_integral`` (its current is rebuilt from the equilibrium
coefficients, not taken from ``pyqsc_jax.plasma``). Here the source tube ``r <= a`` is
held fixed and the field is evaluated at laboratory points ``x0 + h e_j`` and
``x0 +- h e_j +- h e_k``. Central differences in these fixed Cartesian directions give
all nine gradient entries and all eighteen independent Hessian entries
``H[i, j, k]`` (``j <= k``); nothing computed from the predicted Hessian enters the
extraction.

Every observation point is located in the source coordinates ``(phi, q1, q2)`` by an
independent Newton inversion of the quadratic map, whose residual is checked. The
labels only place the quadrature nodes around the observation. The integrable
coincidence is handled by two unrelated schemes:

* ``"polar"``: polar labels about the observation times log-graded toroidal nodes;
* ``"spherical"``: a contact-preserving spherical ball about the observation (the
  ``rho**2`` Jacobian cancels the kernel), plus a smooth outer region.

Their agreement shows that the extracted derivatives do not depend on the treatment
of the coincidence.

The predicted gradient and Hessian are asymptotic in ``a``, so each comparison is a
convergence test in the radius. Steps are proportional to the radius (``h = a/4``,
``a/8`` and, in the full sweep, ``a/16``) and Richardson-combined, which keeps every
observation within ``0.36 a`` of the axis.

The fast default subset is one finite-current case at two radii. Set
``PYQSC_RUN_VOLUME_SWEEP=1`` for every case, three radii, three steps and both
coincidence schemes.
"""

from __future__ import annotations

import os
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_plasma_volume_integral import (
    AXISYMMETRIC,
    QA,
    _interpolate,
    _source_tables,
    _to_cartesian,
)

import pyqsc_jax as qsc

RUN_SWEEP = os.environ.get("PYQSC_RUN_VOLUME_SWEEP") == "1"
full_sweep = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUN_SWEEP, reason="Set PYQSC_RUN_VOLUME_SWEEP=1 for the full sweep."),
]

SECTION = 7  # phi = 0.36 rad on the nfp=2 axis: generic, away from symmetry planes.
QUADRATURE = {
    "polar": {"coarse": (150, 16, 32), "fine": (250, 24, 48)},
    "spherical": {"coarse": (24, 12, 16, 60, 12, 24), "fine": (36, 16, 24, 90, 16, 36)},
}


class _Source:
    """The fixed volume source of one solution and one radius, callable at points."""

    def __init__(self, solution, radius):
        self.tables, self.ell = _source_tables(solution)
        self.period = 2 * np.pi / int(solution.inputs.axis.nfp)
        self.radius = radius
        self.with_quadratic_current = "w_11" in self.tables

    def at(self, phi):
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        values = {
            key: _interpolate(table, self.period, np.mod(phi, self.period))
            for key, table in self.tables.items()
        }
        for key in values:
            if key != "d_varphi":
                values[key] = _to_cartesian(values[key], phi)
        return values

    @staticmethod
    def position(values, q1, q2):
        return (
            values["axis"]
            + q1 * values["map_1"]
            + q2 * values["map_2"]
            + (q1 * q1 + q2 * q2) * values["map_20"]
            + (q1 * q1 - q2 * q2) * values["map_2c"]
            + 2 * q1 * q2 * values["map_2s"]
        )

    def current(self, values, q1, q2):
        current = values["w_0"] + q1 * values["w_1"] + q2 * values["w_2"]
        if self.with_quadratic_current:
            current = current + q1 * q1 * values["w_11"] + q1 * q2 * values["w_12"]
            current = current + q2 * q2 * values["w_22"]
        return current

    def locate(self, point, phi_guess):
        """Newton inversion of the quadratic map; returns ``(phi, q1, q2)`` and residual."""

        unknown = np.array([phi_guess, 0.0, 0.0])
        delta = 1.0e-6
        for _ in range(30):
            phi, q1, q2 = unknown
            values = self.at(np.array([phi - delta, phi, phi + delta]))
            positions = self.position(values, q1, q2)
            residual = positions[1] - point
            if np.linalg.norm(residual) < 1.0e-15:
                break
            jacobian = np.stack(
                (
                    (positions[2] - positions[0]) / (2 * delta),
                    values["map_1"][1]
                    + 2 * q1 * (values["map_20"][1] + values["map_2c"][1])
                    + 2 * q2 * values["map_2s"][1],
                    values["map_2"][1]
                    + 2 * q2 * (values["map_20"][1] - values["map_2c"][1])
                    + 2 * q1 * values["map_2s"][1],
                ),
                axis=1,
            )
            unknown = unknown - np.linalg.solve(jacobian, residual)
        phi, q1, q2 = unknown
        residual = self.position(self.at(phi), q1, q2)[0] - point
        return unknown, float(np.linalg.norm(residual))

    def _sum(self, point, phi, q1, q2, measure):
        """``sum(measure * W x (x - x') / |x - x'|**3) / 4 pi`` over source nodes.

        ``phi`` has shape ``(n,)``; ``q1``, ``q2`` and ``measure`` have shape ``(n, m)``.
        """

        values = {key: value[:, None, :] for key, value in self.at(phi).items()}
        separation = point - self.position(values, q1[..., None], q2[..., None])
        kernel = np.cross(self.ell * self.current(values, q1[..., None], q2[..., None]), separation)
        kernel /= np.linalg.norm(separation, axis=-1, keepdims=True) ** 3
        weight = measure * values["d_varphi"][..., 0]
        return np.einsum("nm,nmi->i", weight, kernel) / (4 * np.pi)

    def _disk_reach(self, q1, q2, cosine, sine):
        projection = q1 * cosine + q2 * sine
        return -projection + np.sqrt(projection**2 + self.radius**2 - q1**2 - q2**2)

    def field_polar(self, point, labels, nodes):
        toroidal, radial, angular = nodes
        phi0, q10, q20 = labels
        unit, unit_weight = np.polynomial.legendre.leggauss(toroidal)
        unit, unit_weight = (unit + 1) / 2, unit_weight / 2
        closest = 1.0e-8
        stretch = np.arcsinh(np.pi / closest)
        offset = closest * np.sinh(stretch * unit)
        weight = closest * stretch * np.cosh(stretch * unit) * unit_weight
        offset, weight = np.r_[-offset[::-1], offset], np.r_[weight[::-1], weight]
        angle = np.arange(angular) * 2 * np.pi / angular
        cosine, sine = np.cos(angle), np.sin(angle)
        reach = self._disk_reach(q10, q20, cosine, sine)
        node, node_weight = np.polynomial.legendre.leggauss(radial)
        node, node_weight = (node + 1) / 2, node_weight / 2
        distance = node[:, None] * reach[None, :]
        area = node_weight[:, None] * reach * distance * 2 * np.pi / angular
        q1 = np.broadcast_to((q10 + distance * cosine).ravel(), (offset.size, distance.size))
        q2 = np.broadcast_to((q20 + distance * sine).ravel(), (offset.size, distance.size))
        measure = np.outer(weight, area.ravel())
        return self._sum(point, phi0 + offset, q1, q2, measure)

    def field_spherical(self, point, labels, nodes):
        azimuthal, polar, radial, outer_toroidal, outer_radial, outer_angular = nodes
        phi0, q10, q20 = labels
        speed = np.linalg.norm(
            np.diff(self.position(self.at(phi0 + np.array([-1e-6, 1e-6])), q10, q20), axis=0)
        ) / (2e-6)
        half_width = 2 * self.radius  # Slab |xi_0| <= half_width, xi_0 = speed*(phi - phi0).
        # Contact-preserving ball about the observation in xi = (xi_0, q1 - q10, q2 - q20).
        psi = np.arange(azimuthal) * 2 * np.pi / azimuthal
        cosine, sine = np.cos(psi), np.sin(psi)
        disk = self._disk_reach(q10, q20, cosine, sine)
        ratio = half_width / disk
        crossing = ratio / np.sqrt(1 + ratio**2)
        unit, unit_weight = np.polynomial.legendre.leggauss(polar)
        rho_unit, rho_weight = np.polynomial.legendre.leggauss(radial)
        rho_unit, rho_weight = (rho_unit + 1) / 2, rho_weight / 2
        phis, q1s, q2s, measures = [], [], [], []
        # Polar angle about the xi_0 axis, split where the ray leaves through the slab
        # face instead of the tube wall; every piece is analytic in the angle.
        corner = np.arccos(crossing)
        zero = np.zeros_like(corner)
        for low, high in ((zero, corner), (corner, np.pi - corner), (np.pi - corner, zero + np.pi)):
            theta = (low + high)[:, None] / 2 + (high - low)[:, None] / 2 * unit[None, :]
            mu, transverse = np.cos(theta), np.sin(theta)
            mu_weight = (high - low)[:, None] / 2 * unit_weight[None, :] * transverse
            reach = np.minimum(
                disk[:, None] / np.maximum(transverse, 1e-300),
                half_width / np.maximum(np.abs(mu), 1e-300),
            )
            rho = reach[..., None] * rho_unit
            measure = mu_weight[..., None] * reach[..., None] * rho_weight * rho**2
            phis.append((phi0 + mu[..., None] * rho / speed).ravel())
            q1s.append((q10 + transverse[..., None] * cosine[:, None, None] * rho).ravel())
            q2s.append((q20 + transverse[..., None] * sine[:, None, None] * rho).ravel())
            measures.append(measure.ravel() * (2 * np.pi / azimuthal) / speed)
        field = self._sum(
            point,
            np.concatenate(phis),
            np.concatenate(q1s)[:, None],
            np.concatenate(q2s)[:, None],
            np.concatenate(measures)[:, None],
        )
        # Smooth outer region |phi - phi0| > half_width / speed over the whole torus.
        start = half_width / speed
        unit, unit_weight = np.polynomial.legendre.leggauss(outer_toroidal)
        unit, unit_weight = (unit + 1) / 2, unit_weight / 2
        stretch = np.log(1 + (np.pi - start) / start)
        offset = start + start * np.expm1(stretch * unit)
        weight = start * stretch * np.exp(stretch * unit) * unit_weight
        offset, weight = np.r_[-offset[::-1], offset], np.r_[weight[::-1], weight]
        angle = np.arange(outer_angular) * 2 * np.pi / outer_angular
        node, node_weight = np.polynomial.legendre.leggauss(outer_radial)
        node, node_weight = (node + 1) / 2 * self.radius, node_weight / 2 * self.radius
        shape = (offset.size, outer_radial * outer_angular)
        q1 = np.broadcast_to(np.outer(node, np.cos(angle)).ravel(), shape)
        q2 = np.broadcast_to(np.outer(node, np.sin(angle)).ravel(), shape)
        area = np.outer(node_weight * node, np.full(outer_angular, 2 * np.pi / outer_angular))
        measure = np.outer(weight, area.ravel())
        return field + self._sum(point, phi0 + offset, q1, q2, measure)

    def field(self, point, phi_guess, scheme, nodes):
        labels, residual = self.locate(point, phi_guess)
        method = self.field_polar if scheme == "polar" else self.field_spherical
        return method(point, labels, nodes), residual


def cartesian_derivatives(source, axis_point, phi0, step, scheme, nodes):
    """Central-difference gradient and Hessian in fixed laboratory directions."""

    residuals = []

    def field(offset):
        value, residual = source.field(axis_point + offset, phi0, scheme, nodes)
        residuals.append(residual)
        return value

    basis = np.eye(3) * step
    centre = field(np.zeros(3))
    plus = [field(basis[j]) for j in range(3)]
    minus = [field(-basis[j]) for j in range(3)]
    gradient = np.stack([(plus[j] - minus[j]) / (2 * step) for j in range(3)], axis=1)
    hessian = np.zeros((3, 3, 3))
    for j in range(3):
        hessian[:, j, j] = (plus[j] - 2 * centre + minus[j]) / step**2
        for k in range(j + 1, 3):
            mixed = (
                field(basis[j] + basis[k])
                - field(basis[j] - basis[k])
                - field(-basis[j] + basis[k])
                + field(-basis[j] - basis[k])
            ) / (4 * step**2)
            hessian[:, j, k] = hessian[:, k, j] = mixed
    return gradient, hessian, max(residuals)


CASES = {
    # key: (parameters, section, radii). Orientation choices, each a relabelling of the
    # same physical equilibrium, so any sign slip in either code path shows up:
    # ``sG=-1`` reverses the Boozer Jacobian sign; ``spsi=-1`` with ``I2 -> -I2``
    # reverses the toroidal flux and the current with it; the mirror ``zs -> -zs``,
    # ``etabar -> -etabar``, ``I2 -> -I2`` reverses the torsion (handedness) of the axis.
    "qa-pressure": ({**QA, "I2": 0.0, "p2": -6.0e5}, SECTION, (0.04, 0.02, 0.01)),
    "qa-current": ({**QA, "I2": 0.6, "p2": -6.0e5}, SECTION, (0.04, 0.02, 0.01)),
    "qa-current-sG": ({**QA, "I2": 0.6, "p2": -6.0e5, "sG": -1}, SECTION, (0.04, 0.02, 0.01)),
    "qa-current-spsi": ({**QA, "I2": -0.6, "p2": -6.0e5, "spsi": -1}, SECTION, (0.04, 0.02, 0.01)),
    "qa-current-mirror": (
        {**QA, "zs": -QA["zs"], "etabar": -QA["etabar"], "I2": -0.6, "p2": -6.0e5},
        SECTION,
        (0.04, 0.02, 0.01),
    ),
    # ``spsi=-1`` alone at fixed ``I2`` is a different equilibrium (iotaN = 0.265) whose
    # quadratic map folds at r_singularity = 0.0143, so it needs smaller radii.
    "qa-current-spsi-only": (
        {**QA, "I2": 0.6, "p2": -6.0e5, "spsi": -1},
        SECTION,
        (0.005, 0.0025, 0.00125),
    ),
    "circular": ({**AXISYMMETRIC, "nfp": 4, "I2": 0.9, "p2": -2.0e5}, 3, (0.04, 0.02, 0.01)),
}
# Relative Frobenius error bounds (gradient, Hessian) at the smallest radius: about twice
# the measured values, which all converge as a**2 (see ``_check_case``).
BOUNDS = {
    "qa-pressure": (1.0e-2, 1.4e-2),
    "qa-current": (1.0e-3, 1.0e-3),
    "qa-current-sG": (1.0e-3, 1.0e-3),
    "qa-current-spsi": (1.0e-3, 1.0e-3),
    "qa-current-mirror": (1.0e-3, 1.0e-3),
    "qa-current-spsi-only": (3.0e-3, 1.4e-2),
    "circular": (4.0e-4, 8.0e-4),
}
STEPS = (4, 8, 16)  # h = a/4, a/8, a/16.


@cache
def _solution(key):
    solution = qsc.near_axis(**CASES[key][0]).solution
    assert bool(solution.root_report.converged & solution.second_order.linear_report.converged)
    return solution


@cache
def measure(key, radius, scheme="polar", level="fine", steps=STEPS):
    """Numerical derivatives at each step, Richardson pairs, and the prediction."""

    solution = _solution(key)
    section = CASES[key][1]
    source = _Source(solution, radius)
    axis_point = np.asarray(solution.geometry.position_cartesian)[section]
    phi0 = float(np.asarray(solution.phi)[section])
    nodes = QUADRATURE[scheme][level]
    raw = [
        cartesian_derivatives(source, axis_point, phi0, radius / step, scheme, nodes)
        for step in steps
    ]
    # Central differences err by h**2 * (fourth-order term): Richardson removes it.
    richardson = [
        tuple((4 * fine[i] - coarse[i]) / 3 for i in range(2))
        for coarse, fine in zip(raw[:-1], raw[1:], strict=True)
    ]
    data = qsc.plasma_hessian_on_axis(solution, formal_radius=radius)
    return {
        "raw": [(gradient, hessian) for gradient, hessian, _ in raw],
        "richardson": richardson,
        "gradient": richardson[0][0],
        "hessian": richardson[0][1],
        "inverse_residual": max(residual for *_, residual in raw),
        "predicted_gradient": np.asarray(data.field.gradient)[section],
        "predicted_hessian": np.asarray(data.hessian)[section],
    }


def relative(numerical, predicted):
    return float(np.linalg.norm(numerical - predicted) / np.linalg.norm(predicted))


def _errors(result):
    gradient = relative(result["gradient"], result["predicted_gradient"])
    hessian = relative(result["hessian"], result["predicted_hessian"])
    return gradient, hessian


def _check_radius(result, error, *, zero_current):
    """Checks at one radius that do not depend on the convergence rate."""

    gradient, predicted_gradient = result["gradient"], result["predicted_gradient"]
    hessian, predicted_hessian = result["hessian"], result["predicted_hessian"]
    scale_g, scale_h = np.linalg.norm(predicted_gradient), np.linalg.norm(predicted_hessian)
    # The observation labels come from an independent Newton inversion of the map.
    assert result["inverse_residual"] < 1.0e-13
    # Step refinement: halving every step moves the Richardson value by a small
    # fraction of the asymptotic error being measured. With two steps only, the
    # Richardson correction itself must be that small.
    if len(result["richardson"]) > 1:
        (coarse_g, coarse_h), (fine_g, fine_h) = result["richardson"][:2]
    else:
        (coarse_g, coarse_h), (fine_g, fine_h) = result["richardson"][0], result["raw"][1]
    assert np.linalg.norm(coarse_g - fine_g) / scale_g < 0.05 * error[0]
    assert np.linalg.norm(coarse_h - fine_h) / scale_h < 0.05 * error[1]
    # Every one of the 9 gradient and 18 independent Hessian entries, not just the norm.
    assert np.max(np.abs(gradient - predicted_gradient)) / scale_g < error[0]
    assert np.max(np.abs(hessian - predicted_hessian)) / scale_h < error[1]
    # Maxwell identities are NOT imposed on the numerical tensors; they are diagnostics.
    assert np.max(np.abs(np.einsum("iik->k", hessian))) / scale_h < 0.05 * error[1]
    assert abs(np.trace(gradient)) / scale_g < 0.05 * error[0]
    if zero_current:
        # Validate the five independent entries of the symmetric trace-free gradient
        # one by one on the raw tensor before using its symmetry as a diagnostic.
        for i, j in ((0, 0), (1, 1), (0, 1), (0, 2), (1, 2)):
            assert abs(gradient[i, j] - predicted_gradient[i, j]) / scale_g < error[0]
        assert np.linalg.norm(gradient - gradient.T) / scale_g < 0.1 * error[0]
    else:
        # The on-axis current makes G - G^T = mu0 J x (.) nonzero; it must match, so an
        # accidental transpose (error ~ 2) or sign flip cannot pass.
        antisymmetric = (gradient - gradient.T) / 2
        predicted_antisymmetric = (predicted_gradient - predicted_gradient.T) / 2
        assert np.linalg.norm(predicted_antisymmetric) > 0.5 * scale_g
        assert relative(antisymmetric, predicted_antisymmetric) < 0.5 * error[0]
        assert relative(gradient.T, predicted_gradient) > 1.0


def _check_case(key, radii=None, *, steps=STEPS, levels=True):
    """Convergence in ``a`` of both tensors, plus per-radius and quadrature checks."""

    radii = CASES[key][2] if radii is None else radii
    assert all(np.isclose(coarse, 2 * fine) for coarse, fine in zip(radii, radii[1:], strict=False))
    zero_current = CASES[key][0]["I2"] == 0.0
    bound = np.array(BOUNDS[key])
    errors = []
    for radius in radii:
        result = measure(key, radius, steps=steps)
        errors.append(_errors(result))
        # The asymptotic error scales as a**2, so the per-radius bound does too.
        _check_radius(result, bound * (radius / radii[-1]) ** 2, zero_current=zero_current)
    errors = np.array(errors)
    # Both tensors converge as a**2 (observed orders 1.85-2.01): a transposed, sign-
    # flipped or missing term would leave an order-one error that does not shrink.
    assert np.all(errors[-1] < bound), errors
    ratios = errors[:-1] / errors[1:]
    assert np.all((3.0 < ratios) & (ratios < 4.6)), ratios
    if levels:
        # Source-quadrature refinement at the smallest radius.
        fine = measure(key, radii[-1], steps=steps)
        coarse = measure(key, radii[-1], level="coarse", steps=steps)
        for index, name in enumerate(("gradient", "hessian")):
            change = np.linalg.norm(coarse[name] - fine[name])
            assert change / np.linalg.norm(fine["predicted_" + name]) < 1.0e-2 * errors[-1, index]
    return errors


def test_fixed_cartesian_derivatives_of_finite_current_nonplanar_section():
    # Fast default subset: generic nonplanar section, finite current, two radii.
    errors = _check_case("qa-current", radii=(0.02, 0.01), steps=STEPS[:2])
    assert errors[-1, 0] < 1.0e-3 and errors[-1, 1] < 1.0e-3


@pytest.mark.parametrize("key", [pytest.param(key, marks=full_sweep) for key in CASES])
def test_fixed_cartesian_derivatives_full_sweep(key):
    _check_case(key)


@pytest.mark.parametrize(
    "key", [pytest.param(key, marks=full_sweep) for key in ("qa-pressure", "qa-current")]
)
def test_extracted_derivatives_do_not_depend_on_the_coincidence_treatment(key):
    radius = 0.01
    polar = measure(key, radius)
    spherical = measure(key, radius, scheme="spherical", steps=STEPS[:2])
    errors = _errors(polar)
    for index, name in enumerate(("gradient", "hessian")):
        change = np.linalg.norm(spherical[name] - polar[name])
        assert change / np.linalg.norm(polar["predicted_" + name]) < 1.0e-2 * errors[index]


def test_field_gradient_and_hessian_tangents_through_a_circular_section():
    # On the circular axis X1c = etabar and Y1s = 1/etabar: the section is exactly
    # circular at etabar = 1 and nearly circular at 1.001. The existing Hessian check in
    # test_plasma_volume_integral covers only the exact circle and only the Hessian.
    def jet(value):
        solution = qsc.near_axis(**{**AXISYMMETRIC, "etabar": value}, I2=0.4, p2=-2.0e5).solution
        data = qsc.plasma_hessian_on_axis(solution, formal_radius=0.03)
        return data.field.field.field, data.field.gradient, data.hessian

    tangent_of, value_of = jax.jit(jax.jacfwd(jet)), jax.jit(jet)
    step = 1.0e-6
    for etabar in (1.0, 1.001):
        tangents = tangent_of(etabar)
        plus, minus = value_of(etabar + step), value_of(etabar - step)
        for tangent, upper, lower in zip(tangents, plus, minus, strict=True):
            finite_difference = (upper - lower) / (2 * step)
            scale = float(jnp.max(jnp.abs(tangent)))
            assert bool(jnp.all(jnp.isfinite(tangent)))
            assert scale > 0.0
            np.testing.assert_allclose(tangent, finite_difference, rtol=0, atol=1.0e-6 * scale)

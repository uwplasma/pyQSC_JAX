"""Volume Biot-Savart and closed-form checks of the plasma field, gradient and Hessian.

The reference is a brute-force quadrature of the volume law over the tube
``0 <= r <= a``. Its kernel is never expanded, so it uses no inner/outer matching,
and its weighted current is rebuilt here from the equilibrium coefficients rather
than taken from ``pyqsc_jax.plasma``. Agreement therefore tests the asymptotic
inversion itself, which the Maxwell identities alone cannot do: they fix the curl
and divergence of the plasma field but not its harmonic part.

The source is a prescribed conserved current on the truncated quadratic map, not an
exact finite-radius equilibrium, so each comparison is a convergence test in ``a``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.second_order import MU0

QA = {
    "rc": jnp.array([1.0, 0.09]),
    "zs": jnp.array([0.0, -0.09]),
    "etabar": 0.95,
    "nfp": 2,
    "nphi": 61,
    "order": "r2",
    "B0": 1.0,
    "B2c": -0.7,
}
AXISYMMETRIC = {
    "rc": jnp.array([1.0]),
    "zs": jnp.array([0.0]),
    "etabar": 1.0,
    "nfp": 1,
    "nphi": 31,
    "order": "r2",
    "B0": 1.0,
    "B2c": 0.0,
}
SECTION = 7  # Away from a stellarator-symmetry plane.


def _solve(**parameters):
    solution = qsc.near_axis(**parameters).solution
    assert bool(solution.root_report.converged & solution.second_order.linear_report.converged)
    return solution


def _frame(solution, section):
    geometry = solution.geometry
    return np.stack(
        [
            np.asarray(getattr(geometry, name + "_cartesian"))[section]
            for name in ("tangent", "normal", "binormal")
        ]
    )


def _to_cartesian(vector, phi):
    cosine, sine = np.cos(phi)[:, None], np.sin(phi)[:, None]
    return np.concatenate(
        (
            vector[:, :1] * cosine - vector[:, 1:2] * sine,
            vector[:, :1] * sine + vector[:, 1:2] * cosine,
            vector[:, 2:],
        ),
        axis=1,
    )


def _interpolate(samples, period, points):
    """Trigonometric interpolant of uniform one-period samples of a periodic array."""

    count = samples.shape[0]
    coefficients = np.fft.fft(samples, axis=0) / count
    wavenumber = np.fft.fftfreq(count, 1.0 / count)
    phase = np.exp(2j * np.pi * np.outer(points, wavenumber) / period)
    return (phase @ coefficients.reshape(count, -1)).real.reshape(points.shape + samples.shape[1:])


def _source_tables(solution):
    """Cylindrical-component tables of the map and of ``W / (ell * r)`` as a polynomial.

    The linear coefficients hold for any ``I2``. The quadratic coefficients are the
    ``r**3`` weighted current of the ``I2 = I4 = 0`` branch, which the order-``a**2``
    gradient requires; they are omitted at finite current, where only leading-order
    derivatives are tested.
    """

    geometry, second, inputs = solution.geometry, solution.second_order, solution.inputs
    sG, spsi, B0 = inputs.sG, inputs.spsi, float(inputs.B0)
    chi, ell = sG * spsi, float(geometry.abs_G0_over_B0)
    d_ds = np.asarray(geometry.d_d_varphi) / ell
    x, y, y_sigma = (np.asarray(v) for v in (solution.X1c, solution.Y1s, solution.Y1c))
    torsion, curvature = np.asarray(geometry.torsion), np.asarray(geometry.curvature)
    current = 2 * chi * float(inputs.I2)
    G2 = float(second.G2)
    C2 = G2 + int(second.N_helicity) * float(inputs.I2)
    B_bar, beta_s = spsi * B0, float(second.beta_1s)
    tangent, normal, binormal = (
        np.asarray(getattr(geometry, name + "_cylindrical"))
        for name in ("tangent", "normal", "binormal")
    )

    def column(value):
        return (np.asarray(value, dtype=float) * np.ones_like(x))[:, None]

    def vector(t, n, b):
        return column(t) * tangent + column(n) * normal + column(b) * binormal

    zero = 0.0 * x
    tables = {
        "axis": np.stack((np.asarray(solution.R0), zero, np.asarray(solution.Z0)), axis=1),
        "map_1": vector(zero, x, y_sigma),
        "map_2": vector(zero, zero, y),
        "map_20": vector(second.Z20, second.X20, second.Y20),
        "map_2c": vector(second.Z2c, second.X2c, second.Y2c),
        "map_2s": vector(second.Z2s, second.X2s, second.Y2s),
        "w_0": vector(current, zero, zero),
        "w_1": vector(
            -chi * B_bar * beta_s - current * curvature * x,
            current * (d_ds @ x - torsion * y_sigma),
            current * (d_ds @ y_sigma + torsion * x) - 2 * chi * C2 * y / ell,
        ),
        "w_2": vector(
            zero,
            -current * torsion * y + 2 * chi * C2 * x / ell,
            current * (d_ds @ y) + 2 * chi * C2 * y_sigma / ell,
        ),
        "d_varphi": np.asarray(geometry.d_varphi_d_phi)[:, None],
    }
    if float(inputs.I2) == 0.0:
        beta_2s = (
            MU0
            * float(inputs.p2)
            * sG
            * B0
            * ell
            / (B_bar * B0**2 * float(solution.iotaN))
            * (1.5 * float(inputs.etabar) ** 2 - 2 * float(inputs.B2c) / B0)
        )
        shape = 4 * chi * G2 / ell
        X2s, X2c, Y2s, Y2c, Z2s, Z2c = (
            np.asarray(getattr(second, name)) for name in ("X2s", "X2c", "Y2s", "Y2c", "Z2s", "Z2c")
        )
        pressure = chi * B_bar * beta_s
        tables["w_11"] = vector(
            -2 * chi * B_bar * beta_2s + pressure * curvature * x - shape * Z2s,
            -pressure * (d_ds @ x - torsion * y_sigma) - shape * X2s,
            -pressure * (d_ds @ y_sigma + torsion * x) - shape * Y2s,
        )
        tables["w_22"] = vector(2 * chi * B_bar * beta_2s + shape * Z2s, shape * X2s, shape * Y2s)
        tables["w_12"] = vector(
            2 * shape * Z2c,
            pressure * torsion * y + 2 * shape * X2c,
            -pressure * (d_ds @ y) + 2 * shape * Y2c,
        )
    return tables, ell


def volume_field(solution, radius, section, labels=(0.0, 0.0), *, nodes=(300, 24, 48)):
    """Return the observation point and its plasma field from the unexpanded volume law.

    The observation is at the regular flux labels ``(q1, q2)`` of one axis section.
    Source labels are polar about the observation, and the toroidal nodes are uniform
    in the logarithm of the separation, which resolves the integrable coincidence.
    """

    toroidal, radial, angular = nodes
    tables, ell = _source_tables(solution)
    period = 2 * np.pi / int(solution.inputs.axis.nfp)
    phi_observation = float(np.asarray(solution.phi)[section])
    unit, unit_weight = np.polynomial.legendre.leggauss(toroidal)
    unit, unit_weight = (unit + 1) / 2, unit_weight / 2
    closest = 1.0e-8
    stretch = np.arcsinh(np.pi / closest)
    offset = closest * np.sinh(stretch * unit)
    weight = closest * stretch * np.cosh(stretch * unit) * unit_weight
    offset, weight = np.r_[-offset[::-1], offset], np.r_[weight[::-1], weight]
    phi = phi_observation + offset
    source = {k: _interpolate(v, period, np.mod(phi, period)) for k, v in tables.items()}
    local = {k: v[section : section + 1] for k, v in tables.items()}
    for key in source:
        if key != "d_varphi":
            source[key] = _to_cartesian(source[key], phi)
            local[key] = _to_cartesian(local[key], np.array([phi_observation]))

    def position(table, q1, q2):
        return (
            table["axis"]
            + q1 * table["map_1"]
            + q2 * table["map_2"]
            + (q1 * q1 + q2 * q2) * table["map_20"]
            + (q1 * q1 - q2 * q2) * table["map_2c"]
            + 2 * q1 * q2 * table["map_2s"]
        )

    observation = position(local, *labels)[0]
    angle = np.arange(angular) * 2 * np.pi / angular
    cosine, sine = np.cos(angle), np.sin(angle)
    projection = labels[0] * cosine + labels[1] * sine
    reach = -projection + np.sqrt(projection**2 + radius**2 - labels[0] ** 2 - labels[1] ** 2)
    node, node_weight = np.polynomial.legendre.leggauss(radial)
    node, node_weight = (node + 1) / 2, node_weight / 2
    toroidal_measure = (weight * source["d_varphi"][:, 0])[None, :, None]
    field = np.zeros(3)
    for fraction, fraction_weight in zip(node, node_weight, strict=True):
        distance = fraction * reach
        q1 = (labels[0] + distance * cosine)[:, None, None]
        q2 = (labels[1] + distance * sine)[:, None, None]
        current = source["w_0"] + q1 * source["w_1"] + q2 * source["w_2"]
        if "w_11" in source:
            current = current + q1 * q1 * source["w_11"] + q1 * q2 * source["w_12"]
            current = current + q2 * q2 * source["w_22"]
        separation = observation - position(source, q1, q2)
        kernel = np.cross(ell * current, separation)
        kernel /= np.linalg.norm(separation, axis=-1, keepdims=True) ** 3
        area = (fraction_weight * reach * distance * 2 * np.pi / angular)[:, None, None]
        field += np.sum(kernel * toroidal_measure * area, axis=(0, 1)) / (4 * np.pi)
    return observation, field


def _taylor_errors(solution, radius, section):
    """Odd (gradient) and even (Hessian) parts of the field at interior points."""

    data = qsc.plasma_hessian_on_axis(solution, formal_radius=radius)
    gradient = np.asarray(data.field.gradient)[section]
    hessian = np.asarray(data.hessian)[section]
    axis = np.asarray(solution.geometry.position_cartesian)[section]
    centre = volume_field(solution, radius, section)[1]

    def predicted(point):
        step = point - axis
        return gradient @ step + 0.5 * np.einsum("ijk,j,k->i", hessian, step, step)

    odd, even, odd_scale, even_scale = [], [], [], []
    for angle in np.arange(4) * np.pi / 4:
        labels = 0.25 * radius * np.array([np.cos(angle), np.sin(angle)])
        plus, field_plus = volume_field(solution, radius, section, tuple(labels))
        minus, field_minus = volume_field(solution, radius, section, tuple(-labels))
        odd_direct = (field_plus - field_minus) / 2
        even_direct = (field_plus + field_minus) / 2 - centre
        odd.append(np.linalg.norm(odd_direct - (predicted(plus) - predicted(minus)) / 2))
        even.append(np.linalg.norm(even_direct - (predicted(plus) + predicted(minus)) / 2))
        odd_scale.append(np.linalg.norm(odd_direct))
        even_scale.append(np.linalg.norm(even_direct))
    return max(odd) / max(odd_scale), max(even) / max(even_scale)


def _transverse_gradient(solution, radius, section):
    """Richardson-extracted normal and binormal derivatives, in Frenet components."""

    frame = _frame(solution, section)
    axis = np.asarray(solution.geometry.position_cartesian)[section]
    hessian = np.asarray(qsc.plasma_hessian_on_axis(solution, formal_radius=radius).hessian)
    hessian = hessian[section]

    def quadratic(point):
        return 0.5 * np.einsum("ijk,j,k->i", hessian, point - axis, point - axis)

    columns = []
    for direction in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        steps = []
        for fraction in (0.2, 0.1):
            labels = tuple(fraction * radius * direction)
            plus, field_plus = volume_field(solution, radius, section, labels)
            minus, field_minus = volume_field(solution, radius, section, tuple(-np.array(labels)))
            leakage = (quadratic(plus) - quadratic(minus)) / 2
            steps.append(((field_plus - field_minus) / 2 - leakage, (plus - minus) / 2))
        # Removes the cubic term: odd(h) = G h + C h**3.
        columns.append(tuple((8 * fine - coarse) / 3 for coarse, fine in zip(*steps, strict=True)))
    displacement = np.stack([frame @ columns[0][1], frame @ columns[1][1]], axis=1)[1:]
    change = np.stack([frame @ columns[0][0], frame @ columns[1][0]], axis=1)
    return change @ np.linalg.inv(displacement)


@pytest.mark.parametrize(
    ("parameters", "section"),
    [
        ({**QA, "I2": 0.0, "p2": -6.0e5}, SECTION),
        ({**QA, "I2": 0.6, "p2": -6.0e5}, SECTION),
        ({**AXISYMMETRIC, "I2": 0.9, "p2": -2.0e5}, 3),
    ],
    ids=["pressure-only", "current-and-pressure", "axisymmetric"],
)
def test_matched_axis_field_converges_to_volume_integral(parameters, section):
    solution = _solve(**parameters)
    errors = []
    for radius in (0.02, 0.01):
        matched = np.asarray(qsc.plasma_field_on_axis(solution, formal_radius=radius).field)
        direct = volume_field(solution, radius, section)[1]
        errors.append(np.linalg.norm(matched[section] - direct) / np.linalg.norm(direct))
    assert errors[1] < 2.0e-3
    # The field is order a**2 and its remainder is order a**4 log(a).
    assert 3.3 < errors[0] / errors[1] < 4.5


@pytest.mark.parametrize(
    ("parameters", "section"),
    [({**QA, "I2": 0.6, "p2": -6.0e5}, SECTION), ({**AXISYMMETRIC, "I2": 0.9, "p2": -2.0e5}, 3)],
    ids=["current-and-pressure", "axisymmetric"],
)
def test_finite_current_gradient_and_hessian_converge_to_volume_integral(parameters, section):
    solution = _solve(**parameters)
    coarse, fine = (_taylor_errors(solution, radius, section) for radius in (0.02, 0.01))
    # A transposed gradient would fail here at order one: its antisymmetric part is j.
    assert fine[0] < 1.0e-3
    assert fine[1] < 3.0e-3
    assert 3.0 < coarse[0] / fine[0] < 5.0
    assert 3.0 < coarse[1] / fine[1] < 5.0


def test_zero_current_hessian_converges_to_volume_integral():
    solution = _solve(**QA, I2=0.0, p2=-6.0e5)
    coarse, fine = (_taylor_errors(solution, radius, SECTION)[1] for radius in (0.02, 0.01))
    assert fine < 1.0e-2
    assert 3.0 < coarse / fine < 5.0


def test_zero_current_gradient_converges_to_volume_integral():
    solution = _solve(**QA, I2=0.0, p2=-6.0e5)
    frame = _frame(solution, SECTION)
    errors, sizes = [], []
    for radius in (0.01, 0.005):
        data = qsc.plasma_gradient_on_axis(solution, formal_radius=radius)
        predicted = (frame @ np.asarray(data.gradient)[SECTION] @ frame.T)[:, 1:]
        direct = _transverse_gradient(solution, radius, SECTION)
        errors.append(np.max(np.abs(direct - predicted)))
        sizes.append(np.max(np.abs(direct)))
    # The gradient is order a**2 and its remainder is order a**4 log(a).
    assert errors[0] < 2.0e-5
    assert errors[0] < 1.0e-2 * sizes[0]
    assert 3.5 < sizes[0] / sizes[1] < 4.5
    assert 12.0 < errors[0] / errors[1] < 20.0


def test_zero_current_field_and_gradient_obey_closed_forms_and_maxwell():
    solution = _solve(**QA, I2=0.0, p2=-6.0e5)
    radius = 0.01
    data = qsc.plasma_gradient_on_axis(solution, formal_radius=radius)
    inputs, geometry = solution.inputs, solution.geometry
    ell = geometry.abs_G0_over_B0
    x, sigma = solution.X1c, solution.Y1c / solution.Y1s
    denominator = (1 + x**2) ** 2 + sigma**2
    scale = 2 * MU0 * inputs.p2 / inputs.B0 * radius**2 * ell * inputs.etabar * x
    scale = scale / (solution.iotaN * denominator)
    expected = jnp.stack(
        (
            inputs.sG * MU0 * inputs.p2 / inputs.B0 * radius**2 * jnp.ones_like(x),
            inputs.sG * scale * sigma,
            -inputs.spsi * scale * (1 + x**2),
        ),
        axis=-1,
    )
    frame = jnp.stack(
        (geometry.tangent_cartesian, geometry.normal_cartesian, geometry.binormal_cartesian),
        axis=-2,
    )
    field_frenet = jnp.einsum("nai,ni->na", frame, data.field.field)
    np.testing.assert_allclose(field_frenet, expected, rtol=0, atol=1.0e-15)

    gradient = np.asarray(data.gradient)
    assert np.max(np.abs(gradient)) > 1.0e-3  # Not the vanishing uniform-channel term.
    np.testing.assert_allclose(gradient, np.swapaxes(gradient, -1, -2), atol=1.0e-16)
    assert float(data.maximum_divergence) < 1.0e-16
    assert float(data.maximum_external_asymmetry) < 1.0e-8
    assert float(data.maximum_external_trace) < 1.0e-8
    quarter = qsc.plasma_gradient_on_axis(solution, formal_radius=radius / 2)
    np.testing.assert_allclose(4 * np.asarray(quarter.gradient), gradient, rtol=1.0e-12, atol=0)


def test_axisymmetric_field_recovers_tokamak_vertical_field():
    major_radius, current, pressure, radius = 1.0, 0.9, -2.0e5, 0.01
    solution = _solve(**AXISYMMETRIC, I2=current, p2=pressure)
    field = np.asarray(qsc.plasma_field_on_axis(solution, formal_radius=radius).field)
    frenet = np.einsum("ai,i->a", _frame(solution, 3), field[3])
    tangent = radius**2 * (MU0 * pressure + current**2)
    binormal = (
        current
        * radius**2
        / (2 * major_radius)
        * (np.log(8 * major_radius / radius) - 1.25 - MU0 * pressure / current**2)
    )
    np.testing.assert_allclose(frenet, [tangent, 0.0, binormal], rtol=0, atol=1.0e-12)


def test_vacuum_solution_has_no_plasma_jet():
    solution = _solve(**QA, I2=0.0, p2=0.0)
    data = qsc.plasma_hessian_on_axis(solution, formal_radius=0.01)
    assert float(jnp.max(jnp.abs(data.field.field.field))) == 0.0
    assert float(jnp.max(jnp.abs(data.field.gradient))) == 0.0
    assert float(jnp.max(jnp.abs(data.hessian))) == 0.0
    np.testing.assert_array_equal(data.field.external_field, solution.B_axis)


def test_hessian_is_differentiable_through_a_circular_section():
    # A principal-axis (SVD) construction has an undefined derivative where the ellipse is a
    # circle, which makes every coil or axis optimization through such a point return NaN.
    def hessian(etabar):
        solution = qsc.near_axis(**{**AXISYMMETRIC, "etabar": etabar}, I2=0.4, p2=-2.0e5).solution
        return qsc.plasma_hessian_on_axis(solution, formal_radius=0.03).hessian

    tangent = jax.jacfwd(hessian)(1.0)
    step = 1.0e-6
    finite_difference = (hessian(1.0 + step) - hessian(1.0 - step)) / (2 * step)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    np.testing.assert_allclose(tangent, finite_difference, rtol=0, atol=1.0e-7)

from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.second_order import MU0


def finite_pressure_solution(order="r2", **overrides):
    parameters = {
        "rc": [1.0, 0.09],
        "zs": [0.0, -0.09],
        "nfp": 2,
        "etabar": 0.95,
        "I2": 0.9,
        "p2": -600000.0,
        "B2c": -0.7,
        "nphi": 31,
        "order": order,
    }
    parameters.update(overrides)
    return qsc.Qsc(**parameters)


def test_curvo_profile_values_margins_and_scaling():
    solution = finite_pressure_solution()
    criteria = qsc.Criteria.from_curvo_2025(major_radius=2.0, B0=3.0)
    report = criteria.evaluate(solution)

    assert report.profile_name == "curvo_2025"
    assert len(report.evaluations) == 10
    assert set(report.values) == set(report.margins)
    np.testing.assert_allclose(criteria.minimum_L_grad_B, 0.2)
    np.testing.assert_allclose(criteria.minimum_axis_radius, 0.6)
    np.testing.assert_allclose(criteria.minimum_singular_radius, 0.1)
    np.testing.assert_allclose(criteria.minimum_L_grad_grad_B, 0.2)
    np.testing.assert_allclose(criteria.maximum_B20_variation, 3.75)

    beta = -MU0 * solution.inputs.p2 * solution.r_singularity**2 / solution.inputs.B0**2
    np.testing.assert_allclose(report["beta"].value, beta)
    np.testing.assert_allclose(
        report["B20_variation"].margin,
        criteria.maximum_B20_variation - solution.B20_variation,
    )
    np.testing.assert_allclose(
        report["minimum_L_grad_B"].margin,
        np.min(solution.L_grad_B) - criteria.minimum_L_grad_B,
    )
    assert report["B20_variation"].sense == "max"
    assert report["B20_variation"].units == "T/m^2"


def test_profile_is_fully_configurable_and_reports_aggregate_pass():
    solution = finite_pressure_solution()
    report = qsc.Criteria(
        minimum_axis_length=-1.0,
        minimum_abs_iota=0.0,
        maximum_elongation=1.0e6,
        minimum_L_grad_B=0.0,
        minimum_axis_radius=0.0,
        minimum_singular_radius=0.0,
        minimum_L_grad_grad_B=0.0,
        maximum_B20_variation=1.0e6,
        minimum_beta=0.0,
        minimum_DMerc_times_r2=-1.0e6,
    ).evaluate(solution)

    assert report.passed
    assert all(bool(evaluation.passed) for evaluation in report.evaluations)
    with pytest.raises(KeyError, match="missing"):
        _ = report["missing"]


def test_strict_criteria_reject_equality_and_inclusive_criteria_accept_it():
    solution = finite_pressure_solution()
    baseline = qsc.Criteria.from_curvo_2025().evaluate(solution)
    criteria = qsc.Criteria.from_curvo_2025(
        minimum_axis_length=float(solution.axis_length),
        minimum_abs_iota=float(abs(solution.iota)),
        minimum_DMerc_times_r2=float(solution.DMerc_times_r2),
    )
    report = criteria.evaluate(solution)

    assert not bool(report["axis_length"].passed)
    assert bool(report["abs_iota"].passed)
    assert not bool(report["DMerc_times_r2"].passed)
    assert baseline["axis_length"].sense == "strict_min"


def test_criterion_margin_is_jittable_and_differentiable():
    criteria = qsc.Criteria.from_curvo_2025()

    def beta_margin(p2):
        return criteria.evaluate(finite_pressure_solution(nphi=15, p2=p2))["beta"].margin

    p2 = -600000.0
    value = beta_margin(p2)
    jitted = jax.jit(beta_margin)(p2)
    tangent = jax.jvp(beta_margin, (p2,), (1.0,))[1]
    step = 1.0
    finite_difference = (beta_margin(p2 + step) - beta_margin(p2 - step)) / (2 * step)

    np.testing.assert_allclose(jitted, value)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-6)


def test_profile_guards_and_second_order_requirement():
    with pytest.raises(ValueError, match="major_radius"):
        qsc.Criteria.from_curvo_2025(major_radius=0.0)
    with pytest.raises(ValueError, match="B0"):
        qsc.Criteria.from_curvo_2025(B0=0.0)
    with pytest.raises(ValueError, match="Unknown criteria"):
        qsc.Criteria.from_curvo_2025(unknown_threshold=1.0)
    with pytest.raises(ValueError, match="second-order"):
        qsc.Criteria.from_curvo_2025().evaluate(finite_pressure_solution(order="r1"))


def test_threshold_replacement_changes_only_selected_policy():
    criteria = qsc.Criteria.from_curvo_2025()
    changed = replace(criteria, minimum_abs_iota=0.4)

    assert criteria.minimum_abs_iota == 0.2
    assert changed.minimum_abs_iota == 0.4
    assert changed.maximum_elongation == criteria.maximum_elongation

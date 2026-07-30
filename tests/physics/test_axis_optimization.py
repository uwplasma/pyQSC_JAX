from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.axis_optimization import (
    _candidate_is_distinct,
    _copy_retained_axis_modes,
    _halton_box,
    _internal_from_physical,
    _physical_from_internal,
    _selector_value,
    _validate_problem,
    _verification_passes,
)


def circular_problem(**overrides):
    parameters = {
        "axis": qsc.Axis(rc=[1.0], zs=[0.0], nfp=1),
        "variable_indices": (0,),
        "lower_bounds": jnp.asarray([0.8]),
        "upper_bounds": jnp.asarray([1.2]),
        "etabar": 1.0,
        "I2": 0.1,
        "nphi": 15,
    }
    parameters.update(overrides)
    return qsc.AxisSearchProblem(**parameters)


def small_options(**overrides):
    parameters = {
        "coarse_samples": 2,
        "local_starts": 1,
        "maximum_iterations": 2,
        "verification_multipliers": (1, 2),
        "verified_zero_tolerance": 1.0e-10,
        "verification_tail_tolerance": 1.0e-8,
    }
    parameters.update(overrides)
    return qsc.AxisSearchOptions(**parameters)


def test_stellarator_symmetric_indices_and_low_discrepancy_box():
    axis = qsc.Axis(rc=[1.0, 0.1, 0.01], zs=[0.0, -0.1, 0.02], nfp=3)

    assert qsc.stellarator_symmetric_variable_indices(axis) == (1, 2, 10, 11)
    assert qsc.stellarator_symmetric_variable_indices(axis, modes=(2,)) == (2, 11)
    with pytest.raises(ValueError, match="modes"):
        qsc.stellarator_symmetric_variable_indices(axis, modes=(0,))
    with pytest.raises(ValueError, match="modes"):
        qsc.stellarator_symmetric_variable_indices(axis, modes=(3,))
    samples = _halton_box(3, 3, jnp.float64)
    np.testing.assert_allclose(
        samples,
        [
            [0.5, 1 / 3, 0.2],
            [0.25, 2 / 3, 0.4],
            [0.75, 1 / 9, 0.6],
        ],
    )


def test_bound_transform_round_trip_and_strict_interior():
    lower = jnp.asarray([-2.0, 1.0])
    upper = jnp.asarray([4.0, 5.0])
    physical = jnp.asarray([-1.0, 4.0])
    internal = _internal_from_physical(physical, lower, upper)
    reconstructed = _physical_from_internal(internal, lower, upper)

    np.testing.assert_allclose(reconstructed, physical)
    assert np.all(np.asarray(reconstructed) > np.asarray(lower))
    assert np.all(np.asarray(reconstructed) < np.asarray(upper))


def test_fourier_continuation_copies_retained_modes_and_runs_stages():
    low_axis = qsc.Axis(rc=[1.0], zs=[0.0], nfp=1)
    high_axis = qsc.Axis(rc=[0.9, 0.0], zs=[0.0, 0.0], nfp=1)
    copied = _copy_retained_axis_modes(low_axis, high_axis)
    np.testing.assert_allclose(copied.rc, [1.0, 0.0])
    with pytest.raises(ValueError, match="same nfp"):
        _copy_retained_axis_modes(low_axis, replace(high_axis, nfp=2))

    stages = (
        circular_problem(axis=low_axis, nphi=15),
        qsc.AxisSearchProblem(
            axis=high_axis,
            variable_indices=(1, 7),
            lower_bounds=jnp.asarray([-0.01, -0.01]),
            upper_bounds=jnp.asarray([0.01, 0.01]),
            etabar=1.0,
            I2=0.1,
            nphi=15,
        ),
    )
    continuation = qsc.continue_axis_search(
        stages,
        options=small_options(
            coarse_samples=1,
            maximum_iterations=1,
            verification_multipliers=(1,),
        ),
    )

    assert continuation.complete
    assert len(continuation.stages) == 2
    assert continuation.final is continuation.stages[-1]
    assert continuation.final.best is not None
    np.testing.assert_allclose(continuation.final.best.solution.axis.rc[0], 1.0)


def test_search_certifies_only_the_verified_nonnegative_zero():
    result = qsc.search_axis(circular_problem(), options=small_options())

    assert result.status == "verified_zero"
    assert result.global_certificate
    assert "global lower bound" in result.message
    assert result.best is not None
    assert result.best.feasible
    assert result.best.primary_residual < 1.0e-12
    assert result.best.local_report.converged
    assert result.best.local_report.rejected_steps >= 1
    assert result.local_starts_attempted == 1
    assert result.distinct_basins == 1
    assert result.search_budget == 3 + int(result.best.local_report.function_evaluations)
    np.testing.assert_allclose(result.coarse_variables[0], [1.0])
    assert np.all(np.asarray(result.coarse_feasible))
    assert not _candidate_is_distinct(
        result.best.variables,
        [result.best],
        jnp.asarray([0.8]),
        jnp.asarray([1.2]),
        1.0e-3,
    )


def test_search_failure_and_ill_conditioning_are_distinct_statuses():
    coarse_failure = qsc.search_axis(
        circular_problem(etabar=0.0),
        options=small_options(
            coarse_samples=1,
            verification_multipliers=(1,),
        ),
    )
    assert coarse_failure.status == "solver_failure"
    assert coarse_failure.best is None
    assert coarse_failure.local_starts_attempted == 0

    local_failure = qsc.search_axis(
        circular_problem(nphi=7),
        options=small_options(
            coarse_samples=1,
            maximum_iterations=1,
            verification_multipliers=(1,),
        ),
    )
    assert local_failure.status == "solver_failure"
    assert local_failure.best is None
    assert local_failure.local_starts_attempted == 1

    ill_conditioned = qsc.search_axis(
        circular_problem(),
        options=small_options(
            coarse_samples=1,
            maximum_iterations=1,
            verification_multipliers=(1,),
            maximum_linear_condition_number=1.0e-30,
        ),
    )
    assert ill_conditioned.status == "ill_conditioned"
    assert ill_conditioned.best is None
    assert ill_conditioned.local_starts_attempted == 1


def test_search_reports_no_feasible_candidate_for_failed_hard_profile():
    impossible = qsc.Criteria.from_curvo_2025(minimum_abs_iota=2.0)
    result = qsc.search_axis(
        circular_problem(
            criteria=impossible,
            selector="maximum_criteria_margin",
        ),
        options=small_options(verification_multipliers=(1,)),
        seeds=(jnp.asarray([0.9]),),
    )

    assert result.status == "no_feasible_candidate"
    assert not result.global_certificate
    assert result.best is not None
    assert not result.best.feasible
    assert result.best.criteria_report is not None
    assert not result.best.criteria_report.passed
    assert result.coarse_variables.shape == (4, 1)


def test_nonzero_search_improves_locally_without_global_claim():
    axis = qsc.Axis(
        rc=[1.0, 0.155, 0.0102],
        zs=[0.0, 0.154, 0.0111],
        nfp=2,
    )
    indices = qsc.stellarator_symmetric_variable_indices(axis, modes=(1,))
    problem = qsc.AxisSearchProblem(
        axis=axis,
        variable_indices=indices,
        lower_bounds=jnp.asarray([0.11, 0.11]),
        upper_bounds=jnp.asarray([0.19, 0.19]),
        etabar=0.64,
        nphi=15,
        selector="minimum_maximum_elongation",
    )
    result = qsc.search_axis(
        problem,
        options=small_options(
            coarse_samples=2,
            maximum_iterations=3,
            verified_zero_tolerance=1.0e-8,
            verification_tail_tolerance=1.0,
            verification_multipliers=(1,),
        ),
    )

    assert result.status == "best_found"
    assert not result.global_certificate
    assert "no global-minimum certificate" in result.message
    assert result.best.primary_residual < result.coarse_residuals[0]
    assert result.best.local_report.accepted_steps >= 1
    assert result.best.selector_value < 0


def test_verification_rejects_an_underresolved_nominal_zero():
    solution = qsc.optimize_B2c(
        qsc.Qsc(
            rc=[1.0, 0.155, 0.0102],
            zs=[0.0, 0.154, 0.0111],
            nfp=2,
            etabar=0.64,
            nphi=15,
            order="r2",
        )
    ).solution
    verification = qsc.verify_B20_resolution(solution, multipliers=(1, 2))

    assert not _verification_passes(
        verification,
        small_options(
            verified_zero_tolerance=0.05,
            verification_relative_tolerance=1.0e-6,
            verification_tail_tolerance=1.0e-6,
        ),
    )


def test_every_canonical_selector_has_documented_orientation():
    solution = qsc.Qsc(
        rc=[1.0, 0.09],
        zs=[0.0, -0.09],
        nfp=2,
        etabar=0.95,
        I2=0.9,
        p2=-600000.0,
        B2c=-0.7,
        nphi=15,
        order="r2",
    )
    criteria = qsc.Criteria.from_curvo_2025()
    report = criteria.evaluate(solution)
    base = circular_problem(axis=solution.axis, variable_indices=(1,))

    np.testing.assert_allclose(
        _selector_value(
            replace(base, selector="maximum_singular_radius"),
            solution,
            report,
        ),
        solution.r_singularity,
    )
    np.testing.assert_allclose(
        _selector_value(
            replace(base, selector="maximum_minimum_L_grad_B"),
            solution,
            report,
        ),
        np.min(solution.L_grad_B),
    )
    np.testing.assert_allclose(
        _selector_value(
            replace(base, selector="maximum_minimum_L_grad_grad_B"),
            solution,
            report,
        ),
        np.min(solution.L_grad_grad_B),
    )
    np.testing.assert_allclose(
        _selector_value(
            replace(base, selector="minimum_maximum_elongation"),
            solution,
            report,
        ),
        -np.max(solution.elongation),
    )
    assert (
        _selector_value(
            replace(base, selector="minimum_axis_sobolev_norm"),
            solution,
            report,
        )
        < 0
    )
    assert (
        _selector_value(
            replace(
                base,
                selector="target_axis_length",
                target_axis_length=float(solution.axis_length),
            ),
            solution,
            report,
        )
        == 0
    )
    assert np.isfinite(
        _selector_value(
            replace(
                base,
                criteria=criteria,
                selector="maximum_criteria_margin",
            ),
            solution,
            report,
        )
    )


@pytest.mark.parametrize(
    ("problem", "options", "message"),
    [
        (circular_problem(variable_indices=()), small_options(), "variable_indices"),
        (
            circular_problem(variable_indices=(0, 0)),
            small_options(),
            "variable_indices",
        ),
        (circular_problem(variable_indices=(4,)), small_options(), "variable_indices"),
        (circular_problem(lower_bounds=[0.8, 0.9]), small_options(), "bounds"),
        (circular_problem(lower_bounds=[1.2]), small_options(), "strictly ordered"),
        (circular_problem(lower_bounds=[1.0]), small_options(), "strictly inside"),
        (circular_problem(nphi=4), small_options(), "nphi"),
        (
            circular_problem(target_iota=0.1),
            small_options(),
            "target_iota",
        ),
        (
            circular_problem(solve_for="etabar"),
            small_options(),
            "target_iota",
        ),
        (
            circular_problem(solve_for="bad"),
            small_options(),
            "solve_for",
        ),
        (
            circular_problem(selector="target_axis_length"),
            small_options(),
            "target_axis_length",
        ),
        (
            circular_problem(selector="maximum_criteria_margin"),
            small_options(),
            "criteria",
        ),
        (
            circular_problem(selector="bad"),
            small_options(),
            "selector",
        ),
        (
            circular_problem(),
            small_options(coarse_samples=0),
            "positive integers",
        ),
        (
            circular_problem(),
            small_options(initial_damping=0.0),
            "must be positive",
        ),
        (
            circular_problem(),
            small_options(verification_multipliers=()),
            "nonempty",
        ),
    ],
)
def test_search_policy_guards(problem, options, message):
    with pytest.raises(ValueError, match=message):
        _validate_problem(problem, options)


def test_valid_inverse_search_policy_reaches_post_branch_validation():
    _validate_problem(
        circular_problem(
            solve_for="I2",
            target_iota=0.1,
        ),
        small_options(),
    )


def test_explicit_seed_guard():
    with pytest.raises(ValueError, match="explicit seed"):
        qsc.search_axis(
            circular_problem(),
            options=small_options(),
            seeds=(jnp.asarray([1.3]),),
        )
    with pytest.raises(ValueError, match="At least one"):
        qsc.continue_axis_search(())

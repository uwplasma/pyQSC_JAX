"""Branch-aware global-to-local optimization of magnetic-axis coefficients."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Literal

import jax
import jax.numpy as jnp

from pyqsc_jax.axis import Axis
from pyqsc_jax.criteria import Criteria, CriteriaReport
from pyqsc_jax.first_order import solve
from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.optimize import (
    B20Diagnostics,
    B20ResolutionVerification,
    optimize_B2c,
    verify_B20_resolution,
)

SearchStatus = Literal[
    "verified_zero",
    "best_found",
    "no_feasible_candidate",
    "branch_fold",
    "ill_conditioned",
    "solver_failure",
    "verification_failure",
]

Selector = Literal[
    "maximum_singular_radius",
    "maximum_minimum_L_grad_B",
    "maximum_minimum_L_grad_grad_B",
    "minimum_maximum_elongation",
    "minimum_axis_sobolev_norm",
    "target_axis_length",
    "maximum_criteria_margin",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LocalLeastSquaresReport:
    """Convergence and damping evidence for one local refinement."""

    initial_residual_norm: jax.Array
    residual_norm: jax.Array
    gradient_infinity_norm: jax.Array
    step_norm: jax.Array
    damping: jax.Array
    jacobian_condition_number: jax.Array
    iterations: jax.Array
    accepted_steps: jax.Array
    rejected_steps: jax.Array
    function_evaluations: jax.Array
    converged: jax.Array
    finite: jax.Array


@dataclass(frozen=True)
class AxisSearchProblem:
    """Physical inputs, bounded axis variables, and branch policy."""

    axis: Axis
    variable_indices: tuple[int, ...]
    lower_bounds: jax.Array
    upper_bounds: jax.Array
    etabar: float
    B0: float = 1.0
    sigma0: float = 0.0
    I2: float = 0.0
    p2: float = 0.0
    B2s: float = 0.0
    nphi: int = 31
    sG: int = 1
    spsi: int = 1
    solve_for: str = "iota"
    target_iota: float | None = None
    criteria: Criteria | None = None
    selector: Selector = "maximum_singular_radius"
    target_axis_length: float | None = None


@dataclass(frozen=True)
class AxisSearchOptions:
    """Deterministic exploration, local solve, clustering, and verification policy."""

    coarse_samples: int = 32
    local_starts: int = 4
    maximum_iterations: int = 20
    initial_damping: float = 1.0e-3
    residual_tolerance: float = 1.0e-9
    gradient_tolerance: float = 1.0e-8
    step_tolerance: float = 1.0e-10
    primary_tolerance: float = 1.0e-6
    verified_zero_tolerance: float = 1.0e-8
    verification_relative_tolerance: float = 5.0e-3
    verification_tail_tolerance: float = 1.0e-6
    verification_multipliers: tuple[int, ...] = (1, 2, 4)
    basin_distance_tolerance: float = 1.0e-3
    maximum_linear_condition_number: float = 1.0e14


@dataclass(frozen=True)
class AxisSearchCandidate:
    """One distinct locally refined basin and its independent checks."""

    variables: jax.Array
    solution: NearAxisSolution
    diagnostics: B20Diagnostics
    local_report: LocalLeastSquaresReport
    criteria_report: CriteriaReport | None
    verification: B20ResolutionVerification
    primary_residual: float
    selector_value: float
    feasible: bool


@dataclass(frozen=True)
class AxisSearchResult:
    """Reproducible search result with explicit global-certificate semantics."""

    status: SearchStatus
    best: AxisSearchCandidate | None
    basins: tuple[AxisSearchCandidate, ...]
    coarse_variables: jax.Array
    coarse_residuals: jax.Array
    coarse_feasible: jax.Array
    search_budget: int
    local_starts_attempted: int
    distinct_basins: int
    global_certificate: bool
    message: str


@dataclass(frozen=True)
class AxisSearchContinuation:
    """Warm-started low-to-high Fourier search stages."""

    stages: tuple[AxisSearchResult, ...]
    final: AxisSearchResult | None
    complete: bool


def stellarator_symmetric_variable_indices(
    axis: Axis,
    *,
    modes: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Return packed indices for nonconstant ``rc`` and ``zs`` coefficients."""

    if modes is None:
        modes = tuple(range(1, axis.nfourier))
    if any(
        not isinstance(mode, int) or isinstance(mode, bool) or mode < 1 or mode >= axis.nfourier
        for mode in modes
    ):
        raise ValueError("modes must contain valid positive retained Fourier modes.")
    zs_offset = 3 * axis.nfourier
    return tuple(modes) + tuple(zs_offset + mode for mode in modes)


def _first_primes(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % prime for prime in primes if prime <= sqrt(candidate)):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    while index:
        result += fraction * (index % base)
        index //= base
        fraction /= base
    return result


def _halton_box(count: int, dimension: int, dtype: jnp.dtype) -> jax.Array:
    primes = _first_primes(dimension)
    values = [[_radical_inverse(sample + 1, base) for base in primes] for sample in range(count)]
    return jnp.asarray(values, dtype=dtype)


def _validate_problem(problem: AxisSearchProblem, options: AxisSearchOptions) -> None:
    indices = problem.variable_indices
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("variable_indices must be nonempty and unique.")
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= problem.axis.dofs.size
        for index in indices
    ):
        raise ValueError("variable_indices contains an invalid packed-axis index.")
    lower = jnp.asarray(problem.lower_bounds)
    upper = jnp.asarray(problem.upper_bounds)
    if lower.shape != (len(indices),) or upper.shape != (len(indices),):
        raise ValueError("lower_bounds and upper_bounds must match variable_indices.")
    if not bool(jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper) & (lower < upper))):
        raise ValueError("Every variable bound must be finite and strictly ordered.")
    initial = problem.axis.dofs[jnp.asarray(indices)]
    if not bool(jnp.all((initial > lower) & (initial < upper))):
        raise ValueError("The initial axis variables must lie strictly inside their bounds.")
    if problem.nphi < 3 or problem.nphi % 2 != 1:
        raise ValueError("nphi must be an odd integer >= 3.")
    if problem.solve_for == "iota":
        if problem.target_iota is not None:
            raise ValueError("target_iota requires solve_for='etabar' or solve_for='I2'.")
    elif problem.solve_for in ("etabar", "I2"):
        if problem.target_iota is None:
            raise ValueError("target_iota is required for an inverse solve.")
    else:
        raise ValueError("solve_for must be 'iota', 'etabar', or 'I2'.")
    if problem.selector == "target_axis_length" and problem.target_axis_length is None:
        raise ValueError("target_axis_length is required by the selected policy.")
    if problem.selector == "maximum_criteria_margin" and problem.criteria is None:
        raise ValueError("maximum_criteria_margin requires a criteria profile.")
    if problem.selector not in (
        "maximum_singular_radius",
        "maximum_minimum_L_grad_B",
        "maximum_minimum_L_grad_grad_B",
        "minimum_maximum_elongation",
        "minimum_axis_sobolev_norm",
        "target_axis_length",
        "maximum_criteria_margin",
    ):
        raise ValueError("Unknown canonical selector.")

    positive_integers = (
        options.coarse_samples,
        options.local_starts,
        options.maximum_iterations,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in positive_integers
    ):
        raise ValueError("Search sample, start, and iteration counts must be positive integers.")
    positive_values = (
        options.initial_damping,
        options.residual_tolerance,
        options.gradient_tolerance,
        options.step_tolerance,
        options.primary_tolerance,
        options.verified_zero_tolerance,
        options.verification_relative_tolerance,
        options.verification_tail_tolerance,
        options.basin_distance_tolerance,
        options.maximum_linear_condition_number,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("Search tolerances, damping, and condition limit must be positive.")
    if not options.verification_multipliers:
        raise ValueError("verification_multipliers must be nonempty.")


def _physical_from_internal(
    internal: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return midpoint + half_width * jnp.tanh(internal)


def _internal_from_physical(
    physical: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return jnp.arctanh((physical - midpoint) / half_width)


def _solve_axis_candidate(
    problem: AxisSearchProblem,
    variables: jax.Array,
) -> tuple[NearAxisSolution, B20Diagnostics]:
    indices = jnp.asarray(problem.variable_indices)
    axis = problem.axis.with_dofs(problem.axis.dofs.at[indices].set(variables))
    keyword_arguments = {
        "axis": axis,
        "etabar": problem.etabar,
        "B0": problem.B0,
        "sigma0": problem.sigma0,
        "I2": problem.I2,
        "p2": problem.p2,
        "B2c": 0.0,
        "B2s": problem.B2s,
        "nphi": problem.nphi,
        "order": "r2",
        "sG": problem.sG,
        "spsi": problem.spsi,
        "solve_for": problem.solve_for,
    }
    if problem.solve_for != "iota":
        keyword_arguments["iota"] = problem.target_iota
    optimized = optimize_B2c(solve(**keyword_arguments))
    return optimized.solution, optimized.diagnostics


def _projected_residual(
    problem: AxisSearchProblem,
    variables: jax.Array,
) -> jax.Array:
    solution, diagnostics = _solve_axis_candidate(problem, variables)
    weights = solution.geometry.d_l_d_phi
    return jnp.sqrt(weights / jnp.sum(weights)) * diagnostics.anomaly / solution.inputs.B0


def _coarse_metrics(
    problem: AxisSearchProblem,
    variables: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    solution, diagnostics = _solve_axis_candidate(problem, variables)
    finite = (
        jnp.isfinite(diagnostics.weighted_l2)
        & solution.root_report.converged
        & solution.linear_report.converged
        & solution.geometry.diagnostics.frenet_valid
        & solution.geometry.diagnostics.cylindrical_coordinates_valid
    )
    return jnp.where(finite, diagnostics.weighted_l2, jnp.inf), finite


def _levenberg_marquardt(
    residual_function,
    initial: jax.Array,
    options: AxisSearchOptions,
) -> tuple[jax.Array, LocalLeastSquaresReport]:
    residual_and_jacobian = jax.jit(
        lambda value: (residual_function(value), jax.jacrev(residual_function)(value))
    )
    residual = jax.jit(residual_function)
    internal = initial
    current_residual, jacobian = residual_and_jacobian(internal)
    initial_norm = jnp.linalg.norm(current_residual)
    damping = options.initial_damping
    accepted = 0
    rejected = 0
    function_evaluations = 1
    converged = False
    step_norm = jnp.asarray(jnp.inf, dtype=initial.dtype)
    gradient_norm = jnp.asarray(jnp.inf, dtype=initial.dtype)

    for _iteration in range(1, options.maximum_iterations + 1):
        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ current_residual
        gradient_norm = jnp.linalg.norm(gradient, ord=jnp.inf)
        diagonal_scale = jnp.maximum(jnp.diag(normal_matrix), 1.0)
        step = -jnp.linalg.solve(
            normal_matrix + damping * jnp.diag(diagonal_scale),
            gradient,
        )
        step_norm = jnp.linalg.norm(step)
        candidate = internal + step
        candidate_residual = residual(candidate)
        function_evaluations += 1
        current_cost = 0.5 * jnp.sum(current_residual**2)
        candidate_cost = 0.5 * jnp.sum(candidate_residual**2)
        predicted_reduction = -gradient @ step - 0.5 * step @ normal_matrix @ step
        ratio = (current_cost - candidate_cost) / jnp.maximum(
            predicted_reduction,
            jnp.finfo(current_cost.dtype).tiny,
        )
        accept = bool(
            jnp.isfinite(candidate_cost)
            & jnp.isfinite(step_norm)
            & (candidate_cost < current_cost)
            & (ratio > 0)
        )
        if accept:
            internal = candidate
            current_residual, jacobian = residual_and_jacobian(internal)
            function_evaluations += 1
            accepted += 1
            damping = jnp.maximum(
                damping * jnp.maximum(1 / 3, 1 - (2 * ratio - 1) ** 3),
                jnp.finfo(current_cost.dtype).eps,
            )
        else:
            rejected += 1
            damping = damping * 2

        residual_norm = jnp.linalg.norm(current_residual)
        converged = bool(
            (residual_norm <= options.residual_tolerance)
            | (gradient_norm <= options.gradient_tolerance)
            | (step_norm <= options.step_tolerance * (1 + jnp.linalg.norm(internal)))
        )
        if converged:
            break

    normal_eigenvalues = jnp.maximum(
        jnp.linalg.eigvalsh(jacobian.T @ jacobian),
        0,
    )
    largest_eigenvalue = normal_eigenvalues[-1]
    smallest_eigenvalue = normal_eigenvalues[0]
    rank_deficient = smallest_eigenvalue <= (jnp.finfo(jacobian.dtype).eps * largest_eigenvalue)
    condition_number = jnp.where(
        rank_deficient,
        jnp.inf,
        jnp.sqrt(largest_eigenvalue / smallest_eigenvalue),
    )
    condition_number = jnp.where(jnp.isnan(condition_number), jnp.inf, condition_number)
    residual_converged = jnp.linalg.norm(current_residual) <= options.residual_tolerance
    derivative_finite = jnp.all(jnp.isfinite(jacobian))
    finite = (
        jnp.all(jnp.isfinite(internal))
        & jnp.all(jnp.isfinite(current_residual))
        & (derivative_finite | residual_converged)
    )
    return internal, LocalLeastSquaresReport(
        initial_residual_norm=initial_norm,
        residual_norm=jnp.linalg.norm(current_residual),
        gradient_infinity_norm=gradient_norm,
        step_norm=step_norm,
        damping=jnp.asarray(damping),
        jacobian_condition_number=condition_number,
        iterations=jnp.asarray(_iteration),
        accepted_steps=jnp.asarray(accepted),
        rejected_steps=jnp.asarray(rejected),
        function_evaluations=jnp.asarray(function_evaluations),
        converged=jnp.asarray(converged),
        finite=finite,
    )


def _axis_sobolev_norm(axis: Axis) -> jax.Array:
    modes = jnp.arange(axis.nfourier, dtype=axis.dofs.dtype)
    weights = (1 + modes**2) ** 2
    return jnp.sqrt(jnp.sum(weights * (axis.rc**2 + axis.rs**2 + axis.zc**2 + axis.zs**2)))


def _selector_value(
    problem: AxisSearchProblem,
    solution: NearAxisSolution,
    criteria_report: CriteriaReport | None,
) -> float:
    if problem.selector == "maximum_singular_radius":
        value = solution.r_singularity
    elif problem.selector == "maximum_minimum_L_grad_B":
        value = jnp.min(solution.L_grad_B)
    elif problem.selector == "maximum_minimum_L_grad_grad_B":
        value = jnp.min(solution.L_grad_grad_B)
    elif problem.selector == "minimum_maximum_elongation":
        value = -jnp.max(solution.elongation)
    elif problem.selector == "minimum_axis_sobolev_norm":
        value = -_axis_sobolev_norm(solution.inputs.axis)
    elif problem.selector == "target_axis_length":
        value = -jnp.abs(solution.axis_length - problem.target_axis_length)
    else:
        normalized_margins = [
            evaluation.margin / jnp.maximum(jnp.abs(evaluation.threshold), 1.0)
            for evaluation in criteria_report.evaluations
        ]
        value = jnp.min(jnp.stack(normalized_margins))
    return float(value)


def _candidate_is_distinct(
    variables: jax.Array,
    candidates: list[AxisSearchCandidate],
    lower: jax.Array,
    upper: jax.Array,
    tolerance: float,
) -> bool:
    scaled = (variables - lower) / (upper - lower)
    return all(
        float(jnp.linalg.norm(scaled - (candidate.variables - lower) / (upper - lower))) > tolerance
        for candidate in candidates
    )


def _copy_retained_axis_modes(source: Axis, target: Axis) -> Axis:
    if source.nfp != target.nfp:
        raise ValueError("Fourier continuation stages must use the same nfp.")
    retained = min(source.nfourier, target.nfourier)
    return Axis(
        rc=target.rc.at[:retained].set(source.rc[:retained]),
        rs=target.rs.at[:retained].set(source.rs[:retained]),
        zc=target.zc.at[:retained].set(source.zc[:retained]),
        zs=target.zs.at[:retained].set(source.zs[:retained]),
        nfp=target.nfp,
    )


def _verification_passes(
    verification: B20ResolutionVerification,
    options: AxisSearchOptions,
) -> bool:
    weighted_zero = verification.weighted_l2 <= options.verified_zero_tolerance
    maximum_zero = verification.grid_maximum <= options.verified_zero_tolerance
    relative_l2_stable = (
        verification.relative_weighted_l2_change <= options.verification_relative_tolerance
    ) | weighted_zero
    relative_maximum_stable = (
        verification.relative_grid_maximum_change <= options.verification_relative_tolerance
    ) | maximum_zero
    spectral_tail_resolved = (
        verification.fourier_tail_ratio[-1] <= options.verification_tail_tolerance
    ) | maximum_zero[-1]
    return bool(
        jnp.all(weighted_zero)
        & jnp.all(maximum_zero)
        & jnp.all(relative_l2_stable)
        & jnp.all(relative_maximum_stable)
        & spectral_tail_resolved
    )


def search_axis(
    problem: AxisSearchProblem,
    *,
    options: AxisSearchOptions | None = None,
    seeds: tuple[jax.Array, ...] = (),
) -> AxisSearchResult:
    """Explore, refine, cluster, and independently verify bounded axis candidates."""

    if options is None:
        options = AxisSearchOptions()
    _validate_problem(problem, options)
    lower = jnp.asarray(problem.lower_bounds, dtype=problem.axis.dofs.dtype)
    upper = jnp.asarray(problem.upper_bounds, dtype=problem.axis.dofs.dtype)
    initial = problem.axis.dofs[jnp.asarray(problem.variable_indices)]
    for seed in seeds:
        seed_array = jnp.asarray(seed)
        if seed_array.shape != initial.shape or not bool(
            jnp.all(jnp.isfinite(seed_array) & (seed_array > lower) & (seed_array < upper))
        ):
            raise ValueError("Every explicit seed must be finite, in bounds, and correctly sized.")

    samples = lower + (upper - lower) * _halton_box(
        options.coarse_samples,
        len(problem.variable_indices),
        initial.dtype,
    )
    explicit = jnp.stack((initial, *(jnp.asarray(seed) for seed in seeds)))
    coarse_variables = jnp.concatenate((explicit, samples), axis=0)
    coarse_residuals, coarse_feasible = jax.jit(
        jax.vmap(lambda variables: _coarse_metrics(problem, variables))
    )(coarse_variables)

    finite_indices = [
        int(index) for index in jnp.argsort(coarse_residuals) if bool(coarse_feasible[index])
    ]
    selected_indices = finite_indices[: options.local_starts]
    if not selected_indices:
        return AxisSearchResult(
            status="solver_failure",
            best=None,
            basins=(),
            coarse_variables=coarse_variables,
            coarse_residuals=coarse_residuals,
            coarse_feasible=coarse_feasible,
            search_budget=int(coarse_variables.shape[0]),
            local_starts_attempted=0,
            distinct_basins=0,
            global_certificate=False,
            message="No coarse candidate produced a finite converged near-axis solve.",
        )

    residual_function = lambda internal: _projected_residual(  # noqa: E731
        problem,
        _physical_from_internal(internal, lower, upper),
    )
    candidates: list[AxisSearchCandidate] = []
    local_evaluations = 0
    ill_conditioned_count = 0
    for index in selected_indices:
        start = coarse_variables[index]
        internal_start = _internal_from_physical(start, lower, upper)
        internal, local_report = _levenberg_marquardt(
            residual_function,
            internal_start,
            options,
        )
        local_evaluations += int(local_report.function_evaluations)
        variables = _physical_from_internal(internal, lower, upper)
        if not bool(local_report.finite):
            continue
        solution, diagnostics = _solve_axis_candidate(problem, variables)
        if solution.linear_report.matrix_condition_number > options.maximum_linear_condition_number:
            ill_conditioned_count += 1
            continue
        criteria_report = None if problem.criteria is None else problem.criteria.evaluate(solution)
        feasible = bool(
            solution.root_report.converged
            & solution.linear_report.converged
            & solution.geometry.diagnostics.frenet_valid
            & solution.geometry.diagnostics.cylindrical_coordinates_valid
        ) and (criteria_report is None or criteria_report.passed)
        if not _candidate_is_distinct(
            variables,
            candidates,
            lower,
            upper,
            options.basin_distance_tolerance,
        ):
            continue
        verification = verify_B20_resolution(
            solution,
            multipliers=options.verification_multipliers,
        )
        candidates.append(
            AxisSearchCandidate(
                variables=variables,
                solution=solution,
                diagnostics=diagnostics,
                local_report=local_report,
                criteria_report=criteria_report,
                verification=verification,
                primary_residual=float(diagnostics.weighted_l2),
                selector_value=_selector_value(problem, solution, criteria_report),
                feasible=feasible,
            )
        )

    search_budget = int(coarse_variables.shape[0]) + local_evaluations
    if not candidates:
        status: SearchStatus = (
            "ill_conditioned"
            if ill_conditioned_count == len(selected_indices)
            else "solver_failure"
        )
        return AxisSearchResult(
            status=status,
            best=None,
            basins=(),
            coarse_variables=coarse_variables,
            coarse_residuals=coarse_residuals,
            coarse_feasible=coarse_feasible,
            search_budget=search_budget,
            local_starts_attempted=len(selected_indices),
            distinct_basins=0,
            global_certificate=False,
            message="Local refinement did not produce a finite, acceptable candidate.",
        )

    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        best = min(candidates, key=lambda candidate: candidate.primary_residual)
        return AxisSearchResult(
            status="no_feasible_candidate",
            best=best,
            basins=tuple(candidates),
            coarse_variables=coarse_variables,
            coarse_residuals=coarse_residuals,
            coarse_feasible=coarse_feasible,
            search_budget=search_budget,
            local_starts_attempted=len(selected_indices),
            distinct_basins=len(candidates),
            global_certificate=False,
            message="The search found converged basins, but none passed the hard criteria.",
        )

    primary_feasible = [
        candidate
        for candidate in feasible
        if candidate.primary_residual <= options.primary_tolerance
    ]
    if primary_feasible:
        best = max(primary_feasible, key=lambda candidate: candidate.selector_value)
    else:
        best = min(feasible, key=lambda candidate: candidate.primary_residual)

    if best.solution.inverse is not None and bool(best.solution.branch_fold):
        status = "branch_fold"
        message = "The selected inverse-solve candidate lies at a detected branch fold."
    elif best.primary_residual <= options.verified_zero_tolerance:
        if _verification_passes(best.verification, options):
            status = "verified_zero"
            message = (
                "The independently verified nonnegative B20 residual reaches its "
                "global lower bound of zero within the requested tolerances."
            )
        else:
            status = "verification_failure"
            message = "The nominal zero failed independent resolution or spectral checks."
    else:
        status = "best_found"
        message = (
            "This is the best basin found within the reported search budget; "
            "there is no global-minimum certificate."
        )
    return AxisSearchResult(
        status=status,
        best=best,
        basins=tuple(candidates),
        coarse_variables=coarse_variables,
        coarse_residuals=coarse_residuals,
        coarse_feasible=coarse_feasible,
        search_budget=search_budget,
        local_starts_attempted=len(selected_indices),
        distinct_basins=len(candidates),
        global_certificate=status == "verified_zero",
        message=message,
    )


def continue_axis_search(
    stages: tuple[AxisSearchProblem, ...],
    *,
    options: AxisSearchOptions | None = None,
    initial_seeds: tuple[jax.Array, ...] = (),
) -> AxisSearchContinuation:
    """Warm-start successively richer Fourier problems from the prior best basin."""

    if not stages:
        raise ValueError("At least one Fourier-continuation stage is required.")
    results: list[AxisSearchResult] = []
    previous_solution: NearAxisSolution | None = None
    for stage_index, stage in enumerate(stages):
        if previous_solution is not None:
            stage = replace(
                stage,
                axis=_copy_retained_axis_modes(previous_solution.inputs.axis, stage.axis),
            )
        result = search_axis(
            stage,
            options=options,
            seeds=initial_seeds if stage_index == 0 else (),
        )
        results.append(result)
        if result.best is None:
            break
        previous_solution = result.best.solution
    complete = len(results) == len(stages) and all(result.best is not None for result in results)
    return AxisSearchContinuation(
        stages=tuple(results),
        final=results[-1] if results else None,
        complete=complete,
    )

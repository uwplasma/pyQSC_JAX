"""Named, scalable stellarator design-criteria profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.second_order import MU0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CriterionEvaluation:
    """One signed criterion margin and its pass/fail result."""

    value: jax.Array
    threshold: jax.Array
    margin: jax.Array
    passed: jax.Array
    name: str = field(metadata={"static": True})
    sense: str = field(metadata={"static": True})
    units: str = field(metadata={"static": True})


@dataclass(frozen=True)
class CriteriaReport:
    """Evaluation of every item in one named criteria profile."""

    profile_name: str
    evaluations: tuple[CriterionEvaluation, ...]

    @property
    def passed(self) -> bool:
        """Whether every criterion passes."""

        return all(bool(evaluation.passed) for evaluation in self.evaluations)

    @property
    def margins(self) -> dict[str, jax.Array]:
        """Signed raw margins keyed by criterion name."""

        return {evaluation.name: evaluation.margin for evaluation in self.evaluations}

    @property
    def values(self) -> dict[str, jax.Array]:
        """Measured values keyed by criterion name."""

        return {evaluation.name: evaluation.value for evaluation in self.evaluations}

    def __getitem__(self, name: str) -> CriterionEvaluation:
        for evaluation in self.evaluations:
            if evaluation.name == name:
                return evaluation
        raise KeyError(name)


@dataclass(frozen=True)
class Criteria:
    """Configurable thresholds for a named near-axis design profile."""

    minimum_axis_length: float
    minimum_abs_iota: float
    maximum_elongation: float
    minimum_L_grad_B: float
    minimum_axis_radius: float
    minimum_singular_radius: float
    minimum_L_grad_grad_B: float
    maximum_B20_variation: float
    minimum_beta: float
    minimum_DMerc_times_r2: float
    profile_name: str = "custom"

    @classmethod
    def from_curvo_2025(
        cls,
        *,
        major_radius: float = 1.0,
        B0: float = 1.0,
        **overrides: float,
    ) -> Criteria:
        """Return the scalable version of Curvo et al. (2025), Table 3."""

        if major_radius <= 0:
            raise ValueError("major_radius must be positive.")
        if B0 <= 0:
            raise ValueError("B0 must be positive.")
        parameters = {
            "minimum_axis_length": 0.0,
            "minimum_abs_iota": 0.2,
            "maximum_elongation": 10.0,
            "minimum_L_grad_B": 0.1 * major_radius,
            "minimum_axis_radius": 0.3 * major_radius,
            "minimum_singular_radius": 0.05 * major_radius,
            "minimum_L_grad_grad_B": 0.1 * major_radius,
            "maximum_B20_variation": 5.0 * B0 / major_radius**2,
            "minimum_beta": 1.0e-4,
            "minimum_DMerc_times_r2": 0.0,
            "profile_name": "curvo_2025",
        }
        unknown = set(overrides) - set(parameters)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown criteria override(s): {names}.")
        parameters.update(overrides)
        return cls(**parameters)

    def evaluate(self, solution: NearAxisSolution) -> CriteriaReport:
        """Evaluate values, signed margins, and pass flags."""

        if solution.second_order is None:
            raise ValueError("The criteria profile requires a second-order solution.")
        beta = -MU0 * solution.inputs.p2 * solution.r_singularity**2 / solution.inputs.B0**2
        values = (
            (
                "axis_length",
                solution.axis_length,
                self.minimum_axis_length,
                "strict_min",
                "m",
            ),
            ("abs_iota", jnp.abs(solution.iota), self.minimum_abs_iota, "min", "1"),
            (
                "maximum_elongation",
                jnp.max(solution.elongation),
                self.maximum_elongation,
                "max",
                "1",
            ),
            (
                "minimum_L_grad_B",
                jnp.min(solution.L_grad_B),
                self.minimum_L_grad_B,
                "min",
                "m",
            ),
            (
                "minimum_axis_radius",
                jnp.min(solution.R0),
                self.minimum_axis_radius,
                "min",
                "m",
            ),
            (
                "singular_radius",
                solution.r_singularity,
                self.minimum_singular_radius,
                "min",
                "m",
            ),
            (
                "minimum_L_grad_grad_B",
                jnp.min(solution.L_grad_grad_B),
                self.minimum_L_grad_grad_B,
                "min",
                "m",
            ),
            (
                "B20_variation",
                solution.B20_variation,
                self.maximum_B20_variation,
                "max",
                "T/m^2",
            ),
            ("beta", beta, self.minimum_beta, "min", "1"),
            (
                "DMerc_times_r2",
                solution.DMerc_times_r2,
                self.minimum_DMerc_times_r2,
                "strict_min",
                "1",
            ),
        )
        evaluations = []
        for name, value, threshold, sense, units in values:
            value = jnp.asarray(value)
            threshold = jnp.asarray(threshold, dtype=value.dtype)
            if sense == "max":
                margin = threshold - value
                passed = value <= threshold
            else:
                margin = value - threshold
                passed = value > threshold if sense == "strict_min" else value >= threshold
            evaluations.append(
                CriterionEvaluation(
                    value=value,
                    threshold=threshold,
                    margin=margin,
                    passed=passed,
                    name=name,
                    sense=sense,
                    units=units,
                )
            )
        return CriteriaReport(
            profile_name=self.profile_name,
            evaluations=tuple(evaluations),
        )

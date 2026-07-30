"""Small immutable result and diagnostic models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp

from pyqsc_jax.axis import Axis

if TYPE_CHECKING:
    from pyqsc_jax.geometry import AxisGeometry


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GeometryDiagnostics:
    """Validity and conditioning information for sampled axis geometry."""

    minimum_speed: jax.Array
    minimum_curvature: jax.Array
    minimum_cylindrical_radius: jax.Array
    maximum_frame_orthogonality_error: jax.Array
    minimum_frame_determinant: jax.Array
    frenet_valid: jax.Array
    cylindrical_coordinates_valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RootSolveReport:
    """Convergence evidence for a nonlinear root solve."""

    initial_residual_norm: jax.Array
    residual_norm: jax.Array
    tolerance: jax.Array
    step_norm: jax.Array
    iterations: jax.Array
    backtracking_steps: jax.Array
    jacobian_condition_number: jax.Array
    converged: jax.Array
    finite: jax.Array
    stagnated: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearSolveReport:
    """Residual and conditioning evidence for a dense linear solve."""

    residual_norm: jax.Array
    relative_residual_norm: jax.Array
    matrix_condition_number: jax.Array
    finite: jax.Array
    converged: jax.Array
    well_conditioned: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NearAxisInputs:
    """Normalized immutable inputs to a near-axis solve."""

    axis: Axis
    etabar: jax.Array
    B0: jax.Array = 1.0
    sigma0: jax.Array = 0.0
    I2: jax.Array = 0.0
    p2: jax.Array = 0.0
    B2c: jax.Array = 0.0
    B2s: jax.Array = 0.0
    nphi: int = field(default=61, metadata={"static": True})
    order: int = field(default=1, metadata={"static": True})
    sG: int = field(default=1, metadata={"static": True})
    spsi: int = field(default=1, metadata={"static": True})
    solve_for: str = field(default="iota", metadata={"static": True})

    def __post_init__(self) -> None:
        if not isinstance(self.nphi, int) or isinstance(self.nphi, bool) or self.nphi < 3:
            raise ValueError("nphi must be an integer >= 3.")
        if self.order not in (1, 2, 3):
            raise ValueError("order must be 1, 2, or 3.")
        if self.sG not in (-1, 1):
            raise ValueError("sG must be +1 or -1.")
        if self.spsi not in (-1, 1):
            raise ValueError("spsi must be +1 or -1.")
        if self.solve_for not in ("iota", "etabar", "I2"):
            raise ValueError("solve_for must be 'iota', 'etabar', or 'I2'.")
        for name in ("etabar", "B0", "sigma0", "I2", "p2", "B2c", "B2s"):
            value = jnp.asarray(getattr(self, name))
            if value.ndim:
                raise ValueError(f"{name} must be a scalar.")
            object.__setattr__(self, name, value)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SecondOrderData:
    """Complete second-order coefficient solution and direct diagnostics."""

    linear_report: LinearSolveReport
    V1: jax.Array
    V2: jax.Array
    V3: jax.Array
    X20: jax.Array
    X2s: jax.Array
    X2c: jax.Array
    Y20: jax.Array
    Y2s: jax.Array
    Y2c: jax.Array
    Z20: jax.Array
    Z2s: jax.Array
    Z2c: jax.Array
    beta_1s: jax.Array
    B20: jax.Array
    B20_mean: jax.Array
    B20_anomaly: jax.Array
    B20_residual: jax.Array
    B20_variation: jax.Array
    G2: jax.Array
    N_helicity: jax.Array
    d_curvature_d_varphi: jax.Array
    d_torsion_d_varphi: jax.Array
    d_X20_d_varphi: jax.Array
    d_X2s_d_varphi: jax.Array
    d_X2c_d_varphi: jax.Array
    d_Y20_d_varphi: jax.Array
    d_Y2s_d_varphi: jax.Array
    d_Y2c_d_varphi: jax.Array
    d_Z20_d_varphi: jax.Array
    d_Z2s_d_varphi: jax.Array
    d_Z2c_d_varphi: jax.Array
    d2_X1c_d_varphi2: jax.Array
    d2_Y1c_d_varphi2: jax.Array
    d2_Y1s_d_varphi2: jax.Array
    X20_untwisted: jax.Array
    X2s_untwisted: jax.Array
    X2c_untwisted: jax.Array
    Y20_untwisted: jax.Array
    Y2s_untwisted: jax.Array
    Y2c_untwisted: jax.Array
    Z20_untwisted: jax.Array
    Z2s_untwisted: jax.Array
    Z2c_untwisted: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NearAxisSolution:
    """Canonical immutable near-axis solution.

    Sampled vector arrays use a leading ``nphi`` axis. Vector components are
    Cartesian unless a field name explicitly contains ``cylindrical``.
    """

    inputs: NearAxisInputs
    geometry: AxisGeometry
    root_report: RootSolveReport
    sigma: jax.Array
    iota: jax.Array
    iotaN: jax.Array
    helicity: jax.Array
    G0: jax.Array
    X1s: jax.Array
    X1c: jax.Array
    Y1s: jax.Array
    Y1c: jax.Array
    X1s_untwisted: jax.Array
    X1c_untwisted: jax.Array
    Y1s_untwisted: jax.Array
    Y1c_untwisted: jax.Array
    elongation: jax.Array
    mean_elongation: jax.Array
    B_axis_cylindrical: jax.Array
    B_axis: jax.Array
    grad_B_axis_cylindrical: jax.Array
    grad_B_axis: jax.Array
    L_grad_B: jax.Array
    second_order: SecondOrderData | None = None

    _SECOND_ORDER_NAMES: ClassVar[frozenset[str]] = frozenset(
        field.name for field in SecondOrderData.__dataclass_fields__.values()
    )

    def __getattr__(self, name: str):
        if name in self._SECOND_ORDER_NAMES:
            second_order = object.__getattribute__(self, "second_order")
            if second_order is None:
                raise AttributeError(f"First-order solution has no {name!r} quantity.")
            return getattr(second_order, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}.")

    @property
    def axis(self) -> Axis:
        return self.inputs.axis

    @property
    def phi(self) -> jax.Array:
        return self.geometry.samples.phi

    @property
    def varphi(self) -> jax.Array:
        return self.geometry.varphi

    @property
    def R0(self) -> jax.Array:
        return self.geometry.samples.R

    @property
    def Z0(self) -> jax.Array:
        return self.geometry.samples.Z

    @property
    def curvature(self) -> jax.Array:
        return self.geometry.curvature

    @property
    def torsion(self) -> jax.Array:
        return self.geometry.torsion

    @property
    def axis_length(self) -> jax.Array:
        return self.geometry.axis_length

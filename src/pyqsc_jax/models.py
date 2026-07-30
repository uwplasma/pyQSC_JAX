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
class InverseSolveDiagnostics:
    """Local branch and fold diagnostics for a target-transform solve."""

    target_iota: jax.Array
    achieved_iota: jax.Array
    solved_value: jax.Array
    response_derivative: jax.Array
    absolute_response_derivative: jax.Array
    fold_tolerance: jax.Array
    branch_fold: jax.Array
    parameter_sign: jax.Array
    parameter: str = field(metadata={"static": True})


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
class MercierDiagnostics:
    """Leading near-axis magnetic-well and Mercier contributions."""

    d2_volume_d_psi2: jax.Array
    DGeod_times_r2: jax.Array
    DWell_times_r2: jax.Array
    DMerc_times_r2: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FieldJet:
    """Total on-axis magnetic field through two Cartesian derivatives.

    ``gradient`` and ``hessian`` use field-component-first ordering:
    ``gradient[n, i, j] = d B_i / d x_j`` and
    ``hessian[n, i, j, k] = d² B_i / (d x_j d x_k)``.
    """

    field: jax.Array
    gradient: jax.Array
    hessian: jax.Array
    hessian_frenet: jax.Array
    coordinate_jacobian: jax.Array
    inverse_coordinate_jacobian: jax.Array
    coordinate_hessian: jax.Array
    minimum_absolute_coordinate_jacobian: jax.Array
    maximum_field_error: jax.Array
    maximum_gradient_error: jax.Array
    maximum_divergence: jax.Array
    maximum_derivative_asymmetry: jax.Array
    maximum_divergence_gradient: jax.Array
    grad_grad_B_inverse_scale_length_vs_varphi: jax.Array
    L_grad_grad_B: jax.Array
    grad_grad_B_inverse_scale_length: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SingularityDiagnostics:
    """First loss of regularity in the quadratic near-axis coordinate map."""

    r_singularity: jax.Array
    r_singularity_vs_varphi: jax.Array
    inv_r_singularity_vs_varphi: jax.Array
    theta_singularity_vs_varphi: jax.Array
    residual_norm_vs_varphi: jax.Array
    maximum_residual_norm: jax.Array
    g0: jax.Array
    g1c: jax.Array
    g1s: jax.Array
    g20: jax.Array
    g2s: jax.Array
    g2c: jax.Array
    angular_resolution: int = field(metadata={"static": True})
    newton_iterations: int = field(metadata={"static": True})


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
class ThirdOrderData:
    """Third-order flux-constraint surface corrections."""

    flux_constraint_coefficient: jax.Array
    B0_order_a_squared_to_cancel: jax.Array
    flux_constraint_residual: jax.Array
    consistency_error: jax.Array
    X3s1: jax.Array
    X3c1: jax.Array
    Y3s1: jax.Array
    Y3c1: jax.Array
    Z3s1: jax.Array
    Z3c1: jax.Array
    X3s3: jax.Array
    X3c3: jax.Array
    Y3s3: jax.Array
    Y3c3: jax.Array
    Z3s3: jax.Array
    Z3c3: jax.Array
    d_X3c1_d_varphi: jax.Array
    d_Y3s1_d_varphi: jax.Array
    d_Y3c1_d_varphi: jax.Array
    X3s1_untwisted: jax.Array
    X3c1_untwisted: jax.Array
    Y3s1_untwisted: jax.Array
    Y3c1_untwisted: jax.Array
    Z3s1_untwisted: jax.Array
    Z3c1_untwisted: jax.Array
    X3s3_untwisted: jax.Array
    X3c3_untwisted: jax.Array
    Y3s3_untwisted: jax.Array
    Y3c3_untwisted: jax.Array
    Z3s3_untwisted: jax.Array
    Z3c3_untwisted: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ShearData:
    """Order-r-squared rotational-transform correction and intermediates."""

    B31c: jax.Array
    iota2: jax.Array
    numerator: jax.Array
    denominator: jax.Array
    Lambda_tilde: jax.Array
    integrating_factor: jax.Array
    sigma_average: jax.Array
    Z31c: jax.Array
    Z31s: jax.Array
    X31c: jax.Array
    X31s: jax.Array
    Y31s: jax.Array
    stellarator_symmetric: jax.Array


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
    mercier: MercierDiagnostics | None = None
    field_jet: FieldJet | None = None
    singularity: SingularityDiagnostics | None = None
    third_order: ThirdOrderData | None = None
    shear: ShearData | None = None
    inverse: InverseSolveDiagnostics | None = None

    _SECOND_ORDER_NAMES: ClassVar[frozenset[str]] = frozenset(
        field.name for field in SecondOrderData.__dataclass_fields__.values()
    )
    _THIRD_ORDER_NAMES: ClassVar[frozenset[str]] = frozenset(
        field.name for field in ThirdOrderData.__dataclass_fields__.values()
    )
    _SHEAR_NAMES: ClassVar[frozenset[str]] = frozenset(
        field.name for field in ShearData.__dataclass_fields__.values()
    )
    _INVERSE_NAMES: ClassVar[frozenset[str]] = frozenset(
        field.name for field in InverseSolveDiagnostics.__dataclass_fields__.values()
    )
    _MERCIER_NAMES: ClassVar[frozenset[str]] = frozenset(
        field.name for field in MercierDiagnostics.__dataclass_fields__.values()
    )
    _FIELD_JET_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "grad_grad_B_axis",
            "grad_grad_B",
            "L_grad_grad_B",
            "grad_grad_B_inverse_scale_length_vs_varphi",
            "grad_grad_B_inverse_scale_length",
        }
    )
    _SINGULARITY_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "r_singularity",
            "r_singularity_vs_varphi",
            "inv_r_singularity_vs_varphi",
            "r_singularity_basic_vs_varphi",
            "r_singularity_theta_vs_varphi",
            "r_singularity_residual_sqnorm",
        }
    )

    def __getattr__(self, name: str):
        if name in self._SECOND_ORDER_NAMES:
            second_order = object.__getattribute__(self, "second_order")
            if second_order is None:
                raise AttributeError(f"First-order solution has no {name!r} quantity.")
            return getattr(second_order, name)
        if name in self._THIRD_ORDER_NAMES:
            third_order = object.__getattribute__(self, "third_order")
            if third_order is None:
                raise AttributeError(f"Lower-order solution has no {name!r} quantity.")
            return getattr(third_order, name)
        if name in self._SHEAR_NAMES:
            shear = object.__getattribute__(self, "shear")
            if shear is None:
                raise AttributeError(
                    f"Magnetic shear has not been calculated; no {name!r} quantity."
                )
            return getattr(shear, name)
        if name in self._INVERSE_NAMES:
            inverse = object.__getattribute__(self, "inverse")
            if inverse is None:
                raise AttributeError(f"Forward solution has no inverse diagnostic {name!r}.")
            return getattr(inverse, name)
        if name in self._MERCIER_NAMES:
            mercier = object.__getattribute__(self, "mercier")
            if mercier is None:
                raise AttributeError("First-order solution has no Mercier diagnostics.")
            return getattr(mercier, name)
        if name in self._FIELD_JET_NAMES:
            raise AttributeError("First-order solution has no second-derivative field jet.")
        if name in self._SINGULARITY_NAMES:
            raise AttributeError("First-order solution has no singular-radius diagnostics.")
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}.")

    def _require_mercier(self) -> MercierDiagnostics:
        mercier = object.__getattribute__(self, "mercier")
        if mercier is None:
            raise AttributeError("First-order solution has no Mercier diagnostics.")
        return mercier

    def _require_field_jet(self) -> FieldJet:
        field_jet = object.__getattribute__(self, "field_jet")
        if field_jet is None:
            raise AttributeError("First-order solution has no second-derivative field jet.")
        return field_jet

    def _require_singularity(self) -> SingularityDiagnostics:
        singularity = object.__getattribute__(self, "singularity")
        if singularity is None:
            raise AttributeError("First-order solution has no singular-radius diagnostics.")
        return singularity

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

    @property
    def d2_volume_d_psi2(self) -> jax.Array:
        return self._require_mercier().d2_volume_d_psi2

    @property
    def DGeod_times_r2(self) -> jax.Array:
        return self._require_mercier().DGeod_times_r2

    @property
    def DWell_times_r2(self) -> jax.Array:
        return self._require_mercier().DWell_times_r2

    @property
    def DMerc_times_r2(self) -> jax.Array:
        return self._require_mercier().DMerc_times_r2

    @property
    def grad_grad_B_axis(self) -> jax.Array:
        """Cartesian Hessian in ``(sample, field, derivative, derivative)`` order."""

        return self._require_field_jet().hessian

    @property
    def grad_grad_B(self) -> jax.Array:
        """pyQSC-compatible Frenet Hessian in ``(sample, d, d, field)`` order."""

        return self._require_field_jet().hessian_frenet

    @property
    def L_grad_grad_B(self) -> jax.Array:
        return self._require_field_jet().L_grad_grad_B

    @property
    def grad_grad_B_inverse_scale_length_vs_varphi(self) -> jax.Array:
        return self._require_field_jet().grad_grad_B_inverse_scale_length_vs_varphi

    @property
    def grad_grad_B_inverse_scale_length(self) -> jax.Array:
        return self._require_field_jet().grad_grad_B_inverse_scale_length

    @property
    def r_singularity(self) -> jax.Array:
        return self._require_singularity().r_singularity

    @property
    def r_singularity_vs_varphi(self) -> jax.Array:
        return self._require_singularity().r_singularity_vs_varphi

    @property
    def inv_r_singularity_vs_varphi(self) -> jax.Array:
        return self._require_singularity().inv_r_singularity_vs_varphi

    @property
    def r_singularity_basic_vs_varphi(self) -> jax.Array:
        return self._require_singularity().r_singularity_vs_varphi

    @property
    def r_singularity_theta_vs_varphi(self) -> jax.Array:
        return self._require_singularity().theta_singularity_vs_varphi

    @property
    def r_singularity_residual_sqnorm(self) -> jax.Array:
        return self._require_singularity().residual_norm_vs_varphi ** 2

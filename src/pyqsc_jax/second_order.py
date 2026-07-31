"""Complete second-order quasisymmetric near-axis coefficient solve."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from pyqsc_jax.models import NearAxisSolution, SecondOrderData
from pyqsc_jax.solvers import implicit_dense_linear_solve

MU0 = 4e-7 * jnp.pi


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SecondOrderResiduals:
    """Independent residuals of the four coupled second-order equations."""

    force_balance_1: jax.Array
    force_balance_2: jax.Array
    area_constraint_1: jax.Array
    area_constraint_2: jax.Array

    @property
    def maximum_absolute(self) -> jax.Array:
        """Maximum absolute collocation residual across all four equations."""

        return jnp.max(
            jnp.abs(
                jnp.stack(
                    (
                        self.force_balance_1,
                        self.force_balance_2,
                        self.area_constraint_1,
                        self.area_constraint_2,
                    )
                )
            )
        )


def _assemble_periodic_system(
    first_order: NearAxisSolution,
    *,
    X2s: jax.Array,
    X2c: jax.Array,
    Z20: jax.Array,
    Z2s: jax.Array,
    Z2c: jax.Array,
    beta_1s: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Assemble the coupled periodic system for ``X20`` and ``Y20``."""

    inputs = first_order.inputs
    geometry = first_order.geometry
    D = geometry.d_d_varphi
    lp = geometry.abs_G0_over_B0
    X1c = first_order.X1c
    Y1s = first_order.Y1s
    Y1c = first_order.Y1c
    curvature = first_order.curvature
    torsion = first_order.torsion
    iotaN = first_order.iotaN
    sign_product = inputs.sG * inputs.spsi
    current_factor = inputs.spsi * inputs.I2 / inputs.B0

    Y2s_from_X20 = -sign_product * curvature**2 / inputs.etabar**2
    Y2s_inhomogeneous = sign_product * (
        -curvature / 2 + curvature**2 / inputs.etabar**2 * (-X2c + X2s * first_order.sigma)
    )
    Y2c_from_X20 = -sign_product * curvature**2 * first_order.sigma / inputs.etabar**2
    Y2c_inhomogeneous = (
        sign_product * curvature**2 / inputs.etabar**2 * (X2s + X2c * first_order.sigma)
    )

    fX0_from_X20 = -4 * sign_product * lp * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    fX0_from_Y20 = -torsion * lp - 4 * sign_product * lp * Z2s + 2 * current_factor * lp
    fX0_inhomogeneous = (
        curvature * lp * Z20
        - 4 * sign_product * lp * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c)
        - current_factor * (curvature * sign_product / 2) * lp
        + beta_1s * lp * Y1c / 2
    )

    fXs_from_X20 = (
        -torsion * lp * Y2s_from_X20
        - 4 * sign_product * lp * Y2c_from_X20 * Z20
        + 2 * current_factor * lp * Y2s_from_X20
    )
    fXs_from_Y20 = -4 * sign_product * lp * (-Z2c + Z20)
    fXs_inhomogeneous = (
        D @ X2s
        - 2 * iotaN * X2c
        - torsion * lp * Y2s_inhomogeneous
        + curvature * lp * Z2s
        - 4 * sign_product * lp * Y2c_inhomogeneous * Z20
        - current_factor * (curvature * sign_product / 2 - 2 * Y2s_inhomogeneous) * lp
        - lp * beta_1s * Y1s / 2
    )

    fXc_from_X20 = (
        -torsion * lp * Y2c_from_X20
        + 4 * sign_product * lp * Y2s_from_X20 * Z20
        + 2 * current_factor * lp * Y2c_from_X20
    )
    fXc_from_Y20 = -torsion * lp - 4 * sign_product * lp * Z2s + 2 * current_factor * lp
    fXc_inhomogeneous = (
        D @ X2c
        + 2 * iotaN * X2s
        - torsion * lp * Y2c_inhomogeneous
        + curvature * lp * Z2c
        + 4 * sign_product * lp * Y2s_inhomogeneous * Z20
        - current_factor * (curvature * sign_product / 2 - 2 * Y2c_inhomogeneous) * lp
        - lp * beta_1s * Y1c / 2
    )

    fY0_from_X20 = torsion * lp - 2 * current_factor * lp
    fY0_from_Y20 = jnp.zeros_like(curvature)
    fY0_inhomogeneous = (
        -4 * sign_product * lp * (X2s * Z2c - X2c * Z2s)
        + current_factor * curvature * X1c**2 * lp / 2
        - lp * beta_1s * X1c / 2
    )

    fYs_from_X20 = -2 * iotaN * Y2c_from_X20 - 4 * sign_product * lp * Z2c
    fYs_from_Y20 = jnp.full_like(curvature, -2 * iotaN)
    fYs_inhomogeneous = (
        D @ Y2s_inhomogeneous
        - 2 * iotaN * Y2c_inhomogeneous
        + torsion * lp * X2s
        + 4 * sign_product * lp * X2c * Z20
        - 2 * current_factor * X2s * lp
    )

    fYc_from_X20 = 2 * iotaN * Y2s_from_X20 + 4 * sign_product * lp * Z2s
    fYc_from_Y20 = jnp.zeros_like(curvature)
    fYc_inhomogeneous = (
        D @ Y2c_inhomogeneous
        + 2 * iotaN * Y2s_inhomogeneous
        + torsion * lp * X2c
        - 4 * sign_product * lp * X2s * Z20
        - current_factor * (-curvature * X1c**2 / 2 + 2 * X2c) * lp
        + lp * beta_1s * X1c / 2
    )

    block_00 = Y1c[:, None] * D * Y2s_from_X20[None, :] - Y1s[:, None] * D * Y2c_from_X20[None, :]
    block_01 = -2 * Y1s[:, None] * D
    block_10 = (
        -X1c[:, None] * D
        + Y1s[:, None] * D * Y2s_from_X20[None, :]
        + Y1c[:, None] * D * Y2c_from_X20[None, :]
    )
    block_11 = jnp.zeros_like(D)

    block_00 = block_00 + jnp.diag(
        X1c * fXs_from_X20 - Y1s * fY0_from_X20 + Y1c * fYs_from_X20 - Y1s * fYc_from_X20
    )
    block_01 = block_01 + jnp.diag(
        X1c * fXs_from_Y20 - Y1s * fY0_from_Y20 + Y1c * fYs_from_Y20 - Y1s * fYc_from_Y20
    )
    block_10 = block_10 + jnp.diag(
        -X1c * fX0_from_X20
        + X1c * fXc_from_X20
        - Y1c * fY0_from_X20
        + Y1s * fYs_from_X20
        + Y1c * fYc_from_X20
    )
    block_11 = block_11 + jnp.diag(
        -X1c * fX0_from_Y20
        + X1c * fXc_from_Y20
        - Y1c * fY0_from_Y20
        + Y1s * fYs_from_Y20
        + Y1c * fYc_from_Y20
    )
    matrix = jnp.concatenate(
        (
            jnp.concatenate((block_00, block_01), axis=1),
            jnp.concatenate((block_10, block_11), axis=1),
        ),
        axis=0,
    )

    right_hand_side_1 = -(
        X1c * fXs_inhomogeneous
        - Y1s * fY0_inhomogeneous
        + Y1c * fYs_inhomogeneous
        - Y1s * fYc_inhomogeneous
    )
    right_hand_side_2 = -(
        -X1c * fX0_inhomogeneous
        + X1c * fXc_inhomogeneous
        - Y1c * fY0_inhomogeneous
        + Y1s * fYs_inhomogeneous
        + Y1c * fYc_inhomogeneous
    )
    right_hand_side = jnp.concatenate((right_hand_side_1, right_hand_side_2))
    return (
        matrix,
        right_hand_side,
        Y2s_from_X20,
        Y2s_inhomogeneous,
        Y2c_from_X20,
        Y2c_inhomogeneous,
    )


def solve_second_order(
    first_order: NearAxisSolution,
    *,
    attach_diagnostics: bool = True,
) -> NearAxisSolution:
    """Add the complete finite-pressure/current second-order solution."""

    inputs = first_order.inputs
    geometry = first_order.geometry
    D = geometry.d_d_varphi
    B0_over_abs_G0 = 1 / geometry.abs_G0_over_B0
    lp = geometry.abs_G0_over_B0
    X1c = first_order.X1c
    Y1s = first_order.Y1s
    Y1c = first_order.Y1c
    curvature = first_order.curvature
    torsion = first_order.torsion
    iotaN = first_order.iotaN
    sign_product = inputs.sG * inputs.spsi

    V1 = X1c**2 + Y1c**2 + Y1s**2
    V2 = 2 * Y1s * Y1c
    V3 = X1c**2 + Y1c**2 - Y1s**2
    factor = -B0_over_abs_G0 / 8
    Z20 = factor * (D @ V1)
    Z2s = factor * (D @ V2 - 2 * iotaN * V3)
    Z2c = factor * (D @ V3 + 2 * iotaN * V2)

    qs = -iotaN * X1c - Y1s * torsion * lp
    qc = D @ X1c - Y1c * torsion * lp
    rs = D @ Y1s - iotaN * Y1c
    rc = D @ Y1c + iotaN * Y1s + X1c * torsion * lp
    X2s = (
        B0_over_abs_G0
        * (
            D @ Z2s
            - 2 * iotaN * Z2c
            + B0_over_abs_G0 * (lp**2 * inputs.B2s / inputs.B0 + (qc * qs + rc * rs) / 2)
        )
        / curvature
    )
    X2c = (
        B0_over_abs_G0
        * (
            D @ Z2c
            + 2 * iotaN * Z2s
            - B0_over_abs_G0
            * (
                -(lp**2) * inputs.B2c / inputs.B0
                + lp**2 * inputs.etabar**2 / 2
                - (qc**2 - qs**2 + rc**2 - rs**2) / 4
            )
        )
        / curvature
    )
    beta_1s = -4 * sign_product * MU0 * inputs.p2 * inputs.etabar * lp / (iotaN * inputs.B0**2)

    (
        matrix,
        right_hand_side,
        Y2s_from_X20,
        Y2s_inhomogeneous,
        Y2c_from_X20,
        Y2c_inhomogeneous,
    ) = _assemble_periodic_system(
        first_order,
        X2s=X2s,
        X2c=X2c,
        Z20=Z20,
        Z2s=Z2s,
        Z2c=Z2c,
        beta_1s=beta_1s,
    )
    solution, linear_report = implicit_dense_linear_solve(matrix, right_hand_side)
    X20, Y20 = jnp.split(solution, 2)
    Y2s = Y2s_inhomogeneous + Y2s_from_X20 * X20
    Y2c = Y2c_inhomogeneous + Y2c_from_X20 * X20 + Y20

    B20 = inputs.B0 * (
        curvature * X20
        - B0_over_abs_G0 * (D @ Z20)
        + inputs.etabar**2 / 2
        - MU0 * inputs.p2 / inputs.B0**2
        - B0_over_abs_G0**2 * (qc**2 + qs**2 + rc**2 + rs**2) / 4
    )
    weights = geometry.d_l_d_phi
    B20_mean = jnp.sum(B20 * weights) / jnp.sum(weights)
    B20_anomaly = B20 - B20_mean
    B20_residual = jnp.sqrt(jnp.sum(B20_anomaly**2 * weights) / jnp.sum(weights)) / inputs.B0
    B20_variation = jnp.max(B20) - jnp.min(B20)
    G2 = -MU0 * inputs.p2 * first_order.G0 / inputs.B0**2 - first_order.iota * inputs.I2
    N_helicity = -first_order.helicity * inputs.axis.nfp

    d_X1c_d_varphi = D @ X1c
    d_Y1c_d_varphi = D @ Y1c
    d_Y1s_d_varphi = D @ Y1s
    untwisting_angle = -first_order.helicity * inputs.axis.nfp * first_order.varphi
    sine = jnp.sin(2 * untwisting_angle)
    cosine = jnp.cos(2 * untwisting_angle)

    def untwist(sine_coefficient, cosine_coefficient):
        return (
            sine_coefficient * cosine + cosine_coefficient * sine,
            -sine_coefficient * sine + cosine_coefficient * cosine,
        )

    X2s_untwisted, X2c_untwisted = untwist(X2s, X2c)
    Y2s_untwisted, Y2c_untwisted = untwist(Y2s, Y2c)
    Z2s_untwisted, Z2c_untwisted = untwist(Z2s, Z2c)
    second_order = SecondOrderData(
        linear_report=linear_report,
        V1=V1,
        V2=V2,
        V3=V3,
        X20=X20,
        X2s=X2s,
        X2c=X2c,
        Y20=Y20,
        Y2s=Y2s,
        Y2c=Y2c,
        Z20=Z20,
        Z2s=Z2s,
        Z2c=Z2c,
        beta_1s=beta_1s,
        B20=B20,
        B20_mean=B20_mean,
        B20_anomaly=B20_anomaly,
        B20_residual=B20_residual,
        B20_variation=B20_variation,
        G2=G2,
        N_helicity=N_helicity,
        d_curvature_d_varphi=D @ curvature,
        d_torsion_d_varphi=D @ torsion,
        d_X20_d_varphi=D @ X20,
        d_X2s_d_varphi=D @ X2s,
        d_X2c_d_varphi=D @ X2c,
        d_Y20_d_varphi=D @ Y20,
        d_Y2s_d_varphi=D @ Y2s,
        d_Y2c_d_varphi=D @ Y2c,
        d_Z20_d_varphi=D @ Z20,
        d_Z2s_d_varphi=D @ Z2s,
        d_Z2c_d_varphi=D @ Z2c,
        d2_X1c_d_varphi2=D @ d_X1c_d_varphi,
        d2_Y1c_d_varphi2=D @ d_Y1c_d_varphi,
        d2_Y1s_d_varphi2=D @ d_Y1s_d_varphi,
        X20_untwisted=X20,
        X2s_untwisted=X2s_untwisted,
        X2c_untwisted=X2c_untwisted,
        Y20_untwisted=Y20,
        Y2s_untwisted=Y2s_untwisted,
        Y2c_untwisted=Y2c_untwisted,
        Z20_untwisted=Z20,
        Z2s_untwisted=Z2s_untwisted,
        Z2c_untwisted=Z2c_untwisted,
    )
    solution = replace(first_order, second_order=second_order)
    if not attach_diagnostics:
        return solution
    from pyqsc_jax.diagnostics import mercier_diagnostics
    from pyqsc_jax.field import total_field_jet
    from pyqsc_jax.singularity import singularity_diagnostics

    return replace(
        solution,
        mercier=mercier_diagnostics(solution),
        field_jet=total_field_jet(solution),
        singularity=singularity_diagnostics(solution),
    )


def second_order_residuals(solution: NearAxisSolution) -> SecondOrderResiduals:
    """Evaluate all four r2 equations independently of matrix assembly."""

    r2 = solution.second_order
    if r2 is None:
        raise ValueError("A second-order solution is required.")
    inputs = solution.inputs
    D = solution.geometry.d_d_varphi
    lp = solution.geometry.abs_G0_over_B0
    sign_product = inputs.sG * inputs.spsi
    current_factor = inputs.spsi * inputs.I2 / inputs.B0

    fX0 = (
        D @ r2.X20
        - solution.torsion * lp * r2.Y20
        + solution.curvature * lp * r2.Z20
        - 4 * sign_product * lp * (r2.Y2c * r2.Z2s - r2.Y2s * r2.Z2c)
        - current_factor * (solution.curvature * solution.X1c * solution.Y1c / 2 - 2 * r2.Y20) * lp
        + lp * r2.beta_1s * solution.Y1c / 2
    )
    fXs = (
        D @ r2.X2s
        - 2 * solution.iotaN * r2.X2c
        - solution.torsion * lp * r2.Y2s
        + solution.curvature * lp * r2.Z2s
        - 4 * sign_product * lp * (-r2.Y20 * r2.Z2c + r2.Y2c * r2.Z20)
        - current_factor * (solution.curvature * solution.X1c * solution.Y1s / 2 - 2 * r2.Y2s) * lp
        - lp * r2.beta_1s * solution.Y1s / 2
    )
    fXc = (
        D @ r2.X2c
        + 2 * solution.iotaN * r2.X2s
        - solution.torsion * lp * r2.Y2c
        + solution.curvature * lp * r2.Z2c
        - 4 * sign_product * lp * (r2.Y20 * r2.Z2s - r2.Y2s * r2.Z20)
        - current_factor * (solution.curvature * solution.X1c * solution.Y1c / 2 - 2 * r2.Y2c) * lp
        - lp * r2.beta_1s * solution.Y1c / 2
    )
    fY0 = (
        D @ r2.Y20
        + solution.torsion * lp * r2.X20
        - 4 * sign_product * lp * (r2.X2s * r2.Z2c - r2.X2c * r2.Z2s)
        - current_factor * (-solution.curvature * solution.X1c**2 / 2 + 2 * r2.X20) * lp
        - lp * r2.beta_1s * solution.X1c / 2
    )
    fYs = (
        D @ r2.Y2s
        - 2 * solution.iotaN * r2.Y2c
        + solution.torsion * lp * r2.X2s
        - 4 * sign_product * lp * (r2.X20 * r2.Z2c - r2.X2c * r2.Z20)
        - 2 * current_factor * r2.X2s * lp
    )
    fYc = (
        D @ r2.Y2c
        + 2 * solution.iotaN * r2.Y2s
        + solution.torsion * lp * r2.X2c
        - 4 * sign_product * lp * (r2.X2s * r2.Z20 - r2.X20 * r2.Z2s)
        - current_factor * (-solution.curvature * solution.X1c**2 / 2 + 2 * r2.X2c) * lp
        + lp * r2.beta_1s * solution.X1c / 2
    )
    force_balance_1 = (
        solution.X1c * fXs - solution.Y1s * fY0 + solution.Y1c * fYs - solution.Y1s * fYc
    )
    force_balance_2 = (
        -solution.X1c * fX0
        + solution.X1c * fXc
        - solution.Y1c * fY0
        + solution.Y1s * fYs
        + solution.Y1c * fYc
    )
    area_constraint_1 = (
        -solution.X1c * r2.Y2c
        + solution.X1c * r2.Y20
        + r2.X2s * solution.Y1s
        + r2.X2c * solution.Y1c
        - r2.X20 * solution.Y1c
    )
    area_constraint_2 = (
        solution.X1c * r2.Y2s
        + r2.X2c * solution.Y1s
        - r2.X2s * solution.Y1c
        + r2.X20 * solution.Y1s
        + sign_product * solution.X1c * solution.curvature / 2
    )
    return SecondOrderResiduals(
        force_balance_1=force_balance_1,
        force_balance_2=force_balance_2,
        area_constraint_1=area_constraint_1,
        area_constraint_2=area_constraint_2,
    )

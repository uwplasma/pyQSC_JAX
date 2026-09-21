"""Third-order flux-constraint corrections.

The concise relation is from Landreman & Sengupta (2019) and is cross-checked
against ``landreman/pyQSC:qsc/calculate_r3.py`` at the audited BSD-2-Clause
upstream commit recorded in the refactor baseline.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from pyqsc_jax.models import NearAxisSolution, ThirdOrderData
from pyqsc_jax.second_order import MU0


def _flux_constraint(solution: NearAxisSolution) -> jax.Array:
    """The flux-constraint coefficient from the parent O(r**3) equations.

    This is the primary route of ``landreman/pyQSC:qsc/calculate_r3.py``, transcribed from it
    verbatim. It shares no algebra with the two shortened forms that
    :func:`solve_third_order` compares it against, so their agreement tests the resolution
    of the second-order solution rather than repeating one expression.
    """

    r2, inputs, geometry = solution.second_order, solution.inputs, solution.geometry
    B0, G0, I2, iotaN = inputs.B0, solution.G0, inputs.I2, solution.iotaN
    B1c, B20 = inputs.etabar * B0, r2.B20
    X1c, Y1c, Y1s = solution.X1c, solution.Y1c, solution.Y1s
    X20, X2c, X2s, Y20, Y2c, Y2s = r2.X20, r2.X2c, r2.X2s, r2.Y20, r2.Y2c, r2.Y2s
    Z20, Z2c, Z2s = r2.Z20, r2.Z2c, r2.Z2s
    torsion, abs_G0_over_B0 = geometry.torsion, geometry.abs_G0_over_B0
    d_X1c_d_varphi = geometry.d_d_varphi @ X1c
    d_Y1c_d_varphi = geometry.d_d_varphi @ Y1c
    return (
        -4 * B0**2 * G0 * X20**2 * Y1c**2
        + 8 * B0**2 * G0 * X20 * X2c * Y1c**2
        - 4 * B0**2 * G0 * X2c**2 * Y1c**2
        - 4 * B0**2 * G0 * X2s**2 * Y1c**2
        + 8 * B0 * G0 * B1c * X1c * X2s * Y1c * Y1s
        + 16 * B0**2 * G0 * X20 * X2s * Y1c * Y1s
        + 2 * B0**2 * I2 * iotaN * X1c**2 * Y1s**2
        - G0 * B1c**2 * X1c**2 * Y1s**2
        - 4 * B0 * G0 * B20 * X1c**2 * Y1s**2
        - 8 * B0 * G0 * B1c * X1c * X20 * Y1s**2
        - 4 * B0**2 * G0 * X20**2 * Y1s**2
        - 8 * B0 * G0 * B1c * X1c * X2c * Y1s**2
        - 8 * B0**2 * G0 * X20 * X2c * Y1s**2
        - 4 * B0**2 * G0 * X2c**2 * Y1s**2
        - 4 * B0**2 * G0 * X2s**2 * Y1s**2
        + 8 * B0**2 * G0 * X1c * X20 * Y1c * Y20
        - 8 * B0**2 * G0 * X1c * X2c * Y1c * Y20
        - 8 * B0**2 * G0 * X1c * X2s * Y1s * Y20
        - 4 * B0**2 * G0 * X1c**2 * Y20**2
        - 8 * B0**2 * G0 * X1c * X20 * Y1c * Y2c
        + 8 * B0**2 * G0 * X1c * X2c * Y1c * Y2c
        + 24 * B0**2 * G0 * X1c * X2s * Y1s * Y2c
        + 8 * B0**2 * G0 * X1c**2 * Y20 * Y2c
        - 4 * B0**2 * G0 * X1c**2 * Y2c**2
        + 8 * B0**2 * G0 * X1c * X2s * Y1c * Y2s
        - 8 * B0 * G0 * B1c * X1c**2 * Y1s * Y2s
        - 8 * B0**2 * G0 * X1c * X20 * Y1s * Y2s
        - 24 * B0**2 * G0 * X1c * X2c * Y1s * Y2s
        - 4 * B0**2 * G0 * X1c**2 * Y2s**2
        - 4 * B0**2 * G0 * X1c**2 * Z20**2
        - 4 * B0**2 * G0 * Y1c**2 * Z20**2
        - 4 * B0**2 * G0 * Y1s**2 * Z20**2
        - 4 * B0**2 * abs_G0_over_B0 * I2 * Y1c * Y1s * Z2c
        + 8 * B0**2 * G0 * X1c**2 * Z20 * Z2c
        + 8 * B0**2 * G0 * Y1c**2 * Z20 * Z2c
        - 8 * B0**2 * G0 * Y1s**2 * Z20 * Z2c
        - 4 * B0**2 * G0 * X1c**2 * Z2c**2
        - 4 * B0**2 * G0 * Y1c**2 * Z2c**2
        - 4 * B0**2 * G0 * Y1s**2 * Z2c**2
        + 2 * B0**2 * abs_G0_over_B0 * I2 * X1c**2 * Z2s
        + 2 * B0**2 * abs_G0_over_B0 * I2 * Y1c**2 * Z2s
        - 2 * B0**2 * abs_G0_over_B0 * I2 * Y1s**2 * Z2s
        + 16 * B0**2 * G0 * Y1c * Y1s * Z20 * Z2s
        - 4 * B0**2 * G0 * X1c**2 * Z2s**2
        - 4 * B0**2 * G0 * Y1c**2 * Z2s**2
        - 4 * B0**2 * G0 * Y1s**2 * Z2s**2
        + B0**2 * abs_G0_over_B0 * I2 * X1c**3 * Y1s * torsion
        + B0**2 * abs_G0_over_B0 * I2 * X1c * Y1c**2 * Y1s * torsion
        + B0**2 * abs_G0_over_B0 * I2 * X1c * Y1s**3 * torsion
        - B0**2 * I2 * X1c * Y1c * Y1s * d_X1c_d_varphi
        + B0**2 * I2 * X1c**2 * Y1s * d_Y1c_d_varphi
    ) / (16 * B0**2 * G0 * X1c**2 * Y1s**2)


def solve_third_order(solution: NearAxisSolution) -> NearAxisSolution:
    """Add the r3 corrections required for consistency through second order."""

    r2 = solution.second_order
    if r2 is None:
        raise ValueError("A second-order solution is required before the third-order correction.")
    inputs = solution.inputs
    length_scale = solution.geometry.abs_G0_over_B0
    D = solution.geometry.d_d_varphi
    d_X1c = D @ solution.X1c
    d_Y1c = D @ solution.Y1c
    first_order_norm = solution.X1c**2 + solution.Y1c**2 + solution.Y1s**2
    Q = (
        -inputs.spsi
        * inputs.B0
        * length_scale
        / (2 * solution.G0**2)
        * (solution.iotaN * inputs.I2 + MU0 * inputs.p2 * solution.G0 / inputs.B0**2)
        + 2 * (r2.X2c * r2.Y2s - r2.X2s * r2.Y2c)
        + inputs.spsi
        * inputs.B0
        / (2 * solution.G0)
        * (length_scale * r2.X20 * solution.curvature - r2.d_Z20_d_varphi)
        + inputs.I2
        / (4 * solution.G0)
        * (
            -length_scale * solution.torsion * first_order_norm
            + solution.Y1c * d_X1c
            - solution.X1c * d_Y1c
        )
    )
    sign_product = inputs.sG * inputs.spsi
    coefficient = _flux_constraint(solution)
    N_helicity = solution.iota - solution.iotaN
    B0_correction = (
        -inputs.sG
        * inputs.B0**2
        * (r2.G2 + inputs.I2 * N_helicity)
        * length_scale
        / (2 * solution.G0**2)
        - inputs.sG * inputs.spsi * inputs.B0 * 2 * (r2.X2c * r2.Y2s - r2.X2s * r2.Y2c)
        - inputs.sG
        * inputs.B0**2
        / (2 * solution.G0)
        * (length_scale * r2.X20 * solution.curvature - r2.d_Z20_d_varphi)
        - inputs.sG
        * inputs.spsi
        * inputs.B0
        * inputs.I2
        / (4 * solution.G0)
        * (
            -length_scale * solution.torsion * first_order_norm
            + solution.Y1c * d_X1c
            - solution.X1c * d_Y1c
        )
    )

    zeros = jnp.zeros_like(solution.X1c)
    X3s1 = zeros
    X3c1 = solution.X1c * coefficient
    Y3s1 = solution.Y1s * coefficient
    Y3c1 = solution.Y1c * coefficient
    angle = -solution.helicity * inputs.axis.nfp * solution.varphi

    def untwist(sine_coefficient, cosine_coefficient, harmonic):
        sine = jnp.sin(harmonic * angle)
        cosine = jnp.cos(harmonic * angle)
        return (
            sine_coefficient * cosine + cosine_coefficient * sine,
            -sine_coefficient * sine + cosine_coefficient * cosine,
        )

    X3s1_untwisted, X3c1_untwisted = untwist(X3s1, X3c1, 1)
    Y3s1_untwisted, Y3c1_untwisted = untwist(Y3s1, Y3c1, 1)
    Z3s1_untwisted, Z3c1_untwisted = untwist(zeros, zeros, 1)
    X3s3_untwisted, X3c3_untwisted = untwist(zeros, zeros, 3)
    Y3s3_untwisted, Y3c3_untwisted = untwist(zeros, zeros, 3)
    Z3s3_untwisted, Z3c3_untwisted = untwist(zeros, zeros, 3)
    third_order = ThirdOrderData(
        flux_constraint_coefficient=coefficient,
        B0_order_a_squared_to_cancel=B0_correction,
        # The two checks pyQSC warns on: both vanish only as the second-order solve resolves.
        flux_constraint_residual=jnp.max(jnp.abs(coefficient + Q / (2 * sign_product))),
        consistency_error=jnp.max(jnp.abs(coefficient - B0_correction / (2 * inputs.B0))),
        X3s1=X3s1,
        X3c1=X3c1,
        Y3s1=Y3s1,
        Y3c1=Y3c1,
        Z3s1=zeros,
        Z3c1=zeros,
        X3s3=zeros,
        X3c3=zeros,
        Y3s3=zeros,
        Y3c3=zeros,
        Z3s3=zeros,
        Z3c3=zeros,
        d_X3c1_d_varphi=D @ X3c1,
        d_Y3s1_d_varphi=D @ Y3s1,
        d_Y3c1_d_varphi=D @ Y3c1,
        X3s1_untwisted=X3s1_untwisted,
        X3c1_untwisted=X3c1_untwisted,
        Y3s1_untwisted=Y3s1_untwisted,
        Y3c1_untwisted=Y3c1_untwisted,
        Z3s1_untwisted=Z3s1_untwisted,
        Z3c1_untwisted=Z3c1_untwisted,
        X3s3_untwisted=X3s3_untwisted,
        X3c3_untwisted=X3c3_untwisted,
        Y3s3_untwisted=Y3s3_untwisted,
        Y3c3_untwisted=Y3c3_untwisted,
        Z3s3_untwisted=Z3s3_untwisted,
        Z3c3_untwisted=Z3c3_untwisted,
    )
    return replace(solution, third_order=third_order)

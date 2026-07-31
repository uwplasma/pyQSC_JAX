"""Third-order flux-constraint corrections.

The concise relation is from Landreman & Sengupta (2019) and is cross-checked
against ``landreman/pyQSC:qsc/calculate_r3.py`` at the audited BSD-2-Clause
upstream commit recorded in the refactor baseline.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from pyqsc_jax.models import NearAxisSolution, ThirdOrderData
from pyqsc_jax.second_order import MU0


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
    coefficient = -Q / (2 * sign_product)
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
        flux_constraint_residual=jnp.max(jnp.abs(Q + 2 * sign_product * coefficient)),
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

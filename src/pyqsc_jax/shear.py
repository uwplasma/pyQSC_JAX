"""Magnetic shear from the order-r-cubed generalized sigma equation.

The equations are adapted from ``landreman/pyQSC:qsc/calculate_r3.py`` at the
audited BSD-2-Clause upstream commit recorded in the refactor baseline. Their
source derivation is Rodríguez et al., Physics of Plasmas 29, 012507 (2022).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp

from pyqsc_jax.models import NearAxisSolution, ShearData

ArrayLike = Any


def solve_magnetic_shear(
    solution: NearAxisSolution,
    *,
    B31c: ArrayLike = 0.0,
) -> NearAxisSolution:
    """Attach the standard-MHS order-r-squared transform correction.

    ``B31c`` uses the inverse-field-squared convention of the generalized
    sigma equation. Current variation at this order and ``B31s`` are zero.
    """

    r2 = solution.second_order
    if r2 is None:
        raise ValueError("A second-order solution is required to calculate magnetic shear.")
    inputs = solution.inputs
    if inputs.sG != 1 or inputs.spsi != 1:
        raise NotImplementedError(
            "Magnetic shear currently requires sG=spsi=1; the upstream signs are not generalized."
        )

    B31c = jnp.asarray(B31c)
    if B31c.ndim != 0:
        raise ValueError("B31c must be a scalar.")

    D = solution.geometry.d_d_varphi
    epsilon_scale = jnp.sqrt(2 / inputs.B0)
    scale2 = epsilon_scale**2
    G2 = r2.G2 * scale2
    G0 = solution.G0
    I2 = inputs.I2 * scale2
    X1c = solution.X1c * epsilon_scale
    Y1c = solution.Y1c * epsilon_scale
    Y1s = solution.Y1s * epsilon_scale
    X20 = r2.X20 * scale2
    X2s = r2.X2s * scale2
    X2c = r2.X2c * scale2
    Y20 = r2.Y20 * scale2
    Y2s = r2.Y2s * scale2
    Y2c = r2.Y2c * scale2
    Z20 = r2.Z20 * scale2
    Z2s = r2.Z2s * scale2
    Z2c = r2.Z2c * scale2
    torsion = -solution.torsion
    curvature = solution.curvature
    iota = solution.iotaN
    dldp = solution.geometry.abs_G0_over_B0
    dX1c = D @ X1c
    dY1c = D @ Y1c
    dY1s = D @ Y1s
    dZ20 = D @ Z20
    dZ2c = D @ Z2c
    dZ2s = D @ Z2s
    dX20 = D @ X20
    dX2c = D @ X2c
    dX2s = D @ X2s
    dY20 = D @ Y20
    dY2c = D @ Y2c
    dY2s = D @ Y2s

    inverse_B0_squared = 1 / inputs.B0**2
    Ba0 = G0
    Ba1 = G2 + solution.iotaN * I2
    eta = inputs.etabar * jnp.sqrt(2) * inverse_B0_squared**0.25
    B1c = -2 * inverse_B0_squared * eta
    B20 = (
        (0.75 * inputs.etabar**2 / jnp.sqrt(inverse_B0_squared) - r2.B20)
        * 4
        * inverse_B0_squared**2
    )
    B31s = jnp.asarray(0, dtype=B31c.dtype)
    I4 = jnp.asarray(0, dtype=B31c.dtype)

    Z31c = (
        -1
        / (3 * Ba0 * X1c * Y1s)
        * (
            2 * iota * (X1c * X2s - Y2c * Y1s + Y1c * Y2s)
            - 2 * Ba0 * X2s * Y1c * Z20
            + 2 * Ba0 * X2c * Y1s * Z20
            + 2 * Ba0 * X1c * Y2s * Z20
            - 4 * Ba0 * X2s * Y1c * Z2c
            - 2 * Ba0 * X20 * Y1s * Z2c
            + 4 * Ba0 * X1c * Y2s * Z2c
            - dldp
            * (
                torsion * (2 * X20 * Y1c + X2c * Y1c - 2 * X1c * Y20 - X1c * Y2c + X2s * Y1s)
                + I2 * (2 * X20 * Y1c + X2c * Y1c - 2 * X1c * Y20 - X1c * Y2c + X2s * Y1s)
                - 2 * curvature * X1c * Z20
                - curvature * X1c * Z2c
            )
            + 2 * Ba0 * X20 * Y1c * Z2s
            + 4 * Ba0 * X2c * Y1c * Z2s
            - 2 * Ba0 * X1c * Y20 * Z2s
            - 4 * Ba0 * X1c * Y2c * Z2s
            + 2 * X1c * dX20
            + X1c * dX2c
            + 2 * Y1c * dY20
            + Y1c * dY2c
            + Y1s * dY2s
        )
    )
    dZ31c = D @ Z31c

    Z31s = (
        1
        / (3 * Ba0 * X1c * Y1s)
        * (
            2 * iota * (X1c * X2c + Y1c * Y2c + Y1s * Y2s)
            - 2 * Ba0 * X2c * Y1c * Z20
            + 2 * Ba0 * X1c * Y2c * Z20
            - 2 * Ba0 * X2s * Y1s * Z20
            + 2 * Ba0 * X20 * Y1c * Z2c
            - 2 * Ba0 * X1c * Y20 * Z2c
            + 4 * Ba0 * X2s * Y1s * Z2c
            + 2 * Ba0 * X20 * Y1s * Z2s
            - 4 * Ba0 * X2c * Y1s * Z2s
            + dldp
            * (
                I2 * X2s * Y1c
                + 2 * I2 * X20 * Y1s
                - I2 * X2c * Y1s
                - I2 * X1c * Y2s
                + torsion * (X2s * Y1c + 2 * X20 * Y1s - X2c * Y1s - X1c * Y2s)
                - curvature * X1c * Z2s
            )
            - X1c * dX2s
            - 2 * Y1s * dY20
            + Y1s * dY2c
            - Y1c * dY2s
        )
    )
    dZ31s = D @ Z31s

    X31c = (
        1
        / (2 * dldp**2 * curvature)
        * (
            -2 * Ba0 * Ba1 * B1c
            - Ba0**2 * B31c
            + 2 * dldp**2 * torsion**2 * X1c * X20
            + 2 * iota**2 * X1c * X2c
            + dldp**2 * torsion**2 * X1c * X2c
            + dldp**2 * curvature**2 * X1c * (2 * X20 + X2c)
            + 3 * dldp * iota * torsion * X2s * Y1c
            + 2 * dldp**2 * torsion**2 * Y1c * Y20
            + 2 * iota**2 * Y1c * Y2c
            + dldp**2 * torsion**2 * Y1c * Y2c
            - 2 * dldp * iota * torsion * X20 * Y1s
            - 3 * dldp * iota * torsion * X2c * Y1s
            - 3 * dldp * iota * torsion * X1c * Y2s
            + 2 * iota**2 * Y1s * Y2s
            + dldp**2 * torsion**2 * Y1s * Y2s
            + 2 * dldp * iota * Z31s
            + 2 * iota * X2s * dX1c
            + 2 * dldp * torsion * Y20 * dX1c
            + dldp * torsion * Y2c * dX1c
            + 2 * dldp * torsion * Y1c * dX20
            + 2 * dX1c * dX20
            + dldp * torsion * Y1c * dX2c
            + dX1c * dX2c
            - iota * X1c * dX2s
            + dldp * torsion * Y1s * dX2s
            - 2 * dldp * torsion * X20 * dY1c
            - dldp * torsion * X2c * dY1c
            + 2 * iota * Y2s * dY1c
            - 2 * dldp * torsion * X1c * dY20
            + 2 * iota * Y1s * dY20
            + 2 * dY1c * dY20
            - dldp * torsion * X1c * dY2c
            + iota * Y1s * dY2c
            + dY1c * dY2c
            - dldp * torsion * X2s * dY1s
            - 2 * iota * Y2c * dY1s
            - iota * Y1c * dY2s
            + dY1s * dY2s
            + dldp
            * curvature
            * (
                -3 * iota * X1c * Z2s
                + dldp * torsion * (Y1c * (2 * Z20 + Z2c) + Y1s * Z2s)
                + 2 * Z20 * dX1c
                + Z2c * dX1c
                - 2 * X1c * dZ20
                - X1c * dZ2c
            )
            + 2 * dldp * dZ31c
        )
    )

    X31s = (
        1
        / (2 * dldp**2 * curvature)
        * (
            -(Ba0**2) * B31s
            + dldp**2 * curvature**2 * X1c * X2s
            + dldp**2 * torsion**2 * X1c * X2s
            + 2 * dldp**2 * torsion**2 * Y20 * Y1s
            - dldp**2 * torsion**2 * Y2c * Y1s
            + dldp**2 * torsion**2 * Y1c * Y2s
            + 2 * iota**2 * (X1c * X2s - Y2c * Y1s + Y1c * Y2s)
            + 2 * dldp**2 * curvature * torsion * Y1s * Z20
            - dldp**2 * curvature * torsion * Y1s * Z2c
            + dldp**2 * curvature * torsion * Y1c * Z2s
            + dldp * torsion * Y2s * dX1c
            + dldp * curvature * Z2s * dX1c
            + 2 * dldp * torsion * Y1s * dX20
            - dldp * torsion * Y1s * dX2c
            + dldp * torsion * Y1c * dX2s
            + dX1c * dX2s
            - dldp * torsion * X2s * dY1c
            - 2 * dldp * torsion * X20 * dY1s
            + dldp * torsion * X2c * dY1s
            + 2 * dY20 * dY1s
            - dY2c * dY1s
            - dldp * torsion * X1c * dY2s
            + dY1c * dY2s
            + iota
            * (
                dldp
                * torsion
                * (2 * X20 * Y1c - 3 * X2c * Y1c - 2 * X1c * Y20 + 3 * X1c * Y2c - 3 * X2s * Y1s)
                + dldp * curvature * X1c * (-2 * Z20 + 3 * Z2c)
                - 2 * dldp * Z31c
                - 2 * X2c * dX1c
                - 2 * X1c * dX20
                + X1c * dX2c
                - 2 * Y2c * dY1c
                - 2 * Y1c * dY20
                + Y1c * dY2c
                - 2 * Y2s * dY1s
                + Y1s * dY2s
            )
            - dldp * curvature * X1c * dZ2s
            + 2 * dldp * dZ31s
        )
    )
    dX31s = D @ X31s

    Y31s = (
        1
        / (4 * Ba0 * X1c)
        * (
            -2 * Ba1 * X1c * Y1s
            + 2 * iota * I2 * X1c * Y1s
            - dldp * (4 * curvature * X20 + torsion * I2 * (X1c**2 + Y1c**2 + Y1s**2))
            + 4 * Ba0 * (X31s * Y1c + 2 * X2s * Y2c - X31c * Y1s - 2 * X2c * Y2s)
            - I2 * Y1c * dX1c
            + I2 * X1c * dY1c
            + 4 * dZ20
        )
    )
    dY31s = D @ Y31s

    Lambda_tilde = 2 / Y1s**2 * (
        Ba0 * inverse_B0_squared * I4 + (Ba1 * inverse_B0_squared + Ba0 * B20) * I2
    ) + 1 / Y1s**2 * (
        -2
        * iota
        * (
            2 * X2c**2
            + X1c * X31c
            + 2 * X2s**2
            + 2 * Y2c**2
            + 2 * Y2s**2
            + Y1s * Y31s
            + 2 * Z2c**2
            + 2 * Z2s**2
        )
        + 2
        * dldp
        * (
            torsion * (-X31s * Y1c - 2 * X2s * Y2c + X31c * Y1s + 2 * X2c * Y2s + X1c * Y31s)
            + curvature * (-2 * X2s * Z2c + 2 * X2c * Z2s + X1c * Z31s)
        )
        - X31s * dX1c
        - 2 * X2s * dX2c
        + 2 * X2c * dX2s
        + X1c * dX31s
        - Y31s * dY1c
        - 2 * Y2s * dY2c
        + 2 * Y2c * dY2s
        + Y1c * dY31s
        - 2 * Z2s * dZ2c
        + 2 * Z2c * dZ2s
    )

    reduced_derivative = D[1:, 1:]
    symmetric_integral = jnp.concatenate(
        (
            jnp.zeros(1, dtype=solution.sigma.dtype),
            jnp.linalg.solve(reduced_derivative, solution.sigma[1:]),
        )
    )
    symmetric_factor = jnp.exp(2 * iota * symmetric_integral)
    denominator_factor = (X1c**2 + Y1c**2 + Y1s**2) / Y1s**2
    symmetric_numerator = jnp.sum(
        symmetric_factor * Lambda_tilde * solution.geometry.d_varphi_d_phi
    )
    symmetric_denominator = jnp.sum(
        symmetric_factor * denominator_factor * solution.geometry.d_varphi_d_phi
    )

    sigma_average = jnp.sum(solution.sigma * solution.geometry.d_varphi_d_phi) / inputs.nphi
    periodic_integral = jnp.linalg.solve(
        reduced_derivative,
        solution.sigma[1:] - sigma_average,
    )
    general_integral = jnp.concatenate(
        (
            jnp.zeros(1, dtype=solution.sigma.dtype),
            periodic_integral + sigma_average * solution.varphi[1:],
        )
    )
    general_factor = jnp.exp(2 * iota * general_integral)
    period = 2 * jnp.pi / inputs.axis.nfp
    factor_extended = jnp.concatenate(
        (general_factor, jnp.exp(jnp.asarray([2 * iota * sigma_average * period])))
    )
    Lambda_extended = jnp.concatenate((Lambda_tilde, Lambda_tilde[:1]))
    denominator_extended = jnp.concatenate((denominator_factor, denominator_factor[:1]))
    varphi_extended = jnp.concatenate((solution.varphi, jnp.asarray([period])))
    general_numerator = jnp.trapezoid(factor_extended * Lambda_extended, varphi_extended)
    general_denominator = jnp.trapezoid(
        factor_extended * denominator_extended,
        varphi_extended,
    )

    stellarator_symmetric = (
        (inputs.sigma0 == 0)
        & (jnp.max(jnp.abs(inputs.axis.rs)) == 0)
        & (jnp.max(jnp.abs(inputs.axis.zc)) == 0)
    )
    numerator = jnp.where(stellarator_symmetric, symmetric_numerator, general_numerator)
    denominator = jnp.where(stellarator_symmetric, symmetric_denominator, general_denominator)
    integrating_factor = jnp.where(stellarator_symmetric, symmetric_factor, general_factor)
    iota2 = inputs.B0 * numerator / (2 * denominator)

    shear = ShearData(
        B31c=B31c,
        iota2=iota2,
        numerator=numerator,
        denominator=denominator,
        Lambda_tilde=Lambda_tilde,
        integrating_factor=integrating_factor,
        sigma_average=sigma_average,
        Z31c=Z31c,
        Z31s=Z31s,
        X31c=X31c,
        X31s=X31s,
        Y31s=Y31s,
        stellarator_symmetric=stellarator_symmetric,
    )
    return replace(solution, shear=shear)

"""Near-axis equilibrium diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from pyqsc_jax.models import MercierDiagnostics, NearAxisSolution
from pyqsc_jax.second_order import MU0


def mercier_diagnostics(solution: NearAxisSolution) -> MercierDiagnostics:
    """Compute the leading magnetic-well and Mercier terms.

    The normalization and signs follow the standard pyQSC implementation.
    """

    r2 = solution.second_order
    if r2 is None:
        raise ValueError("A second-order solution is required for Mercier diagnostics.")

    inputs = solution.inputs
    geometry = solution.geometry
    etabar_squared = inputs.etabar**2
    curvature_squared = solution.curvature**2
    numerator = (
        etabar_squared**2
        + curvature_squared**2 * solution.sigma**2
        + etabar_squared * curvature_squared
    )
    denominator = (
        etabar_squared**2
        + curvature_squared**2 * (1 + solution.sigma**2)
        + 2 * etabar_squared * curvature_squared
    )
    integrand = geometry.d_l_d_phi * numerator / denominator
    weighted_integral = (
        jnp.sum(integrand)
        * (2 * jnp.pi / (inputs.axis.nfp * inputs.nphi))
        * inputs.axis.nfp
        * 2
        * jnp.pi
        / geometry.axis_length
    )
    DGeod_times_r2 = (
        -2
        * MU0**2
        * inputs.p2**2
        * solution.G0**4
        * etabar_squared
        / (jnp.pi**3 * inputs.B0**10 * solution.iotaN**2)
        * weighted_integral
    )
    d2_volume_d_psi2 = (
        4
        * jnp.pi**2
        * jnp.abs(solution.G0)
        / inputs.B0**3
        * (
            3 * etabar_squared
            - 4 * r2.B20_mean / inputs.B0
            + 2 * (r2.G2 + solution.iota * inputs.I2) / solution.G0
        )
    )
    DWell_times_r2 = (
        MU0
        * inputs.p2
        * jnp.abs(solution.G0)
        / (8 * jnp.pi**4 * inputs.B0**3)
        * (d2_volume_d_psi2 - 8 * jnp.pi**2 * MU0 * inputs.p2 * jnp.abs(solution.G0) / inputs.B0**5)
    )
    return MercierDiagnostics(
        d2_volume_d_psi2=d2_volume_d_psi2,
        DGeod_times_r2=DGeod_times_r2,
        DWell_times_r2=DWell_times_r2,
        DMerc_times_r2=DWell_times_r2 + DGeod_times_r2,
    )

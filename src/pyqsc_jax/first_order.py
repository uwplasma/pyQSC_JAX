"""First-order quasisymmetric near-axis construction."""

from typing import Any

import jax
import jax.numpy as jnp

from pyqsc_jax.axis import Axis
from pyqsc_jax.geometry import AxisGeometry, compute_axis_geometry
from pyqsc_jax.models import NearAxisInputs, NearAxisSolution, RootSolveReport
from pyqsc_jax.solvers import DEFAULT_ROOT_OPTIONS, RootSolveOptions, implicit_dense_root

ArrayLike = Any


def sigma_from_state(state: ArrayLike, sigma0: ArrayLike) -> jax.Array:
    """Replace the state-vector iota slot by the prescribed ``sigma(0)``."""

    return jnp.asarray(state).at[0].set(jnp.asarray(sigma0))


def sigma_residual(
    state: ArrayLike,
    *,
    inputs: NearAxisInputs,
    geometry: AxisGeometry,
) -> jax.Array:
    """Periodic first-order sigma-equation residual."""

    state = jnp.asarray(state)
    sigma = sigma_from_state(state, inputs.sigma0)
    iota = state[0]
    helicity = geometry.frame_helicity * inputs.spsi * inputs.sG
    iotaN = iota + helicity * inputs.axis.nfp
    eta_over_curvature_squared = inputs.etabar**2 / geometry.curvature**2
    G0_over_B0 = inputs.sG * geometry.abs_G0_over_B0
    return (
        geometry.d_d_varphi @ sigma
        + iotaN * (eta_over_curvature_squared**2 + 1 + sigma**2)
        - 2
        * eta_over_curvature_squared
        * (-inputs.spsi * geometry.torsion + inputs.I2 / inputs.B0)
        * G0_over_B0
    )


def solve_sigma(
    inputs: NearAxisInputs,
    geometry: AxisGeometry,
    *,
    root_options: RootSolveOptions = DEFAULT_ROOT_OPTIONS,
) -> tuple[jax.Array, jax.Array, RootSolveReport]:
    """Solve for periodic sigma and rotational transform."""

    initial_state = jnp.full((inputs.nphi,), inputs.sigma0)
    initial_state = initial_state.at[0].set(0.0)
    residual_function = lambda state: sigma_residual(  # noqa: E731
        state,
        inputs=inputs,
        geometry=geometry,
    )
    state, report = implicit_dense_root(
        residual_function,
        initial_state,
        options=root_options,
    )
    return sigma_from_state(state, inputs.sigma0), state[0], report


def _assemble_gradient(
    tangent: jax.Array,
    normal: jax.Array,
    binormal: jax.Array,
    *,
    nn: jax.Array,
    bn: jax.Array,
    nb: jax.Array,
    bb: jax.Array,
    tn: jax.Array,
    nt: jax.Array,
    tt: jax.Array,
) -> jax.Array:
    outer = lambda left, right: jnp.einsum("ni,nj->nij", left, right)  # noqa: E731
    return (
        nn[:, None, None] * outer(normal, normal)
        + bn[:, None, None] * outer(binormal, normal)
        + nb[:, None, None] * outer(normal, binormal)
        + bb[:, None, None] * outer(binormal, binormal)
        + tn[:, None, None] * outer(tangent, normal)
        + nt[:, None, None] * outer(normal, tangent)
        + tt[:, None, None] * outer(tangent, tangent)
    )


def first_order_solution(
    inputs: NearAxisInputs,
    geometry: AxisGeometry,
    sigma: jax.Array,
    iota: jax.Array,
    root_report: RootSolveReport,
) -> NearAxisSolution:
    """Assemble first-order shape coefficients, fields, and diagnostics."""

    helicity = geometry.frame_helicity * inputs.spsi * inputs.sG
    iotaN = iota + helicity * inputs.axis.nfp
    G0 = inputs.sG * geometry.abs_G0_over_B0 * inputs.B0
    X1s = jnp.zeros_like(geometry.curvature)
    X1c = inputs.etabar / geometry.curvature
    Y1s = inputs.sG * inputs.spsi * geometry.curvature / inputs.etabar
    Y1c = inputs.sG * inputs.spsi * geometry.curvature * sigma / inputs.etabar

    untwisting_angle = -helicity * inputs.axis.nfp * geometry.varphi
    sine = jnp.sin(untwisting_angle)
    cosine = jnp.cos(untwisting_angle)
    X1s_untwisted = X1s * cosine + X1c * sine
    X1c_untwisted = -X1s * sine + X1c * cosine
    Y1s_untwisted = Y1s * cosine + Y1c * sine
    Y1c_untwisted = -Y1s * sine + Y1c * cosine

    p = X1s**2 + X1c**2 + Y1s**2 + Y1c**2
    q = X1s * Y1c - X1c * Y1s
    elongation = (p + jnp.sqrt(p**2 - 4 * q**2)) / (2 * jnp.abs(q))
    mean_elongation = jnp.sum(elongation * geometry.d_l_d_phi) / jnp.sum(geometry.d_l_d_phi)

    d_X1c_d_varphi = geometry.d_d_varphi @ X1c
    d_Y1s_d_varphi = geometry.d_d_varphi @ Y1s
    d_Y1c_d_varphi = geometry.d_d_varphi @ Y1c
    factor = inputs.spsi * inputs.B0 / geometry.abs_G0_over_B0
    tn = inputs.sG * inputs.B0 * geometry.curvature
    nt = tn
    bb = factor * (X1c * d_Y1s_d_varphi - iotaN * X1c * Y1c)
    nn = factor * (d_X1c_d_varphi * Y1s + iotaN * X1c * Y1c)
    bn = factor * (
        -inputs.sG * inputs.spsi * geometry.abs_G0_over_B0 * geometry.torsion - iotaN * X1c**2
    )
    nb = factor * (
        d_Y1c_d_varphi * Y1s
        - d_Y1s_d_varphi * Y1c
        + inputs.sG * inputs.spsi * geometry.abs_G0_over_B0 * geometry.torsion
        + iotaN * (Y1s**2 + Y1c**2)
    )
    tt = jnp.zeros_like(tn)
    grad_B_axis_cylindrical = _assemble_gradient(
        geometry.tangent_cylindrical,
        geometry.normal_cylindrical,
        geometry.binormal_cylindrical,
        nn=nn,
        bn=bn,
        nb=nb,
        bb=bb,
        tn=tn,
        nt=nt,
        tt=tt,
    )
    grad_B_axis = _assemble_gradient(
        geometry.tangent_cartesian,
        geometry.normal_cartesian,
        geometry.binormal_cartesian,
        nn=nn,
        bn=bn,
        nb=nb,
        bb=bb,
        tn=tn,
        nt=nt,
        tt=tt,
    )
    grad_B_frobenius_squared = jnp.sum(grad_B_axis**2, axis=(-2, -1))
    L_grad_B = inputs.B0 * jnp.sqrt(2 / grad_B_frobenius_squared)
    B_axis_cylindrical = inputs.sG * inputs.B0 * geometry.tangent_cylindrical
    B_axis = inputs.sG * inputs.B0 * geometry.tangent_cartesian

    return NearAxisSolution(
        inputs=inputs,
        geometry=geometry,
        root_report=root_report,
        sigma=sigma,
        iota=iota,
        iotaN=iotaN,
        helicity=helicity,
        G0=G0,
        X1s=X1s,
        X1c=X1c,
        Y1s=Y1s,
        Y1c=Y1c,
        X1s_untwisted=X1s_untwisted,
        X1c_untwisted=X1c_untwisted,
        Y1s_untwisted=Y1s_untwisted,
        Y1c_untwisted=Y1c_untwisted,
        elongation=elongation,
        mean_elongation=mean_elongation,
        B_axis_cylindrical=B_axis_cylindrical,
        B_axis=B_axis,
        grad_B_axis_cylindrical=grad_B_axis_cylindrical,
        grad_B_axis=grad_B_axis,
        L_grad_B=L_grad_B,
    )


def _normalize_order(order: int | str) -> int:
    if order in (1, "r1"):
        return 1
    if order in (2, "r2"):
        return 2
    if order in (3, "r3"):
        raise NotImplementedError("Third-order solves are not implemented yet.")
    raise ValueError("order must be 1, 2, 3, 'r1', 'r2', or 'r3'.")


def solve(
    *,
    axis: Axis,
    etabar: ArrayLike,
    B0: ArrayLike = 1.0,
    sigma0: ArrayLike = 0.0,
    I2: ArrayLike = 0.0,
    p2: ArrayLike = 0.0,
    B2c: ArrayLike = 0.0,
    B2s: ArrayLike = 0.0,
    nphi: int = 61,
    order: int | str = 1,
    sG: int = 1,
    spsi: int = 1,
    solve_for: str = "iota",
    root_options: RootSolveOptions = DEFAULT_ROOT_OPTIONS,
) -> NearAxisSolution:
    """Construct an immutable near-axis solution.

    Only standard first-order ``etabar -> iota`` mode is enabled in this
    milestone. Other validated modes are added without changing this entry
    point.
    """

    normalized_order = _normalize_order(order)
    if solve_for != "iota":
        raise NotImplementedError("Inverse near-axis solve modes are not implemented yet.")
    inputs = NearAxisInputs(
        axis=axis,
        etabar=etabar,
        B0=B0,
        sigma0=sigma0,
        I2=I2,
        p2=p2,
        B2c=B2c,
        B2s=B2s,
        nphi=nphi,
        order=normalized_order,
        sG=sG,
        spsi=spsi,
        solve_for=solve_for,
    )
    geometry = compute_axis_geometry(axis, nphi=nphi)
    sigma, iota, root_report = solve_sigma(
        inputs,
        geometry,
        root_options=root_options,
    )
    solution = first_order_solution(inputs, geometry, sigma, iota, root_report)
    if normalized_order == 2:
        from pyqsc_jax.second_order import solve_second_order

        solution = solve_second_order(solution)
    return solution


def Qsc(
    rc: ArrayLike,
    zs: ArrayLike,
    *,
    rs: ArrayLike = (),
    zc: ArrayLike = (),
    nfp: int = 1,
    **kwargs: Any,
) -> NearAxisSolution:
    """pyQSC-familiar convenience constructor for the immutable solution."""

    return solve(axis=Axis(rc=rc, rs=rs, zc=zc, zs=zs, nfp=nfp), **kwargs)

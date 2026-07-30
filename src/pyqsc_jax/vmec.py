"""Fast, diagnosed conversion of near-axis surfaces to VMEC INDATA files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.second_order import MU0

ArrayLike = Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VmecBoundary:
    """Uniform-phi surface samples and VMEC Fourier coefficients.

    Coefficient arrays have shape ``(2 * ntor + 1, mpol + 1)``. The first
    index is ordered from toroidal mode ``-ntor`` through ``+ntor``.
    """

    R: jax.Array
    Z: jax.Array
    phi0: jax.Array
    RBC: jax.Array
    RBS: jax.Array
    ZBC: jax.Array
    ZBS: jax.Array
    maximum_toroidal_angle_residual: jax.Array
    maximum_R_reconstruction_error: jax.Array
    maximum_Z_reconstruction_error: jax.Array
    ntheta: int = field(metadata={"static": True})
    mpol: int = field(metadata={"static": True})
    ntor: int = field(metadata={"static": True})
    newton_iterations: int = field(metadata={"static": True})


@dataclass(frozen=True)
class VmecInputParameters:
    """Numerical controls written to a VMEC ``&INDATA`` namelist."""

    delt: float = 0.9
    nstep: int = 200
    tcon0: float = 2.0
    ns_array: tuple[int, ...] = (15, 31, 61)
    ftol_array: tuple[float, ...] = (1.0e-10, 1.0e-11, 1.0e-12)
    niter_array: tuple[int, ...] = (2000, 3000, 5000)

    def __post_init__(self) -> None:
        lengths = (len(self.ns_array), len(self.ftol_array), len(self.niter_array))
        if not self.ns_array or len(set(lengths)) != 1:
            raise ValueError("ns_array, ftol_array, and niter_array must be nonempty and aligned.")
        if self.delt <= 0 or self.nstep < 1 or self.tcon0 <= 0:
            raise ValueError("VMEC damping, step interval, and constraint factor must be positive.")
        if any(value < 3 for value in self.ns_array):
            raise ValueError("Every VMEC radial resolution must be at least 3.")
        if any(value <= 0 for value in self.ftol_array):
            raise ValueError("Every VMEC force tolerance must be positive.")
        if any(value < 1 for value in self.niter_array):
            raise ValueError("Every VMEC iteration limit must be positive.")


@dataclass(frozen=True)
class VmecExport:
    """Result of writing a diagnosed VMEC input file."""

    path: Path
    boundary: VmecBoundary
    phiedge: float
    curtor: float
    pressure_axis: float
    lasym: bool
    conversion_seconds: float


def _periodic_interpolate(
    query: jax.Array,
    grid: jax.Array,
    values: jax.Array,
    period: jax.Array,
) -> jax.Array:
    period = jnp.asarray(period)
    wrapped = jnp.mod(query, period)
    extended_grid = jnp.concatenate((grid, period[None]))
    extended_values = jnp.concatenate((values, values[:1]))
    return jnp.interp(wrapped, extended_grid, extended_values)


def _interpolated_displacements(
    solution: NearAxisSolution,
    radius: jax.Array,
    theta: jax.Array,
    phi0: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    period = 2 * jnp.pi / solution.inputs.axis.nfp
    grid = solution.phi
    interpolate = lambda values: _periodic_interpolate(phi0, grid, values, period)  # noqa: E731
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    X = radius * (
        interpolate(solution.X1c_untwisted) * cosine + interpolate(solution.X1s_untwisted) * sine
    )
    Y = radius * (
        interpolate(solution.Y1c_untwisted) * cosine + interpolate(solution.Y1s_untwisted) * sine
    )
    Z = jnp.zeros_like(X)
    if solution.second_order is not None:
        cosine2 = jnp.cos(2 * theta)
        sine2 = jnp.sin(2 * theta)
        X = X + radius**2 * (
            interpolate(solution.X20_untwisted)
            + interpolate(solution.X2c_untwisted) * cosine2
            + interpolate(solution.X2s_untwisted) * sine2
        )
        Y = Y + radius**2 * (
            interpolate(solution.Y20_untwisted)
            + interpolate(solution.Y2c_untwisted) * cosine2
            + interpolate(solution.Y2s_untwisted) * sine2
        )
        Z = Z + radius**2 * (
            interpolate(solution.Z20_untwisted)
            + interpolate(solution.Z2c_untwisted) * cosine2
            + interpolate(solution.Z2s_untwisted) * sine2
        )
    if solution.third_order is not None:
        cosine3 = jnp.cos(3 * theta)
        sine3 = jnp.sin(3 * theta)
        X = X + radius**3 * (
            interpolate(solution.X3c1_untwisted) * cosine
            + interpolate(solution.X3s1_untwisted) * sine
            + interpolate(solution.X3c3_untwisted) * cosine3
            + interpolate(solution.X3s3_untwisted) * sine3
        )
        Y = Y + radius**3 * (
            interpolate(solution.Y3c1_untwisted) * cosine
            + interpolate(solution.Y3s1_untwisted) * sine
            + interpolate(solution.Y3c3_untwisted) * cosine3
            + interpolate(solution.Y3s3_untwisted) * sine3
        )
        Z = Z + radius**3 * (
            interpolate(solution.Z3c1_untwisted) * cosine
            + interpolate(solution.Z3s1_untwisted) * sine
            + interpolate(solution.Z3c3_untwisted) * cosine3
            + interpolate(solution.Z3s3_untwisted) * sine3
        )
    return X, Y, Z


def _surface_at_axis_angle(
    solution: NearAxisSolution,
    radius: jax.Array,
    theta: jax.Array,
    phi0: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    period = 2 * jnp.pi / solution.inputs.axis.nfp
    grid = solution.phi
    interpolate = lambda values: _periodic_interpolate(phi0, grid, values, period)  # noqa: E731
    X, Y, Z = _interpolated_displacements(solution, radius, theta, phi0)
    normal = solution.geometry.normal_cylindrical
    binormal = solution.geometry.binormal_cylindrical
    tangent = solution.geometry.tangent_cylindrical
    delta_R = (
        X * interpolate(normal[:, 0])
        + Y * interpolate(binormal[:, 0])
        + Z * interpolate(tangent[:, 0])
    )
    delta_phi = (
        X * interpolate(normal[:, 1])
        + Y * interpolate(binormal[:, 1])
        + Z * interpolate(tangent[:, 1])
    )
    delta_Z = (
        X * interpolate(normal[:, 2])
        + Y * interpolate(binormal[:, 2])
        + Z * interpolate(tangent[:, 2])
    )
    axis_R = interpolate(solution.R0)
    R = jnp.hypot(axis_R + delta_R, delta_phi)
    cylindrical_phi = phi0 + jnp.arctan2(delta_phi, axis_R + delta_R)
    cylindrical_Z = interpolate(solution.Z0) + delta_Z
    return R, cylindrical_Z, cylindrical_phi


@partial(
    jax.jit,
    static_argnames=("ntheta", "newton_iterations"),
)
def uniform_cylindrical_surface(
    solution: NearAxisSolution,
    radius: ArrayLike,
    *,
    ntheta: int = 32,
    newton_iterations: int = 6,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map a near-axis boundary to a uniform cylindrical-toroidal grid."""

    radius = jnp.asarray(radius)
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)[:, None]
    target_phi = jnp.broadcast_to(solution.phi[None, :], (ntheta, solution.inputs.nphi))
    phi0 = target_phi

    def newton_step(_iteration, current_phi0):
        angle_function = lambda value: _surface_at_axis_angle(  # noqa: E731
            solution,
            radius,
            theta,
            value,
        )[2]
        cylindrical_phi, derivative = jax.jvp(
            angle_function,
            (current_phi0,),
            (jnp.ones_like(current_phi0),),
        )
        derivative_floor = jnp.sqrt(jnp.finfo(current_phi0.dtype).eps)
        safe_derivative = jnp.where(
            jnp.abs(derivative) > derivative_floor,
            derivative,
            jnp.where(derivative >= 0, derivative_floor, -derivative_floor),
        )
        return current_phi0 - (cylindrical_phi - target_phi) / safe_derivative

    phi0 = jax.lax.fori_loop(0, newton_iterations, newton_step, phi0)
    R, Z, cylindrical_phi = _surface_at_axis_angle(
        solution,
        radius,
        theta,
        phi0,
    )
    residual = jnp.max(jnp.abs(cylindrical_phi - target_phi))
    return R, Z, phi0, residual


def _fft_coefficients(
    values: jax.Array,
    mpol: int,
    ntor: int,
) -> tuple[jax.Array, jax.Array]:
    ntheta, nphi = values.shape
    spectrum = jnp.fft.fft2(values) / (ntheta * nphi)
    poloidal_modes = jnp.arange(mpol + 1)
    toroidal_modes = jnp.arange(-ntor, ntor + 1)
    selected = spectrum[
        poloidal_modes[None, :],
        jnp.mod(-toroidal_modes[:, None], nphi),
    ]
    cosine = 2 * jnp.real(selected)
    sine = -2 * jnp.imag(selected)
    constant_index = ntor
    cosine = cosine.at[constant_index, 0].set(jnp.real(spectrum[0, 0]))
    sine = sine.at[constant_index, 0].set(0)
    cosine = cosine.at[:ntor, 0].set(0)
    sine = sine.at[:ntor, 0].set(0)
    return cosine, sine


def _reconstruct_surface(
    cosine_coefficients: jax.Array,
    sine_coefficients: jax.Array,
    *,
    ntheta: int,
    nphi: int,
    nfp: int,
    mpol: int,
    ntor: int,
) -> jax.Array:
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    phi2d, theta2d = jnp.meshgrid(phi, theta, indexing="xy")
    poloidal_modes = jnp.arange(mpol + 1)
    toroidal_modes = jnp.arange(-ntor, ntor + 1)
    angle = (
        poloidal_modes[None, :, None, None] * theta2d[None, None, :, :]
        - toroidal_modes[:, None, None, None] * nfp * phi2d[None, None, :, :]
    )
    return jnp.sum(
        cosine_coefficients[:, :, None, None] * jnp.cos(angle)
        + sine_coefficients[:, :, None, None] * jnp.sin(angle),
        axis=(0, 1),
    )


@partial(
    jax.jit,
    static_argnames=("ntheta", "mpol", "ntor", "newton_iterations"),
)
def vmec_boundary(
    solution: NearAxisSolution,
    radius: ArrayLike,
    *,
    ntheta: int = 40,
    mpol: int = 12,
    ntor: int = 14,
    newton_iterations: int = 6,
) -> VmecBoundary:
    """Return a uniformly sampled, FFT-projected VMEC boundary."""

    R, Z, phi0, angle_residual = uniform_cylindrical_surface(
        solution,
        radius,
        ntheta=ntheta,
        newton_iterations=newton_iterations,
    )
    RBC, RBS = _fft_coefficients(R, mpol, ntor)
    ZBC, ZBS = _fft_coefficients(Z, mpol, ntor)
    reconstructed_R = _reconstruct_surface(
        RBC,
        RBS,
        ntheta=ntheta,
        nphi=solution.inputs.nphi,
        nfp=solution.inputs.axis.nfp,
        mpol=mpol,
        ntor=ntor,
    )
    reconstructed_Z = _reconstruct_surface(
        ZBC,
        ZBS,
        ntheta=ntheta,
        nphi=solution.inputs.nphi,
        nfp=solution.inputs.axis.nfp,
        mpol=mpol,
        ntor=ntor,
    )
    return VmecBoundary(
        R=R,
        Z=Z,
        phi0=phi0,
        RBC=RBC,
        RBS=RBS,
        ZBC=ZBC,
        ZBS=ZBS,
        maximum_toroidal_angle_residual=angle_residual,
        maximum_R_reconstruction_error=jnp.max(jnp.abs(reconstructed_R - R)),
        maximum_Z_reconstruction_error=jnp.max(jnp.abs(reconstructed_Z - Z)),
        ntheta=ntheta,
        mpol=mpol,
        ntor=ntor,
        newton_iterations=newton_iterations,
    )


def _validated_resolution(
    solution: NearAxisSolution,
    *,
    ntheta: int,
    mpol: int,
    ntor: int,
    newton_iterations: int,
) -> None:
    integers = {
        "ntheta": ntheta,
        "mpol": mpol,
        "ntor": ntor,
        "newton_iterations": newton_iterations,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integers.values()):
        raise ValueError("VMEC conversion resolutions must be integers.")
    if ntheta < 2 * (mpol + 1):
        raise ValueError("ntheta must be at least 2 * (mpol + 1).")
    if mpol < 1 or ntor < 0 or newton_iterations < 1:
        raise ValueError("mpol and Newton iterations must be positive; ntor must be nonnegative.")
    if 2 * ntor + 1 > solution.inputs.nphi:
        raise ValueError("The solution nphi must be at least 2 * ntor + 1.")


def _input_parameters(
    parameters: VmecInputParameters | Mapping[str, Any] | None,
) -> VmecInputParameters:
    if parameters is None:
        return VmecInputParameters()
    if isinstance(parameters, VmecInputParameters):
        return parameters
    allowed = set(VmecInputParameters.__dataclass_fields__)
    unknown = set(parameters) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown VMEC input parameter(s): {names}.")
    converted = dict(parameters)
    for name in ("ns_array", "ftol_array", "niter_array"):
        if name in converted:
            converted[name] = tuple(converted[name])
    return VmecInputParameters(**converted)


def _format_sequence(values: tuple[Any, ...] | jax.Array) -> str:
    return ", ".join(f"{float(value):.16e}" for value in values)


def _format_integer_sequence(values: tuple[int, ...]) -> str:
    return ", ".join(str(int(value)) for value in values)


def _coefficient_lines(
    boundary: VmecBoundary,
    *,
    lasym: bool,
    coefficient_tolerance: float,
) -> list[str]:
    arrays = tuple(
        jax.device_get(value) for value in (boundary.RBC, boundary.RBS, boundary.ZBC, boundary.ZBS)
    )
    RBC, RBS, ZBC, ZBS = arrays
    lines: list[str] = []
    for m in range(boundary.mpol + 1):
        for n in range(-boundary.ntor, boundary.ntor + 1):
            index = n + boundary.ntor
            symmetric_nonzero = (
                abs(RBC[index, m]) > coefficient_tolerance
                or abs(ZBS[index, m]) > coefficient_tolerance
            )
            asymmetric_nonzero = lasym and (
                abs(RBS[index, m]) > coefficient_tolerance
                or abs(ZBC[index, m]) > coefficient_tolerance
            )
            if symmetric_nonzero or asymmetric_nonzero:
                lines.append(
                    f"  RBC({n:03d},{m:03d}) = {RBC[index, m]:+.16e},"
                    f" ZBS({n:03d},{m:03d}) = {ZBS[index, m]:+.16e}"
                )
                if lasym:
                    lines.append(
                        f"  RBS({n:03d},{m:03d}) = {RBS[index, m]:+.16e},"
                        f" ZBC({n:03d},{m:03d}) = {ZBC[index, m]:+.16e}"
                    )
    return lines


def to_vmec(
    solution: NearAxisSolution,
    filename: str | Path,
    *,
    r: float = 0.1,
    parameters: VmecInputParameters | Mapping[str, Any] | None = None,
    ntheta: int = 40,
    mpol: int = 12,
    ntor: int = 14,
    ntor_max: int = 14,
    newton_iterations: int = 6,
    coefficient_tolerance: float = 1.0e-14,
) -> VmecExport:
    """Write a VMEC fixed-boundary input and return conversion diagnostics."""

    radius = float(r)
    if radius <= 0:
        raise ValueError("r must be positive.")
    if not isinstance(ntor_max, int) or isinstance(ntor_max, bool) or ntor_max < 0:
        raise ValueError("ntor_max must be a nonnegative integer.")
    if coefficient_tolerance < 0:
        raise ValueError("coefficient_tolerance must be nonnegative.")
    effective_ntor = min(ntor, ntor_max)
    _validated_resolution(
        solution,
        ntheta=ntheta,
        mpol=mpol,
        ntor=effective_ntor,
        newton_iterations=newton_iterations,
    )
    controls = _input_parameters(parameters)

    start = perf_counter()
    boundary = vmec_boundary(
        solution,
        radius,
        ntheta=ntheta,
        mpol=mpol,
        ntor=effective_ntor,
        newton_iterations=newton_iterations,
    )
    jax.tree.map(lambda value: value.block_until_ready(), boundary)
    conversion_seconds = perf_counter() - start

    asymmetric_amplitude = max(
        float(jnp.max(jnp.abs(boundary.RBS))),
        float(jnp.max(jnp.abs(boundary.ZBC))),
    )
    lasym = asymmetric_amplitude > coefficient_tolerance
    inputs = solution.inputs
    axis = inputs.axis
    phiedge = float(jnp.pi * radius**2 * inputs.B0)
    curtor = float(2 * jnp.pi * inputs.I2 * radius**2 / MU0)
    pressure_axis = float(-inputs.p2 * radius**2)
    lines = [
        "! Generated deterministically by pyQSC_JAX.",
        f"! Near-axis radius r = {radius:.16e}; etabar = {float(inputs.etabar):.16e}.",
        (
            f"! nphi = {inputs.nphi}; order = r{inputs.order}; ntheta = {ntheta};"
            f" mpol = {mpol}; ntor = {effective_ntor}."
        ),
        (
            "! Conversion diagnostics:"
            f" max_phi_residual = {float(boundary.maximum_toroidal_angle_residual):.6e};"
            f" max_R_error = {float(boundary.maximum_R_reconstruction_error):.6e};"
            f" max_Z_error = {float(boundary.maximum_Z_reconstruction_error):.6e}."
        ),
        "&INDATA",
        f"  DELT = {controls.delt:.16e}",
        f"  NSTEP = {controls.nstep}",
        f"  TCON0 = {controls.tcon0:.16e}",
        f"  NS_ARRAY = {_format_integer_sequence(controls.ns_array)}",
        f"  FTOL_ARRAY = {_format_sequence(controls.ftol_array)}",
        f"  NITER_ARRAY = {_format_integer_sequence(controls.niter_array)}",
        f"  LASYM = {'T' if lasym else 'F'}",
        "  LFREEB = F",
        f"  NFP = {axis.nfp}",
        f"  MPOL = {mpol}",
        f"  NTOR = {effective_ntor}",
        f"  PHIEDGE = {phiedge:.16e}",
        "  PRES_SCALE = 1.0000000000000000e+00",
        "  PMASS_TYPE = 'power_series'",
        f"  AM = {pressure_axis:.16e}, {-pressure_axis:.16e}",
        f"  CURTOR = {curtor:.16e}",
        "  NCURR = 1",
        "  PCURR_TYPE = 'power_series'",
        "  AC = 1.0000000000000000e+00",
        f"  RAXIS_CC = {_format_sequence(axis.rc)}",
        f"  RAXIS_CS = {_format_sequence(-axis.rs)}",
        f"  ZAXIS_CC = {_format_sequence(axis.zc)}",
        f"  ZAXIS_CS = {_format_sequence(-axis.zs)}",
        "! Boundary coefficients",
    ]
    lines.extend(
        _coefficient_lines(
            boundary,
            lasym=lasym,
            coefficient_tolerance=coefficient_tolerance,
        )
    )
    lines.append("/")
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return VmecExport(
        path=output_path,
        boundary=boundary,
        phiedge=phiedge,
        curtor=curtor,
        pressure_axis=pressure_axis,
        lasym=lasym,
        conversion_seconds=conversion_seconds,
    )

"""Near-axis boundary surfaces, VMEC INDATA export, and the in-memory VMEX bridge.

VMEX is imported lazily, only when a VMEX function is called.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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
    toroidal_angle_tolerance: jax.Array
    toroidal_angle_converged: jax.Array
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
    query: jax.Array, grid: jax.Array, values: jax.Array, period: jax.Array
) -> jax.Array:
    period = jnp.asarray(period)
    wrapped = jnp.mod(query, period)
    extended_grid = jnp.concatenate((grid, period[None]))
    extended_values = jnp.concatenate((values, values[:1]))
    return jnp.interp(wrapped, extended_grid, extended_values)


def frenet_displacements(
    solution: NearAxisSolution, radius: jax.Array, theta: jax.Array, phi0: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate all available near-axis Frenet displacements."""

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
    solution: NearAxisSolution, radius: jax.Array, theta: jax.Array, phi0: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    period = 2 * jnp.pi / solution.inputs.axis.nfp
    grid = solution.phi
    interpolate = lambda values: _periodic_interpolate(phi0, grid, values, period)  # noqa: E731
    X, Y, Z = frenet_displacements(solution, radius, theta, phi0)
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


@partial(jax.jit, static_argnames=("ntheta", "newton_iterations"))
def uniform_cylindrical_surface(
    solution: NearAxisSolution, radius: ArrayLike, *, ntheta: int = 32, newton_iterations: int = 6
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map a near-axis boundary to a uniform cylindrical-toroidal grid."""

    radius = jnp.asarray(radius)
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)[:, None]
    target_phi = jnp.broadcast_to(solution.phi[None, :], (ntheta, solution.inputs.nphi))
    phi0 = target_phi

    def newton_step(_iteration, current_phi0):
        angle_function = lambda value: _surface_at_axis_angle(  # noqa: E731
            solution, radius, theta, value
        )[2]
        cylindrical_phi, derivative = jax.jvp(
            angle_function, (current_phi0,), (jnp.ones_like(current_phi0),)
        )
        derivative_floor = jnp.sqrt(jnp.finfo(current_phi0.dtype).eps)
        safe_derivative = jnp.where(
            jnp.abs(derivative) > derivative_floor,
            derivative,
            jnp.where(derivative >= 0, derivative_floor, -derivative_floor),
        )
        return current_phi0 - (cylindrical_phi - target_phi) / safe_derivative

    phi0 = jax.lax.fori_loop(0, newton_iterations, newton_step, phi0)
    R, Z, cylindrical_phi = _surface_at_axis_angle(solution, radius, theta, phi0)
    residual = jnp.max(jnp.abs(cylindrical_phi - target_phi))
    return R, Z, phi0, residual


def _fft_coefficients(values: jax.Array, mpol: int, ntor: int) -> tuple[jax.Array, jax.Array]:
    ntheta, nphi = values.shape
    spectrum = jnp.fft.fft2(values) / (ntheta * nphi)
    poloidal_modes = jnp.arange(mpol + 1)
    toroidal_modes = jnp.arange(-ntor, ntor + 1)
    selected = spectrum[poloidal_modes[None, :], jnp.mod(-toroidal_modes[:, None], nphi)]
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


@partial(jax.jit, static_argnames=("ntheta", "mpol", "ntor", "newton_iterations"))
def vmec_boundary(
    solution: NearAxisSolution,
    radius: ArrayLike,
    *,
    ntheta: int = 40,
    mpol: int = 12,
    ntor: int = 14,
    newton_iterations: int = 6,
    toroidal_angle_tolerance: float = 0.0,
) -> VmecBoundary:
    """Return a uniformly sampled, FFT-projected VMEC boundary.

    A zero ``toroidal_angle_tolerance`` selects 100 machine epsilons
    for the active JAX dtype. The returned convergence flag is data, so this
    function remains JIT-compatible; :func:`to_vmec` turns a false flag into
    a hard failure before writing an input file.
    """

    R, Z, phi0, angle_residual = uniform_cylindrical_surface(
        solution, radius, ntheta=ntheta, newton_iterations=newton_iterations
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
    requested_tolerance = jnp.asarray(toroidal_angle_tolerance, dtype=R.dtype)
    automatic_tolerance = 100 * jnp.finfo(R.dtype).eps
    angle_tolerance = jnp.where(requested_tolerance == 0, automatic_tolerance, requested_tolerance)
    return VmecBoundary(
        R=R,
        Z=Z,
        phi0=phi0,
        RBC=RBC,
        RBS=RBS,
        ZBC=ZBC,
        ZBS=ZBS,
        maximum_toroidal_angle_residual=angle_residual,
        toroidal_angle_tolerance=angle_tolerance,
        toroidal_angle_converged=angle_residual <= angle_tolerance,
        maximum_R_reconstruction_error=jnp.max(jnp.abs(reconstructed_R - R)),
        maximum_Z_reconstruction_error=jnp.max(jnp.abs(reconstructed_Z - Z)),
        ntheta=ntheta,
        mpol=mpol,
        ntor=ntor,
        newton_iterations=newton_iterations,
    )


def _validated_resolution(
    solution: NearAxisSolution, *, ntheta: int, mpol: int, ntor: int, newton_iterations: int
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
    boundary: VmecBoundary, *, lasym: bool, coefficient_tolerance: float
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
    toroidal_angle_tolerance: float = 0.0,
    coefficient_tolerance: float = 1.0e-14,
) -> VmecExport:
    """Write a VMEC fixed-boundary input and return conversion diagnostics.

    ``mpol`` is the highest poloidal index written, so the namelist has ``MPOL = mpol + 1``:
    VMEC retains ``m < MPOL``. (pyQSC writes ``MPOL = mpol`` and so drops its ``m = mpol`` row.)
    """

    radius = float(r)
    if not isfinite(radius) or radius <= 0:
        raise ValueError("r must be positive and finite.")
    if not isinstance(ntor_max, int) or isinstance(ntor_max, bool) or ntor_max < 0:
        raise ValueError("ntor_max must be a nonnegative integer.")
    if not isfinite(toroidal_angle_tolerance) or toroidal_angle_tolerance < 0:
        raise ValueError("toroidal_angle_tolerance must be nonnegative and finite.")
    if not isfinite(coefficient_tolerance) or coefficient_tolerance < 0:
        raise ValueError("coefficient_tolerance must be nonnegative and finite.")
    effective_ntor = min(ntor, ntor_max)
    _validated_resolution(
        solution, ntheta=ntheta, mpol=mpol, ntor=effective_ntor, newton_iterations=newton_iterations
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
        toroidal_angle_tolerance=toroidal_angle_tolerance,
    )
    jax.tree.map(lambda value: value.block_until_ready(), boundary)
    conversion_seconds = perf_counter() - start
    if not bool(boundary.toroidal_angle_converged):
        residual = float(boundary.maximum_toroidal_angle_residual)
        tolerance = float(boundary.toroidal_angle_tolerance)
        raise RuntimeError(
            "The near-axis surface could not be represented on a uniform "
            "cylindrical-toroidal grid: the maximum angle residual "
            f"{residual:.6e} exceeds {tolerance:.6e}. Reduce r or increase "
            "newton_iterations; no VMEC input was written."
        )

    lasym = _is_asymmetric(boundary, solution, tolerance=coefficient_tolerance)
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
            f" highest poloidal index = {mpol}; ntor = {effective_ntor}."
        ),
        (
            "! Conversion diagnostics:"
            f" max_phi_residual = {float(boundary.maximum_toroidal_angle_residual):.6e};"
            f" phi_tolerance = {float(boundary.toroidal_angle_tolerance):.6e};"
            " phi_converged = true;"
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
        f"  MPOL = {mpol + 1}",  # VMEC keeps m < MPOL; the boundary includes m = mpol.
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
        _coefficient_lines(boundary, lasym=lasym, coefficient_tolerance=coefficient_tolerance)
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


VMEX_VALIDATED_COMMIT = "2a40d7566be083070ea3ea534fa5d1fc44ad733a"


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VmexRadialQuantities:
    """Differentiable radial and scalar quantities from a converged VMEX state.

    ``iota_vmec`` retains VMEC's native sign convention. ``iota`` uses the
    pyQSC_JAX toroidal-angle convention, which is the negative of VMEC's for
    boundaries produced by :func:`pyqsc_jax.vmec_boundary`.
    """

    s: jax.Array
    iota: jax.Array
    iota_vmec: jax.Array
    qs_surfaces: jax.Array
    quasisymmetry: jax.Array
    magnetic_well: jax.Array
    aspect: jax.Array
    volume: jax.Array
    magnetic_energy: jax.Array
    thermal_energy: jax.Array


@dataclass(frozen=True)
class VmexEquilibrium:
    """A VMEX implicit solution together with user-facing radial quantities."""

    solution: Any
    quantities: VmexRadialQuantities


@dataclass(frozen=True)
class VmexProblem:
    """Reusable static VMEX problem and its differentiable parameter pytree."""

    input: Any
    parameters: Any
    boundary: VmecBoundary
    radius: float
    qs_surfaces: tuple[float, ...]
    helicity_m: int
    helicity_n: int
    ntheta: int
    mpol: int
    ntor: int
    newton_iterations: int
    toroidal_angle_tolerance: float
    ftol: float
    max_iterations: int
    adjoint_tol: float
    multigrid: bool
    device: Any
    vmex_version: str
    validated_commit: str = VMEX_VALIDATED_COMMIT

    @property
    def finite_beta(self) -> bool:
        """Whether the static reference input has a nonzero pressure profile."""

        return bool(np.any(np.asarray(self.input.am)))

    def parameters_for(self, solution: NearAxisSolution, *, radius: Any | None = None) -> Any:
        """Map another near-axis solution into this problem's parameter pytree."""

        return vmex_parameters_from_solution(self, solution, radius=radius)

    def quantities(self, parameters: Any | None = None) -> VmexRadialQuantities:
        """Solve and return differentiable radial quantities."""

        return vmex_radial_quantities(self, parameters)

    def solve(self, parameters: Any | None = None) -> VmexEquilibrium:
        """Solve the fixed-boundary equilibrium and retain the raw VMEX result."""

        return solve_vmex(self, parameters)


def _import_vmex():
    try:
        module = importlib.import_module("vmex")
    except ImportError as error:
        raise ImportError(
            "The differentiable equilibrium interface requires VMEX. Install "
            "pyqsc-jax with the 'vmex' extra, or install the current source with "
            "'python -m pip install git+https://github.com/uwplasma/vmex.git'."
        ) from error
    missing = tuple(
        name for name in ("VmecInput", "implicit", "optimize") if not hasattr(module, name)
    )
    if missing:
        names = ", ".join(missing)
        raise ImportError(f"The installed VMEX does not provide the required public API: {names}.")
    return module


def _validated_surfaces(surfaces: Any) -> tuple[float, ...]:
    values = tuple(float(value) for value in surfaces)
    if any(not np.isfinite(value) or value <= 0 or value > 1 for value in values):
        raise ValueError("Every VMEX quasisymmetry surface must satisfy 0 < s <= 1.")
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise ValueError("VMEX quasisymmetry surfaces must be strictly increasing.")
    return values


def _validated_radial_controls(
    ns_array: Any, ftol_array: Any | None, *, ftol: float, max_iterations: int
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    ns = tuple(int(value) for value in ns_array)
    if not ns or any(value < 3 for value in ns):
        raise ValueError("ns_array must be nonempty and every radial resolution must be >= 3.")
    if any(right <= left for left, right in zip(ns, ns[1:], strict=False)):
        raise ValueError("ns_array must be strictly increasing.")
    if not np.isfinite(ftol) or ftol <= 0:
        raise ValueError("ftol must be positive and finite.")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if ftol_array is None:
        if len(ns) == 1:
            tolerances = (float(ftol),)
        else:
            start = max(float(ftol), 1.0e-8)
            tolerances = tuple(float(value) for value in np.geomspace(start, ftol, len(ns)))
    else:
        tolerances = tuple(float(value) for value in ftol_array)
    if len(tolerances) != len(ns) or any(
        not np.isfinite(value) or value <= 0 for value in tolerances
    ):
        raise ValueError("ftol_array must contain one positive finite value per ns_array stage.")
    return ns, tolerances, (int(max_iterations),) * len(ns)


def _stellarator_symmetric_inputs(solution: NearAxisSolution) -> bool:
    """True when the axis and second-order inputs are stellarator symmetric.

    The boundary is then symmetric by construction; its RBS/ZBC coefficients are FFT
    round-off and must not switch on LASYM.
    """
    inputs = solution.inputs
    values = (inputs.axis.rs, inputs.axis.zc, inputs.sigma0, inputs.B2s)
    try:
        return all(
            float(jnp.max(jnp.abs(jnp.atleast_1d(jnp.asarray(v))))) == 0.0
            if jnp.size(jnp.asarray(v))
            else True
            for v in values
        )
    except jax.errors.ConcretizationTypeError:
        return False


def _is_asymmetric(
    boundary: VmecBoundary, solution: NearAxisSolution | None = None, tolerance: float = 1.0e-13
) -> bool:
    if solution is not None and _stellarator_symmetric_inputs(solution):
        return False
    return (
        max(float(jnp.max(jnp.abs(boundary.RBS))), float(jnp.max(jnp.abs(boundary.ZBC))))
        > tolerance
    )


def _require_converged_boundary(boundary: VmecBoundary) -> None:
    """Reject concrete failures while remaining usable inside JAX tracing."""

    try:
        converged = bool(boundary.toroidal_angle_converged)
    except jax.errors.TracerBoolConversionError:
        return
    if not converged:
        raise RuntimeError(
            "The VMEX boundary angle inversion did not converge. Reduce r or "
            "increase newton_iterations before solving the radial equilibrium."
        )


def _profile_arrays(parameters: Any, solution: NearAxisSolution, radius: Any):
    radius = jnp.asarray(radius)
    pressure_axis = -solution.inputs.p2 * radius**2
    am = jnp.zeros_like(parameters.am)
    am = am.at[0].set(pressure_axis)
    if am.shape[0] > 1:
        am = am.at[1].set(-pressure_axis)
    ac = jnp.zeros_like(parameters.ac).at[0].set(1.0)
    phiedge = jnp.pi * radius**2 * solution.inputs.B0
    curtor = 2 * jnp.pi * solution.inputs.I2 * radius**2 / MU0
    return am, ac, phiedge, curtor


def vmex_parameters_from_solution(
    problem: VmexProblem, solution: NearAxisSolution, *, radius: Any | None = None
) -> Any:
    """Traceably map a near-axis boundary and profiles to VMEX parameters.

    The problem fixes discrete resolution, topology, and solver controls.
    Boundary coefficients, toroidal flux, pressure, and enclosed current
    remain JAX values, so gradients can propagate from a newly constructed
    :class:`NearAxisSolution` through VMEX's converged fixed point.
    """

    if solution.inputs.axis.nfp != problem.input.nfp:
        raise ValueError("The new solution must have the problem's number of field periods.")
    selected_radius = problem.radius if radius is None else radius
    boundary = vmec_boundary(
        solution,
        selected_radius,
        ntheta=problem.ntheta,
        mpol=problem.mpol,
        ntor=problem.ntor,
        newton_iterations=problem.newton_iterations,
        toroidal_angle_tolerance=problem.toroidal_angle_tolerance,
    )
    _require_converged_boundary(boundary)
    boundary_mask = boundary.toroidal_angle_converged
    rbc = jnp.where(boundary_mask, boundary.RBC, jnp.nan)
    rbs = jnp.where(boundary_mask, boundary.RBS, jnp.nan)
    zbc = jnp.where(boundary_mask, boundary.ZBC, jnp.nan)
    zbs = jnp.where(boundary_mask, boundary.ZBS, jnp.nan)
    am, ac, phiedge, curtor = _profile_arrays(problem.parameters, solution, selected_radius)
    return dataclasses.replace(
        problem.parameters,
        rbc=rbc,
        rbs=rbs,
        zbc=zbc,
        zbs=zbs,
        phiedge=phiedge,
        curtor=curtor,
        pres_scale=jnp.asarray(1.0, dtype=phiedge.dtype),
        am=am,
        ac=ac,
    )


def to_vmex_problem(
    solution: NearAxisSolution,
    *,
    r: float = 0.03,
    qs_surfaces: Any = (0.25, 0.5, 0.75, 1.0),
    helicity_m: int = 1,
    helicity_n: int | None = None,
    ntheta: int = 24,
    mpol: int = 6,
    ntor: int = 6,
    newton_iterations: int = 6,
    toroidal_angle_tolerance: float = 0.0,
    ns_array: Any = (15, 31),
    ftol_array: Any | None = None,
    ftol: float = 1.0e-10,
    max_iterations: int = 5000,
    adjoint_tol: float = 1.0e-11,
    multigrid: bool = True,
    device: Any = None,
) -> VmexProblem:
    """Create a differentiable fixed-boundary VMEX problem without disk I/O.

    ``mpol`` is the maximum retained poloidal Fourier index in the
    pyQSC_JAX conversion. VMEX therefore receives ``MPOL = mpol + 1``.
    Pressure and current use the same near-axis-consistent profiles as
    :func:`pyqsc_jax.to_vmec`.
    """

    vmex = _import_vmex()
    radius = float(r)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("r must be positive and finite.")
    surfaces = _validated_surfaces(qs_surfaces)
    ns, tolerances, iteration_limits = _validated_radial_controls(
        ns_array, ftol_array, ftol=ftol, max_iterations=max_iterations
    )
    if not isinstance(helicity_m, int) or isinstance(helicity_m, bool):
        raise ValueError("helicity_m must be an integer.")
    if helicity_n is None:
        helicity_n = -int(np.asarray(solution.helicity))
    if not isinstance(helicity_n, int) or isinstance(helicity_n, bool):
        raise ValueError("helicity_n must be an integer.")
    if not np.isfinite(adjoint_tol) or adjoint_tol <= 0:
        raise ValueError("adjoint_tol must be positive and finite.")
    if not np.isfinite(toroidal_angle_tolerance) or toroidal_angle_tolerance < 0:
        raise ValueError("toroidal_angle_tolerance must be nonnegative and finite.")
    _validated_resolution(
        solution, ntheta=ntheta, mpol=mpol, ntor=ntor, newton_iterations=newton_iterations
    )

    boundary = vmec_boundary(
        solution,
        radius,
        ntheta=ntheta,
        mpol=mpol,
        ntor=ntor,
        newton_iterations=newton_iterations,
        toroidal_angle_tolerance=toroidal_angle_tolerance,
    )
    _require_converged_boundary(boundary)
    lasym = _is_asymmetric(boundary, solution)
    if lasym and surfaces:
        raise NotImplementedError(
            "VMEX's traceable quasisymmetry profile currently supports "
            "stellarator-symmetric equilibria only; pass qs_surfaces=() to "
            "solve an asymmetric equilibrium without that diagnostic."
        )

    inputs = solution.inputs
    axis = inputs.axis
    arrays = tuple(
        np.asarray(jax.device_get(value))
        for value in (boundary.RBC, boundary.RBS, boundary.ZBC, boundary.ZBS)
    )
    rbc, rbs, zbc, zbs = arrays
    pressure_axis = -float(inputs.p2) * radius**2
    am = np.zeros(21)
    am[:2] = (pressure_axis, -pressure_axis)
    ac = np.zeros(21)
    ac[0] = 1.0
    vmex_input = vmex.VmecInput(
        lasym=lasym,
        lfreeb=False,
        nfp=axis.nfp,
        mpol=boundary.RBC.shape[1],
        ntor=boundary.ntor,
        ns_array=np.asarray(ns),
        ftol_array=np.asarray(tolerances),
        niter_array=np.asarray(iteration_limits),
        delt=0.9,
        tcon0=2.0,
        phiedge=np.pi * radius**2 * float(inputs.B0),
        pres_scale=1.0,
        am=am,
        ncurr=1,
        ac=ac,
        curtor=2 * np.pi * float(inputs.I2) * radius**2 / MU0,
        raxis_c=np.asarray(axis.rc),
        raxis_s=-np.asarray(axis.rs),
        zaxis_c=np.asarray(axis.zc),
        zaxis_s=-np.asarray(axis.zs),
        rbc=rbc,
        rbs=rbs,
        zbc=zbc,
        zbs=zbs,
    )
    parameters = vmex.implicit.params_from_input(vmex_input, device=device)
    problem = VmexProblem(
        input=vmex_input,
        parameters=parameters,
        boundary=boundary,
        radius=radius,
        qs_surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        mpol=mpol,
        ntor=ntor,
        newton_iterations=newton_iterations,
        toroidal_angle_tolerance=float(toroidal_angle_tolerance),
        ftol=float(ftol),
        max_iterations=max_iterations,
        adjoint_tol=float(adjoint_tol),
        multigrid=bool(multigrid),
        device=device,
        vmex_version=str(vmex.__version__),
    )
    return dataclasses.replace(problem, parameters=vmex_parameters_from_solution(problem, solution))


def _radial_quantities(problem: VmexProblem, vmex_solution: Any) -> VmexRadialQuantities:
    vmex = _import_vmex()
    runtime = vmex_solution.runtime
    if runtime is None:
        raise RuntimeError("VMEX did not retain the runtime required for radial diagnostics.")
    iota_vmec = vmex.implicit.iota_profile(vmex_solution.state, runtime)
    s = jnp.linspace(0.0, 1.0, iota_vmec.shape[0], dtype=iota_vmec.dtype)
    surfaces = jnp.asarray(problem.qs_surfaces, dtype=iota_vmec.dtype)
    if problem.qs_surfaces:
        qs = vmex.optimize.QuasisymmetryRatioResidual(
            problem.qs_surfaces, problem.helicity_m, problem.helicity_n
        )
        quasisymmetry = qs.profile_state(vmex_solution.state, runtime)
    else:
        quasisymmetry = jnp.zeros((0,), dtype=iota_vmec.dtype)
    return VmexRadialQuantities(
        s=s,
        iota=-iota_vmec,
        iota_vmec=iota_vmec,
        qs_surfaces=surfaces,
        quasisymmetry=quasisymmetry,
        magnetic_well=vmex.optimize.magnetic_well(vmex_solution.state, runtime),
        aspect=vmex.optimize.aspect_ratio(vmex_solution.state, runtime),
        volume=vmex.optimize.volume(vmex_solution.state, runtime),
        magnetic_energy=vmex_solution.wb,
        thermal_energy=vmex_solution.wp,
    )


def solve_vmex(problem: VmexProblem, parameters: Any | None = None) -> VmexEquilibrium:
    """Solve one VMEX problem with implicit-AD-compatible parameters."""

    vmex = _import_vmex()
    selected_parameters = problem.parameters if parameters is None else parameters
    solution = vmex.implicit.run(
        problem.input,
        selected_parameters,
        ftol=problem.ftol,
        max_iterations=problem.max_iterations,
        adjoint_tol=problem.adjoint_tol,
        multigrid=problem.multigrid,
        device=problem.device,
    )
    return VmexEquilibrium(solution=solution, quantities=_radial_quantities(problem, solution))


def vmex_radial_quantities(
    problem: VmexProblem, parameters: Any | None = None
) -> VmexRadialQuantities:
    """Solve and return only the differentiable VMEX diagnostic pytree."""

    return solve_vmex(problem, parameters).quantities

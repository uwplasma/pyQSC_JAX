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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
        lengths = {len(self.ns_array), len(self.ftol_array), len(self.niter_array)}
        _require(
            bool(self.ns_array) and len(lengths) == 1,
            "ns_array, ftol_array, and niter_array must be nonempty and aligned.",
        )
        _require(
            self.delt > 0 and self.nstep >= 1 and self.tcon0 > 0,
            "VMEC damping, step interval, and constraint factor must be positive.",
        )
        _require(min(self.ns_array) >= 3, "Every VMEC radial resolution must be at least 3.")
        _require(min(self.ftol_array) > 0, "Every VMEC force tolerance must be positive.")
        _require(min(self.niter_array) >= 1, "Every VMEC iteration limit must be positive.")


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


def _displacements(
    solution: NearAxisSolution, radius: ArrayLike, theta: ArrayLike, sample
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sum the Frenet displacements ``(X, Y, Z)`` of every available order.

    ``sample`` maps a one-period array on ``solution.phi`` to the evaluation points; the
    untwisted coefficients are used, so theta is the poloidal angle of the Frenet frame.
    """

    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    orders = [(1, (("1c", cosine), ("1s", sine)), "XY")]
    if solution.second_order is not None:
        orders.append(
            (2, (("20", 1.0), ("2c", jnp.cos(2 * theta)), ("2s", jnp.sin(2 * theta))), "XYZ")
        )
    if solution.third_order is not None:
        cosine3, sine3 = jnp.cos(3 * theta), jnp.sin(3 * theta)
        harmonics = (("3c1", cosine), ("3s1", sine), ("3c3", cosine3), ("3s3", sine3))
        orders.append((3, harmonics, "XYZ"))
    displacements = {}
    for power, harmonics, components in orders:
        for component in components:
            terms = [
                sample(getattr(solution, f"{component}{name}_untwisted")) * basis
                for name, basis in harmonics
            ]
            term = radius**power * sum(terms[1:], terms[0])
            previous = displacements.get(component)
            displacements[component] = term if previous is None else previous + term
    X, Y = displacements["X"], displacements["Y"]
    return X, Y, displacements.get("Z", jnp.zeros_like(X))


def frenet_displacements(
    solution: NearAxisSolution, radius: jax.Array, theta: jax.Array, phi0: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Frenet displacements ``(X, Y, Z)`` [m] at radius ``r`` [m], angle ``theta`` and axis
    angle ``phi0``, including every order the solution carries (linear periodic interpolation)."""

    period = 2 * jnp.pi / solution.inputs.axis.nfp
    return _displacements(
        solution, radius, theta, lambda v: _periodic_interpolate(phi0, solution.phi, v, period)
    )


def _surface_at_axis_angle(
    solution: NearAxisSolution, radius: jax.Array, theta: jax.Array, phi0: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Cylindrical ``(R, Z, phi)`` of the surface point attached to axis angle ``phi0``."""

    period = 2 * jnp.pi / solution.inputs.axis.nfp
    interpolate = lambda values: _periodic_interpolate(phi0, solution.phi, values, period)  # noqa: E731
    X, Y, Z = frenet_displacements(solution, radius, theta, phi0)
    geometry = solution.geometry
    delta_R, delta_phi, delta_Z = (
        X * interpolate(geometry.normal_cylindrical[:, k])
        + Y * interpolate(geometry.binormal_cylindrical[:, k])
        + Z * interpolate(geometry.tangent_cylindrical[:, k])
        for k in range(3)
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
    reconstructed_R, reconstructed_Z = (
        _reconstruct_surface(
            cosine,
            sine,
            ntheta=ntheta,
            nphi=solution.inputs.nphi,
            nfp=solution.inputs.axis.nfp,
            mpol=mpol,
            ntor=ntor,
        )
        for cosine, sine in ((RBC, RBS), (ZBC, ZBS))
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
    _require(
        all(_is_integer(value) for value in (ntheta, mpol, ntor, newton_iterations)),
        "VMEC conversion resolutions must be integers.",
    )
    _require(ntheta >= 2 * (mpol + 1), "ntheta must be at least 2 * (mpol + 1).")
    _require(
        mpol >= 1 and ntor >= 0 and newton_iterations >= 1,
        "mpol and Newton iterations must be positive; ntor must be nonnegative.",
    )
    _require(
        2 * ntor + 1 <= solution.inputs.nphi, "The solution nphi must be at least 2 * ntor + 1."
    )


def _near_axis_profiles(inputs: Any, radius: Any) -> tuple[Any, Any, Any]:
    """VMEC ``(AM[0], PHIEDGE, CURTOR)`` of the near-axis solution at boundary radius ``r``.

    Pressure ``p = p0 (1 - s)`` with ``p0 = -p2 r**2``; ``PHIEDGE = pi r**2 B0``; the enclosed
    toroidal current is ``2 pi I2 r**2 / mu0`` (all SI).
    """

    return (
        -inputs.p2 * radius**2,
        jnp.pi * radius**2 * inputs.B0,
        2 * jnp.pi * inputs.I2 * radius**2 / MU0,
    )


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


def _format_sequence(values: Any, integer: bool = False) -> str:
    return ", ".join(str(int(v)) if integer else f"{float(v):.16e}" for v in values)


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
    _require(isfinite(radius) and radius > 0, "r must be positive and finite.")
    _require(_is_integer(ntor_max) and ntor_max >= 0, "ntor_max must be a nonnegative integer.")
    for name, value in (
        ("toroidal_angle_tolerance", toroidal_angle_tolerance),
        ("coefficient_tolerance", coefficient_tolerance),
    ):
        _require(isfinite(value) and value >= 0, f"{name} must be nonnegative and finite.")
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
    pressure_axis, phiedge, curtor = map(float, _near_axis_profiles(inputs, radius))
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
        f"  NS_ARRAY = {_format_sequence(controls.ns_array, integer=True)}",
        f"  FTOL_ARRAY = {_format_sequence(controls.ftol_array)}",
        f"  NITER_ARRAY = {_format_sequence(controls.niter_array, integer=True)}",
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
    _require(
        all(np.isfinite(value) and 0 < value <= 1 for value in values),
        "Every VMEX quasisymmetry surface must satisfy 0 < s <= 1.",
    )
    _require(
        all(right > left for left, right in zip(values, values[1:], strict=False)),
        "VMEX quasisymmetry surfaces must be strictly increasing.",
    )
    return values


def _validated_radial_controls(
    ns_array: Any, ftol_array: Any | None, *, ftol: float, max_iterations: int
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    ns = tuple(int(value) for value in ns_array)
    _require(
        bool(ns) and min(ns) >= 3,
        "ns_array must be nonempty and every radial resolution must be >= 3.",
    )
    _require(
        all(right > left for left, right in zip(ns, ns[1:], strict=False)),
        "ns_array must be strictly increasing.",
    )
    _require(np.isfinite(ftol) and ftol > 0, "ftol must be positive and finite.")
    _require(
        _is_integer(max_iterations) and max_iterations >= 1,
        "max_iterations must be a positive integer.",
    )
    if ftol_array is not None:
        tolerances = tuple(float(value) for value in ftol_array)
    elif len(ns) == 1:
        tolerances = (float(ftol),)
    else:  # Geometric ladder from max(ftol, 1e-8) down to ftol.
        tolerances = tuple(float(v) for v in np.geomspace(max(float(ftol), 1.0e-8), ftol, len(ns)))
    _require(
        len(tolerances) == len(ns) and all(np.isfinite(v) and v > 0 for v in tolerances),
        "ftol_array must contain one positive finite value per ns_array stage.",
    )
    return ns, tolerances, (int(max_iterations),) * len(ns)


def _is_asymmetric(boundary: VmecBoundary, solution: NearAxisSolution, tolerance: float) -> bool:
    """LASYM: false when the axis and second-order inputs are stellarator symmetric.

    The boundary is then symmetric by construction, and its RBS/ZBC coefficients are FFT
    round-off that must not switch on LASYM. Otherwise the coefficients decide. Both callers
    write concrete files or inputs, so the values here are concrete.
    """

    inputs = solution.inputs
    symmetric_inputs = (inputs.axis.rs, inputs.axis.zc, inputs.sigma0, inputs.B2s)
    if not any(np.any(np.asarray(value)) for value in symmetric_inputs):
        return False
    largest = max(float(jnp.max(jnp.abs(boundary.RBS))), float(jnp.max(jnp.abs(boundary.ZBC))))
    return largest > tolerance


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
    # A traced, unconverged inversion poisons the boundary instead of raising.
    coefficients = {
        name.lower(): jnp.where(boundary.toroidal_angle_converged, getattr(boundary, name), jnp.nan)
        for name in ("RBC", "RBS", "ZBC", "ZBS")
    }
    pressure_axis, phiedge, curtor = _near_axis_profiles(
        solution.inputs, jnp.asarray(selected_radius)
    )
    am = jnp.zeros_like(problem.parameters.am).at[0].set(pressure_axis)
    if am.shape[0] > 1:
        am = am.at[1].set(-pressure_axis)
    return dataclasses.replace(
        problem.parameters,
        **coefficients,
        phiedge=phiedge,
        curtor=curtor,
        pres_scale=jnp.asarray(1.0, dtype=phiedge.dtype),
        am=am,
        ac=jnp.zeros_like(problem.parameters.ac).at[0].set(1.0),
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
    _require(np.isfinite(radius) and radius > 0, "r must be positive and finite.")
    surfaces = _validated_surfaces(qs_surfaces)
    ns, tolerances, iteration_limits = _validated_radial_controls(
        ns_array, ftol_array, ftol=ftol, max_iterations=max_iterations
    )
    if helicity_n is None:
        helicity_n = -int(np.asarray(solution.helicity))
    _require(_is_integer(helicity_m), "helicity_m must be an integer.")
    _require(_is_integer(helicity_n), "helicity_n must be an integer.")
    _require(
        np.isfinite(adjoint_tol) and adjoint_tol > 0, "adjoint_tol must be positive and finite."
    )
    _require(
        np.isfinite(toroidal_angle_tolerance) and toroidal_angle_tolerance >= 0,
        "toroidal_angle_tolerance must be nonnegative and finite.",
    )
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
    lasym = _is_asymmetric(boundary, solution, 1.0e-13)
    if lasym and surfaces:
        raise NotImplementedError(
            "VMEX's traceable quasisymmetry profile currently supports "
            "stellarator-symmetric equilibria only; pass qs_surfaces=() to "
            "solve an asymmetric equilibrium without that diagnostic."
        )

    inputs = solution.inputs
    axis = inputs.axis
    rbc, rbs, zbc, zbs = (
        np.asarray(jax.device_get(value))
        for value in (boundary.RBC, boundary.RBS, boundary.ZBC, boundary.ZBS)
    )
    pressure_axis, phiedge, curtor = map(float, _near_axis_profiles(inputs, radius))
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
        phiedge=phiedge,
        pres_scale=1.0,
        am=am,
        ncurr=1,
        ac=ac,
        curtor=curtor,
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

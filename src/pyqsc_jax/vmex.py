"""Optional differentiable fixed-boundary equilibrium interface to VMEX."""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.second_order import MU0
from pyqsc_jax.vmec import VmecBoundary, _validated_resolution, vmec_boundary

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

    def parameters_for(
        self,
        solution: NearAxisSolution,
        *,
        radius: Any | None = None,
    ) -> Any:
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
    ns_array: Any,
    ftol_array: Any | None,
    *,
    ftol: float,
    max_iterations: int,
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


def _is_asymmetric(boundary: VmecBoundary, tolerance: float = 1.0e-13) -> bool:
    return (
        max(
            float(jnp.max(jnp.abs(boundary.RBS))),
            float(jnp.max(jnp.abs(boundary.ZBC))),
        )
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
    problem: VmexProblem,
    solution: NearAxisSolution,
    *,
    radius: Any | None = None,
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
        ns_array,
        ftol_array,
        ftol=ftol,
        max_iterations=max_iterations,
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
        solution,
        ntheta=ntheta,
        mpol=mpol,
        ntor=ntor,
        newton_iterations=newton_iterations,
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
    lasym = _is_asymmetric(boundary)
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
    return dataclasses.replace(
        problem,
        parameters=vmex_parameters_from_solution(problem, solution),
    )


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
            problem.qs_surfaces,
            problem.helicity_m,
            problem.helicity_n,
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
    return VmexEquilibrium(
        solution=solution,
        quantities=_radial_quantities(problem, solution),
    )


def vmex_radial_quantities(
    problem: VmexProblem,
    parameters: Any | None = None,
) -> VmexRadialQuantities:
    """Solve and return only the differentiable VMEX diagnostic pytree."""

    return solve_vmex(problem, parameters).quantities

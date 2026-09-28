"""Differentiable near-axis quasisymmetric stellarator construction in JAX.

All computations require 64-bit floats: enable them before importing JAX
arrays, e.g. ``jax.config.update("jax_enable_x64", True)`` or
``JAX_ENABLE_X64=1``. Importing this package has no global side effects.
"""

from pyqsc_jax.diagnostics import (
    B2cOptimizationResult,
    B20Diagnostics,
    b20_diagnostics,
    mercier_diagnostics,
    optimal_B2c_value,
    optimize_B2c,
    singularity_diagnostics,
    total_field_jet,
)
from pyqsc_jax.first_order import Qsc, solve
from pyqsc_jax.geometry import Axis
from pyqsc_jax.models import (
    FieldJet,
    LinearSolveReport,
    MercierDiagnostics,
    NearAxisInputs,
    NearAxisSolution,
    RootSolveReport,
    SecondOrderData,
    SingularityDiagnostics,
    ThirdOrderData,
)
from pyqsc_jax.near_axis import near_axis
from pyqsc_jax.plasma import (
    PlasmaCurrentSource,
    PlasmaFieldData,
    PlasmaGradientData,
    PlasmaHessianData,
    plasma_current_source,
    plasma_field_on_axis,
    plasma_gradient_on_axis,
    plasma_hessian_on_axis,
    regularized_axis_integral,
)
from pyqsc_jax.second_order import second_order_residuals
from pyqsc_jax.solvers import RootSolveOptions
from pyqsc_jax.third_order import solve_third_order
from pyqsc_jax.vmec import (
    VmecBoundary,
    VmecExport,
    VmecInputParameters,
    VmexEquilibrium,
    VmexProblem,
    VmexRadialQuantities,
    solve_vmex,
    to_vmec,
    to_vmex_problem,
    uniform_cylindrical_surface,
    vmec_boundary,
    vmex_radial_quantities,
)

__all__ = [
    "Axis",
    "B20Diagnostics",
    "B2cOptimizationResult",
    "FieldJet",
    "LinearSolveReport",
    "MercierDiagnostics",
    "NearAxisInputs",
    "NearAxisSolution",
    "PlasmaCurrentSource",
    "PlasmaFieldData",
    "PlasmaGradientData",
    "PlasmaHessianData",
    "Qsc",
    "RootSolveOptions",
    "RootSolveReport",
    "SecondOrderData",
    "SingularityDiagnostics",
    "ThirdOrderData",
    "VmecBoundary",
    "VmecExport",
    "VmecInputParameters",
    "VmexEquilibrium",
    "VmexProblem",
    "VmexRadialQuantities",
    "b20_diagnostics",
    "mercier_diagnostics",
    "near_axis",
    "optimal_B2c_value",
    "optimize_B2c",
    "plasma_current_source",
    "plasma_field_on_axis",
    "plasma_gradient_on_axis",
    "plasma_hessian_on_axis",
    "regularized_axis_integral",
    "second_order_residuals",
    "singularity_diagnostics",
    "solve",
    "solve_third_order",
    "solve_vmex",
    "to_vmec",
    "to_vmex_problem",
    "total_field_jet",
    "uniform_cylindrical_surface",
    "vmec_boundary",
    "vmex_radial_quantities",
]
__version__ = "0.2.0.dev0"

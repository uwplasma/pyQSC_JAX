"""Differentiable near-axis stellarator construction in JAX."""

from pyqsc_jax.axis import Axis
from pyqsc_jax.diagnostics import mercier_diagnostics
from pyqsc_jax.field import total_field_jet
from pyqsc_jax.first_order import Qsc, solve
from pyqsc_jax.models import (
    FieldJet,
    LinearSolveReport,
    MercierDiagnostics,
    NearAxisInputs,
    NearAxisSolution,
    RootSolveReport,
    SecondOrderData,
)
from pyqsc_jax.near_axis import near_axis
from pyqsc_jax.second_order import SecondOrderResiduals, second_order_residuals
from pyqsc_jax.solvers import RootSolveOptions

__all__ = [
    "Axis",
    "FieldJet",
    "LinearSolveReport",
    "MercierDiagnostics",
    "NearAxisInputs",
    "NearAxisSolution",
    "Qsc",
    "RootSolveOptions",
    "RootSolveReport",
    "SecondOrderData",
    "SecondOrderResiduals",
    "near_axis",
    "mercier_diagnostics",
    "second_order_residuals",
    "solve",
    "total_field_jet",
]
__version__ = "0.2.0.dev0"

"""Differentiable near-axis stellarator construction in JAX."""

from pyqsc_jax.axis import Axis
from pyqsc_jax.first_order import Qsc, solve
from pyqsc_jax.models import (
    LinearSolveReport,
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
    "LinearSolveReport",
    "NearAxisInputs",
    "NearAxisSolution",
    "Qsc",
    "RootSolveOptions",
    "RootSolveReport",
    "SecondOrderData",
    "SecondOrderResiduals",
    "near_axis",
    "second_order_residuals",
    "solve",
]
__version__ = "0.2.0.dev0"

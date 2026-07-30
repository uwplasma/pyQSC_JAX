"""Differentiable near-axis stellarator construction in JAX."""

from pyqsc_jax.axis import Axis
from pyqsc_jax.first_order import Qsc, solve
from pyqsc_jax.models import NearAxisInputs, NearAxisSolution, RootSolveReport
from pyqsc_jax.near_axis import near_axis
from pyqsc_jax.solvers import RootSolveOptions

__all__ = [
    "Axis",
    "NearAxisInputs",
    "NearAxisSolution",
    "Qsc",
    "RootSolveOptions",
    "RootSolveReport",
    "near_axis",
    "solve",
]
__version__ = "0.2.0.dev0"

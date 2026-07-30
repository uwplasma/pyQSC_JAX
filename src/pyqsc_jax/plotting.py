"""Optional Matplotlib plotting helpers that return their figure objects."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from pyqsc_jax.axis import Axis, evaluate_axis
from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.plasma import PlasmaHessianData


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting requires Matplotlib. Install pyqsc-jax with the 'plot' extra."
        ) from error
    return plt


def plot_axis(
    solution_or_axis: NearAxisSolution | Axis,
    *,
    ax: Any = None,
    samples: int = 361,
    label: str | None = None,
    **plot_kwargs: Any,
):
    """Plot a full-torus magnetic axis and return ``(figure, axes)``."""

    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 4:
        raise ValueError("samples must be an integer >= 4.")
    axis = (
        solution_or_axis.inputs.axis
        if isinstance(solution_or_axis, NearAxisSolution)
        else solution_or_axis
    )
    if not isinstance(axis, Axis):
        raise TypeError("solution_or_axis must be an Axis or NearAxisSolution.")
    phi = jnp.linspace(0, 2 * jnp.pi, samples)
    axis_samples = evaluate_axis(axis, phi)
    x = axis_samples.R * jnp.cos(phi)
    y = axis_samples.R * jnp.sin(phi)
    z = axis_samples.Z

    plt = _matplotlib()
    if ax is None:
        figure = plt.figure(figsize=(5.5, 4.5))
        ax = figure.add_subplot(111, projection="3d")
    else:
        figure = ax.figure
    defaults = {"linewidth": 2.0}
    defaults.update(plot_kwargs)
    ax.plot(np.asarray(x), np.asarray(y), np.asarray(z), label=label, **defaults)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_box_aspect((1, 1, 1))
    if label is not None:
        ax.legend()
    return figure, ax


def plot_b20(
    solution: NearAxisSolution,
    *,
    ax: Any = None,
    label: str | None = None,
    **plot_kwargs: Any,
):
    """Plot the nonconstant second-order field-strength coefficient."""

    if solution.second_order is None:
        raise ValueError("B20 plotting requires an r2 or r3 solution.")
    plt = _matplotlib()
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.0, 3.6))
    else:
        figure = ax.figure
    normalized_angle = np.asarray(solution.varphi * solution.inputs.axis.nfp / (2 * jnp.pi))
    defaults = {"linewidth": 2.0}
    defaults.update(plot_kwargs)
    ax.plot(
        normalized_angle,
        np.asarray(solution.B20_anomaly),
        label=label,
        **defaults,
    )
    ax.set_xlabel("Boozer angle / field period")
    ax.set_ylabel(r"$B_{20}-\langle B_{20}\rangle$ [T/m$^2$]")
    if label is not None:
        ax.legend()
    return figure, ax


def plot_field_jet_norms(
    result: PlasmaHessianData,
    *,
    axes: Any = None,
):
    """Plot total, plasma, and external field-jet Frobenius norms."""

    plt = _matplotlib()
    if axes is None:
        figure, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
    else:
        axes = np.asarray(axes)
        if axes.shape != (3,):
            raise ValueError("axes must contain exactly three Matplotlib axes.")
        figure = axes[0].figure

    plasma_field = result.field.field.field
    external_field = result.field.external_field
    total_field = plasma_field + external_field
    plasma_gradient = result.field.gradient
    external_gradient = result.field.external_gradient
    total_gradient = plasma_gradient + external_gradient
    plasma_hessian = result.hessian
    external_hessian = result.external_hessian
    total_hessian = plasma_hessian + external_hessian
    samples = np.arange(total_field.shape[0]) / total_field.shape[0]

    tensors = (
        (total_field, plasma_field, external_field, r"$|B|$ [T]"),
        (
            total_gradient,
            plasma_gradient,
            external_gradient,
            r"$|\nabla B|_F$ [T/m]",
        ),
        (
            total_hessian,
            plasma_hessian,
            external_hessian,
            r"$|\nabla\nabla B|_F$ [T/m$^2$]",
        ),
    )
    for axis, (total, plasma, external, ylabel) in zip(axes, tensors, strict=True):
        component_axes = tuple(range(1, total.ndim))
        axis.plot(
            samples,
            np.sqrt(np.sum(np.square(np.asarray(total)), axis=component_axes)),
            label="total",
        )
        axis.plot(
            samples,
            np.sqrt(np.sum(np.square(np.asarray(plasma)), axis=component_axes)),
            label="plasma",
        )
        axis.plot(
            samples,
            np.sqrt(np.sum(np.square(np.asarray(external)), axis=component_axes)),
            label="external",
        )
        axis.set_ylabel(ylabel)
    axes[-1].set_xlabel("axis sample / field period")
    axes[0].legend(ncol=3)
    figure.tight_layout()
    return figure, axes

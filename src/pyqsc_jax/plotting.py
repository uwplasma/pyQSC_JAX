"""Optional Matplotlib plotting helpers that return their figure objects.

Matplotlib, the plasma module and the VMEC surface code are imported lazily,
so importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from pyqsc_jax.geometry import Axis, evaluate_axis
from pyqsc_jax.models import NearAxisSolution

if TYPE_CHECKING:
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
    created = ax is None
    if created:
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
    # A fresh axes is fitted to this axis; a reused one keeps everything already drawn.
    set_axes_equal(ax, *((x, y, z) if created else ()))
    if label is not None:
        ax.legend()
    plt.tight_layout()
    return figure, ax


def surface_coordinates(
    solution: NearAxisSolution, *, radius: float = 0.05, ntheta: int = 32
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return full-torus Cartesian coordinates of a near-axis surface."""

    if radius <= 0:
        raise ValueError("radius must be positive.")
    if not isinstance(ntheta, int) or isinstance(ntheta, bool) or ntheta < 4:
        raise ValueError("ntheta must be an integer >= 4.")
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)[:, None]
    phi0 = solution.phi[None, :]
    from pyqsc_jax.vmec import frenet_displacements

    X, Y, Z = frenet_displacements(solution, jnp.asarray(radius), theta, phi0)
    axis_position = jnp.stack(
        (solution.R0 * jnp.cos(solution.phi), solution.R0 * jnp.sin(solution.phi), solution.Z0),
        axis=-1,
    )
    period_position = (
        axis_position[None, :, :]
        + X[..., None] * solution.geometry.normal_cartesian[None, :, :]
        + Y[..., None] * solution.geometry.binormal_cartesian[None, :, :]
        + Z[..., None] * solution.geometry.tangent_cartesian[None, :, :]
    )
    period = 2 * jnp.pi / solution.inputs.axis.nfp
    period_angles = jnp.arange(solution.inputs.axis.nfp) * period
    cosine = jnp.cos(period_angles)[None, :, None]
    sine = jnp.sin(period_angles)[None, :, None]
    x_period = period_position[..., 0][:, None, :]
    y_period = period_position[..., 1][:, None, :]
    x = x_period * cosine - y_period * sine
    y = x_period * sine + y_period * cosine
    z = jnp.broadcast_to(period_position[..., 2][:, None, :], x.shape)
    return tuple(_close_toroidally(values.reshape(ntheta, -1)) for values in (x, y, z))


def _close_toroidally(values: jax.Array) -> jax.Array:
    return jnp.concatenate((values, values[:, :1]), axis=1)


def surface_field_strength(
    solution: NearAxisSolution, *, radius: float = 0.05, ntheta: int = 32
) -> jax.Array:
    """Near-axis ``|B|`` [T] on the :func:`surface_coordinates` grid.

    ``|B| = B0 (1 + r etabar cos(vartheta))`` at first order, plus
    ``r**2 (B20 + B2c cos(2 vartheta) + B2s sin(2 vartheta))`` for ``r2``/``r3`` solutions,
    with the helical angle ``vartheta = theta - (iota - iotaN) varphi`` at each axis node.
    The ``r**3`` field strength is not included.
    """

    inputs = solution.inputs
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)[:, None]
    helical = theta - (solution.iota - solution.iotaN) * solution.varphi[None, :]
    field_strength = inputs.B0 * (1 + radius * inputs.etabar * jnp.cos(helical))
    if solution.second_order is not None:
        field_strength = field_strength + radius**2 * (
            solution.B20[None, :]
            + inputs.B2c * jnp.cos(2 * helical)
            + inputs.B2s * jnp.sin(2 * helical)
        )
    return _close_toroidally(jnp.tile(field_strength, (1, inputs.axis.nfp)))


def plot_surface_3d(
    solution: NearAxisSolution,
    *,
    radius: float = 0.05,
    ntheta: int = 32,
    ax: Any = None,
    cmap: str = "viridis",
    alpha: float = 0.9,
    plot_axis_line: bool = True,
    color_by: str = "height",
    **surface_kwargs: Any,
):
    """Plot a full-torus near-axis surface with equal x, y, z scales; return ``(figure, axes)``.

    ``color_by="height"`` (default) colors by ``z``; ``color_by="B"`` colors by the near-axis
    ``|B|`` of :func:`surface_field_strength` and adds a colorbar when the axes are created here.
    """

    if color_by not in ("height", "B"):
        raise ValueError("color_by must be 'height' or 'B'.")
    x, y, z = surface_coordinates(solution, radius=radius, ntheta=ntheta)
    plt = _matplotlib()
    created = ax is None
    if created:
        figure = plt.figure(figsize=(6.0, 4.8))
        ax = figure.add_subplot(111, projection="3d")
    else:
        figure = ax.figure
    closed = [np.concatenate((v, v[:1]), axis=0) for v in map(np.asarray, (x, y, z))]
    defaults = {"cmap": cmap, "linewidth": 0, "antialiased": True, "alpha": alpha}
    if color_by == "B":
        from matplotlib import colormaps
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        field_strength = np.asarray(surface_field_strength(solution, radius=radius, ntheta=ntheta))
        field_strength = np.concatenate((field_strength, field_strength[:1]), axis=0)
        norm = Normalize(field_strength.min(), field_strength.max())
        colormap = colormaps[cmap]
        defaults = {**defaults, "facecolors": colormap(norm(field_strength)), "shade": False}
        del defaults["cmap"]
    defaults.update(surface_kwargs)
    ax.plot_surface(*closed, **defaults)
    if color_by == "B" and created:
        colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=colormap), ax=ax, shrink=0.7)
        colorbar.set_label("|B| [T]")
    if plot_axis_line:
        phi = jnp.linspace(0, 2 * jnp.pi, 361)
        axis_samples = evaluate_axis(solution.inputs.axis, phi)
        ax.plot(
            np.asarray(axis_samples.R * jnp.cos(phi)),
            np.asarray(axis_samples.R * jnp.sin(phi)),
            np.asarray(axis_samples.Z),
            color="black",
            linewidth=1.8,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    set_axes_equal(ax, *closed)
    plt.tight_layout()
    return figure, ax


def plot_b20(
    solution: NearAxisSolution, *, ax: Any = None, label: str | None = None, **plot_kwargs: Any
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
    scale = float(solution.R0[0]) ** 2 / float(solution.inputs.B0)
    ax.plot(normalized_angle, np.asarray(solution.B20_anomaly) * scale, label=label, **defaults)
    ax.set_xlabel("Boozer angle / field period")
    ax.set_ylabel(r"$(B_{20}-\langle B_{20}\rangle)\,R_0^2/B_0$")
    if label is not None:
        ax.legend()
    return figure, ax


def plot_field_jet_norms(
    result: PlasmaHessianData, *, axes: Any = None, B0: float = 1.0, R0: float = 1.0
):
    """Plot total, plasma and external field-jet norms in units of B0, B0/R0 and B0/R0**2."""

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
        (total_field / B0, plasma_field / B0, external_field / B0, r"$|B|/B_0$"),
        (
            total_gradient * R0 / B0,
            plasma_gradient * R0 / B0,
            external_gradient * R0 / B0,
            r"$R_0|\nabla B|_F/B_0$",
        ),
        (
            total_hessian * R0**2 / B0,
            plasma_hessian * R0**2 / B0,
            external_hessian * R0**2 / B0,
            r"$R_0^2|\nabla\nabla B|_F/B_0$",
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


def field_split_frenet_components(
    result: PlasmaHessianData, solution: NearAxisSolution
) -> jax.Array:
    """Return total/plasma/external field components in the Frenet frame.

    The result has shape ``(3, 3, nphi)``. The first index orders
    ``(total, plasma, external)`` and the second orders
    ``(tangent, normal, binormal)``.
    """

    plasma = result.field.field.field
    external = result.field.external_field
    fields = jnp.stack((plasma + external, plasma, external))
    frames = jnp.stack(
        (
            solution.geometry.tangent_cartesian,
            solution.geometry.normal_cartesian,
            solution.geometry.binormal_cartesian,
        )
    )
    return jnp.einsum("fpi,cpi->fcp", fields, frames)


def plot_field_split_components(
    result: PlasmaHessianData, solution: NearAxisSolution, *, axes: Any = None
):
    """Plot angle-dependent total/plasma/external Frenet field components."""

    plt = _matplotlib()
    if axes is None:
        figure, axes = plt.subplots(3, 1, figsize=(7.0, 7.2), sharex=True)
    else:
        axes = np.asarray(axes)
        if axes.shape != (3,):
            raise ValueError("axes must contain exactly three Matplotlib axes.")
        figure = axes[0].figure
    components = np.asarray(field_split_frenet_components(result, solution)) / float(
        solution.inputs.B0
    )
    angle = np.asarray(solution.varphi * solution.inputs.axis.nfp / (2 * jnp.pi))
    contributions = ("total", "plasma", "external")
    component_labels = (r"$B_t/B_0$", r"$B_n/B_0$", r"$B_b/B_0$")
    for component_index, (axis, ylabel) in enumerate(zip(axes, component_labels, strict=True)):
        for field_index, label in enumerate(contributions):
            axis.plot(angle, components[field_index, component_index], label=label)
        axis.set_ylabel(ylabel)
    axes[-1].set_xlabel("Boozer angle / field period")
    axes[0].legend(ncol=3)
    figure.tight_layout()
    return figure, axes


def set_axes_equal(ax, *data: Any) -> None:
    """Give a 3D axes equal physical scales: a common cube and a 1:1:1 box aspect.

    With ``data = (x, y, z)`` the cube is centred on the data's bounding box and its side is
    the largest data range; without data the current axis limits are used.
    """

    if data:
        bounds = [(float(np.min(v)), float(np.max(v))) for v in map(np.asarray, data)]
    else:
        bounds = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    half = 0.5 * max(abs(upper - lower) for lower, upper in bounds)
    for setter, (lower, upper) in zip(
        (ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d), bounds, strict=True
    ):
        middle = 0.5 * (lower + upper)
        setter(middle - half, middle + half)
    ax.set_box_aspect((1, 1, 1))

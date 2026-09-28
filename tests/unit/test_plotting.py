"""Plotting helpers: smoke tests on returned objects, axes reuse, and input guards."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

import pyqsc_jax as qsc
from fixtures import solve_configuration
from pyqsc_jax.plotting import (
    field_split_frenet_components,
    plot_axis,
    plot_b20,
    plot_field_jet_norms,
    plot_field_split_components,
    plot_surface_3d,
    surface_coordinates,
    surface_field_strength,
    surface_normal_field_error,
)

matplotlib.use("Agg")


def test_axis_and_b20_plotters_create_and_reuse_axes():
    """plot_axis/plot_b20 return (figure, axes) and draw into supplied axes."""

    solution = solve_configuration("qa", nphi=31)
    figure_axis, axis = plot_axis(solution, samples=31, label="QA")
    figure_b20, b20_axis = plot_b20(solution, label="QA")
    assert plot_axis(solution, ax=axis, samples=31) == (figure_axis, axis)
    assert plot_b20(solution, ax=b20_axis) == (figure_b20, b20_axis)
    assert len(axis.lines) == 2 and len(b20_axis.lines) == 2


def test_surface_and_field_split_plotters():
    """Surface grids close toroidally; field-jet and split-component panels draw finite lines."""

    solution = solve_configuration("plasma_stellarator", nphi=31)
    x, y, z = surface_coordinates(solution, radius=0.05, ntheta=12)
    figure, axis = plot_surface_3d(solution, radius=0.05, ntheta=12)
    assert plot_surface_3d(solution, radius=0.05, ntheta=12, ax=axis, plot_axis_line=False) == (
        figure,
        axis,
    )
    assert x.shape == y.shape == z.shape == (12, solution.inputs.axis.nfp * 31 + 1)
    assert len(axis.collections) == 2

    result = qsc.plasma_hessian_on_axis(solution, formal_radius=0.1)
    assert field_split_frenet_components(result, solution).shape == (3, 3, 31)
    _, axes = plot_field_split_components(result, solution)
    assert axes.shape == (3,) and all(len(item.lines) == 3 for item in axes)
    _, norm_axes = plot_field_jet_norms(result)
    assert all(np.isfinite(line.get_ydata()).all() for a in norm_axes for line in a.lines)
    with pytest.raises(ValueError, match="three"):
        plot_field_split_components(result, solution, axes=axes[:2])
    with pytest.raises(ValueError, match="three"):
        plot_field_jet_norms(result, axes=norm_axes[:2])


def test_3d_plots_have_equal_ranges_and_color_by_field_strength():
    """Equal 1:1:1 ranges; the default facecolors follow the near-axis |B|."""

    solution = solve_configuration("qa", nphi=31)
    _, axis = plot_axis(solution, samples=31)
    _, surface_axis = plot_surface_3d(solution, radius=0.05, ntheta=12, rstride=1, cstride=1)
    for item in (axis, surface_axis):
        limits = (item.get_xlim3d(), item.get_ylim3d(), item.get_zlim3d())
        ranges = [np.ptp(limit) for limit in limits]
        np.testing.assert_allclose(ranges, ranges[0], rtol=1e-12)
        np.testing.assert_allclose(item.get_box_aspect() / item.get_box_aspect()[0], 1.0)
    strength = np.asarray(surface_field_strength(solution, radius=0.05, ntheta=12))
    assert strength.shape == surface_coordinates(solution, radius=0.05, ntheta=12)[0].shape
    np.testing.assert_allclose(np.mean(strength), solution.inputs.B0, rtol=0.05)
    expected = matplotlib.colormaps["viridis"](
        (strength - strength.min()) / (strength.max() - strength.min())
    )
    # Each face takes the colour of its first vertex; 3-D draws depth-sort faces, so compare sets.
    faces = surface_axis.collections[0].get_facecolor()[:, :3]
    expected = expected[:, :-1, :3].reshape(-1, 3)
    np.testing.assert_allclose(np.sort(faces, axis=0), np.sort(expected, axis=0), atol=1e-12)


def test_normal_field_error_colouring():
    """A purely toroidal field is tangent to an axisymmetric surface; Bn needs a field."""

    solution = solve_configuration("plasma_dominant_channel", nphi=31)

    def toroidal(points, B0=1.0, R0=1.0):
        x, y = points[:, 0], points[:, 1]
        R2 = x**2 + y**2
        return np.stack((-B0 * R0 * y / R2, B0 * R0 * x / R2, 0 * x), axis=-1)

    error = surface_normal_field_error(solution, toroidal, radius=0.05, ntheta=24)
    assert np.max(np.abs(error)) < 1e-10
    x, y, z = surface_coordinates(solution, radius=0.05, ntheta=24)
    vertical = np.broadcast_to([0.0, 0.0, 1.0], (*np.shape(x), 3))
    assert (
        np.max(np.abs(surface_normal_field_error(solution, vertical, radius=0.05, ntheta=24))) > 0.5
    )
    figure, axis = plot_surface_3d(solution, radius=0.05, ntheta=24, color_by="Bn", field=toroidal)
    assert "B" in figure.axes[-1].get_ylabel()
    with pytest.raises(ValueError, match="needs a field"):
        plot_surface_3d(solution, color_by="Bn")
    with pytest.raises(ValueError, match="shape"):
        surface_normal_field_error(solution, vertical[:2], radius=0.05, ntheta=24)
    with pytest.raises(ValueError, match="color_by"):
        plot_surface_3d(solution, color_by="pressure")


def test_plotter_guards():
    """Plotters reject r1 solutions for B20, too few samples, non-axes and bad radii."""

    first_order = solve_configuration("qa", nphi=15, order="r1")
    with pytest.raises(ValueError, match="r2"):
        plot_b20(first_order)
    with pytest.raises(ValueError, match="integer"):
        plot_axis(first_order, samples=3)
    with pytest.raises(TypeError, match="Axis"):
        plot_axis(object())
    with pytest.raises(ValueError, match="positive"):
        surface_coordinates(first_order, radius=0)
    with pytest.raises(ValueError, match="integer"):
        surface_coordinates(first_order, ntheta=3)

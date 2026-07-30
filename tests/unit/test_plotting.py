from __future__ import annotations

import matplotlib
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.plotting import (
    field_split_frenet_components,
    plot_axis,
    plot_b20,
    plot_field_jet_norms,
    plot_field_split_components,
    plot_surface_3d,
    surface_coordinates,
)

matplotlib.use("Agg")


def test_axis_and_b20_plotters_return_objects():
    solution = qsc.solve_configuration("qa", nphi=31)
    figure_axis, axis = plot_axis(solution, samples=31, label="QA")
    figure_b20, b20_axis = plot_b20(solution, label="QA")
    reused_axis_figure, reused_axis = plot_axis(solution, ax=axis, samples=31)
    reused_b20_figure, reused_b20_axis = plot_b20(solution, ax=b20_axis)

    assert axis.figure is figure_axis
    assert b20_axis.figure is figure_b20
    assert reused_axis_figure is figure_axis
    assert reused_axis is axis
    assert reused_b20_figure is figure_b20
    assert reused_b20_axis is b20_axis
    assert len(axis.lines) == 2
    assert len(b20_axis.lines) == 2


def test_field_jet_norm_plotter_and_axes_guard():
    solution = qsc.solve_configuration(
        "finite_pressure_current",
        nphi=31,
    )
    result = qsc.plasma_hessian_on_axis(
        solution,
        formal_radius=0.05,
    )
    figure, axes = plot_field_jet_norms(result)

    assert axes.shape == (3,)
    assert all(len(axis.lines) == 3 for axis in axes)
    assert all(np.isfinite(line.get_ydata()).all() for axis in axes for line in axis.lines)
    with pytest.raises(ValueError, match="three"):
        plot_field_jet_norms(result, axes=axes[:2])


def test_surface_and_angle_dependent_field_split_plotters():
    solution = qsc.solve_configuration("plasma_stellarator", nphi=31)
    x, y, z = surface_coordinates(solution, radius=0.05, ntheta=12)
    figure, axis = plot_surface_3d(solution, radius=0.05, ntheta=12)
    reused_figure, reused_axis = plot_surface_3d(
        solution,
        radius=0.05,
        ntheta=12,
        ax=axis,
        plot_axis_line=False,
    )
    result = qsc.plasma_hessian_on_axis(solution, formal_radius=0.1)
    components = field_split_frenet_components(result, solution)
    component_figure, axes = plot_field_split_components(result, solution)

    assert x.shape == y.shape == z.shape == (12, 63)
    assert axis.figure is figure
    assert reused_figure is figure
    assert reused_axis is axis
    assert len(axis.collections) == 2
    assert components.shape == (3, 3, 31)
    assert axes.shape == (3,)
    assert component_figure is axes[0].figure
    assert all(len(item.lines) == 3 for item in axes)
    with pytest.raises(ValueError, match="three"):
        plot_field_split_components(result, solution, axes=axes[:2])


@pytest.mark.physics
@pytest.mark.parametrize(
    ("configuration", "radius"),
    (
        ("database_example_3", 0.075),
        ("b20_optimized_good", 0.075),
        ("database_large_singularity_107579", 0.15),
    ),
)
def test_database_surface_coordinates_are_finite_and_smooth(configuration, radius):
    solution = qsc.solve_configuration(configuration, nphi=121)
    x, y, z = surface_coordinates(solution, radius=radius, ntheta=36)
    points = np.stack((x, y, z), axis=-1)
    toroidal_edges = np.linalg.norm(np.diff(points, axis=1), axis=-1)

    assert np.all(np.isfinite(points))
    assert np.max(toroidal_edges) < 0.25


def test_plotter_guards():
    first_order = qsc.solve_configuration("qa", nphi=15, order="r1")
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

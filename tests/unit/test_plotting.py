from __future__ import annotations

import matplotlib
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax.plotting import plot_axis, plot_b20, plot_field_jet_norms

matplotlib.use("Agg")


def test_axis_and_b20_plotters_return_objects():
    solution = qsc.solve_configuration("qa", nphi=31)
    figure_axis, axis = plot_axis(solution, samples=31, label="QA")
    figure_b20, b20_axis = plot_b20(solution, label="QA")

    assert axis.figure is figure_axis
    assert b20_axis.figure is figure_b20
    assert len(axis.lines) == 1
    assert len(b20_axis.lines) == 1


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


def test_plotter_guards():
    first_order = qsc.solve_configuration("qa", nphi=15, order="r1")
    with pytest.raises(ValueError, match="r2"):
        plot_b20(first_order)
    with pytest.raises(ValueError, match="integer"):
        plot_axis(first_order, samples=3)
    with pytest.raises(TypeError, match="Axis"):
        plot_axis(object())

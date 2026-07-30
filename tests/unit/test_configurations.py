from __future__ import annotations

import numpy as np
import pytest

import pyqsc_jax as qsc


def test_named_configurations_are_immutable_and_solve():
    assert qsc.available_configurations() == (
        "qa",
        "qh",
        "finite_pressure_current",
        "b20_optimized_qa",
        "plasma_dominant_channel",
        "plasma_stellarator",
    )
    qa = qsc.get_configuration("qa")
    first_parameters = qa.parameters(nphi=15)
    second_parameters = qa.parameters(nphi=31)

    assert first_parameters["nphi"] == 15
    assert second_parameters["nphi"] == 31
    solution = qa.solve(nphi=15, order="r1")
    np.testing.assert_allclose(solution.inputs.etabar, qa.etabar)


def test_solve_configuration_preserves_topology_and_overrides():
    qh = qsc.solve_configuration("qh", nphi=31, order="r1")
    finite = qsc.solve_configuration(
        "finite_pressure_current",
        nphi=15,
    )

    assert int(qh.helicity) == 1
    assert finite.second_order is not None
    np.testing.assert_allclose(finite.inputs.I2, 0.9)
    np.testing.assert_allclose(finite.inputs.p2, -600000.0)


def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError, match="Available configurations"):
        qsc.get_configuration("missing")

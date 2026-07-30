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
        "database_example_3",
        "database_low_b20_57409",
        "b20_optimized_good",
        "database_large_singularity_107579",
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


@pytest.mark.physics
def test_database_showcase_configurations_are_traceable_and_screened():
    criteria = qsc.Criteria.from_curvo_2025(minimum_abs_iota=0.4)
    expected_sources = {
        "database_example_3": 3,
        "database_low_b20_57409": 57409,
        "b20_optimized_good": 57409,
        "database_large_singularity_107579": 107579,
        "plasma_stellarator": 52521,
    }

    for name, database_id in expected_sources.items():
        configuration = qsc.get_configuration(name)
        solution = configuration.solve(nphi=61)

        assert configuration.source_database_id == database_id
        assert configuration.source_url == (
            f"https://stellarator.physics.wisc.edu/app/plot/{database_id}"
        )
        assert criteria.evaluate(solution).passed
        assert abs(float(solution.iota)) >= 0.4
        if name == "plasma_stellarator":
            assert float(solution.inputs.I2) == 0.0
            assert float(solution.inputs.p2) != 0.0
            assert float(np.sqrt(np.mean(np.asarray(solution.torsion) ** 2))) > 0.5


def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError, match="Available configurations"):
        qsc.get_configuration("missing")

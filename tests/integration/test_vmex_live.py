from __future__ import annotations

import os

import jax
import numpy as np
import pytest

import pyqsc_jax as qsc

RUN_VMEX = os.environ.get("PYQSC_RUN_VMEX") == "1"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not RUN_VMEX, reason="Set PYQSC_RUN_VMEX=1 for live VMEX tests."),
]


def _problem(solution):
    return qsc.to_vmex_problem(
        solution,
        r=0.02,
        ntheta=8,
        mpol=3,
        ntor=2,
        ns_array=(7,),
        ftol=1.0e-7,
        max_iterations=1200,
        adjoint_tol=1.0e-8,
        multigrid=False,
        qs_surfaces=(0.5, 1.0),
    )


def test_vmex_vacuum_profiles_and_implicit_gradient():
    solution = qsc.solve_configuration("qa", nphi=31)
    problem = _problem(solution)
    result = problem.solve()
    quantities = result.quantities

    assert problem.vmex_version
    assert quantities.iota.shape == quantities.s.shape == (7,)
    assert quantities.quasisymmetry.shape == (2,)
    assert np.all(np.isfinite(np.asarray(quantities.iota)))
    assert np.all(np.isfinite(np.asarray(quantities.quasisymmetry)))
    np.testing.assert_allclose(quantities.iota[0], solution.iota, rtol=8.0e-3)
    assert float(quantities.thermal_energy) == pytest.approx(0.0, abs=1.0e-14)

    value, gradient = jax.value_and_grad(
        lambda parameters: (
            qsc.vmex_radial_quantities(
                problem,
                parameters,
            ).magnetic_well
        )
    )(problem.parameters)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient.rbc)))
    assert float(np.linalg.norm(np.asarray(gradient.rbc))) > 0


def test_vmex_finite_beta_profiles():
    solution = qsc.solve_configuration("plasma_stellarator", nphi=31)
    problem = _problem(solution)
    quantities = problem.quantities()

    assert problem.finite_beta
    assert float(quantities.thermal_energy) > 0
    assert np.all(np.isfinite(np.asarray(quantities.iota)))
    assert np.all(np.isfinite(np.asarray(quantities.quasisymmetry)))

    value, gradient = jax.value_and_grad(
        lambda parameters: (
            qsc.vmex_radial_quantities(
                problem,
                parameters,
            ).magnetic_well
        )
    )(problem.parameters)
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient.pres_scale))
    assert float(abs(gradient.pres_scale)) > 0
    assert np.all(np.isfinite(np.asarray(gradient.rbc)))
    assert float(np.linalg.norm(np.asarray(gradient.rbc))) > 0

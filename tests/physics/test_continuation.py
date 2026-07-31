from __future__ import annotations

import numpy as np
import pytest

import pyqsc_jax as qsc

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)


def test_pseudo_arclength_crosses_etabar_iota_fold():
    branch = qsc.continue_etabar_branch(
        axis=AXIS,
        etabar_start=-1.0,
        etabar_next=-0.98,
        num_points=15,
        nphi=31,
    )

    assert branch.complete
    assert branch.status == "complete"
    assert len(branch.solutions) == 15
    assert branch.etabar.shape == (15,)
    assert branch.iota.shape == (15,)
    assert branch.sigma.shape == (15, 31)
    assert branch.tangents.shape == (15, 2)
    assert int(np.count_nonzero(np.asarray(branch.fold_detected))) == 1
    fold_index = int(np.flatnonzero(np.asarray(branch.fold_detected))[0])
    assert fold_index == 9
    assert float(branch.response_derivative[fold_index - 1]) > 0
    assert float(branch.response_derivative[fold_index]) < 0
    assert float(branch.iota[fold_index]) > float(branch.iota[fold_index - 2])
    assert float(branch.iota[fold_index]) > float(branch.iota[fold_index + 2])
    assert np.all(np.diff(np.asarray(branch.etabar)) > 0)

    for solution in branch.solutions:
        assert bool(solution.root_report.converged)
        assert solution.root_report.residual_norm < 1.0e-12


def test_corrected_points_match_independent_forward_solves():
    branch = qsc.continue_etabar_branch(
        axis=AXIS,
        etabar_start=-0.9,
        etabar_next=-0.88,
        num_points=6,
        step_size=0.018,
        nphi=31,
    )

    for index in (2, 4, 5):
        direct = qsc.solve(axis=AXIS, etabar=branch.etabar[index], nphi=31)
        np.testing.assert_allclose(branch.iota[index], direct.iota, atol=2.0e-13)
        np.testing.assert_allclose(branch.sigma[index], direct.sigma, atol=3.0e-12)

    norms = np.linalg.norm(np.diff(np.stack((branch.etabar, branch.iota), axis=1), axis=0), axis=1)
    np.testing.assert_allclose(norms[1:], 0.018, rtol=8.0e-4)


def test_positive_branch_and_two_point_result():
    branch = qsc.continue_etabar_branch(
        axis=AXIS,
        etabar_start=0.9,
        etabar_next=0.88,
        num_points=2,
        nphi=31,
    )

    assert branch.complete
    assert np.all(np.asarray(branch.etabar) > 0)
    np.testing.assert_allclose(branch.iota[0], 0.41830690943386584, rtol=2.0e-13)


def test_continuation_reports_fixed_sign_boundary():
    branch = qsc.continue_etabar_branch(
        axis=AXIS,
        etabar_start=-0.1,
        etabar_next=-0.05,
        num_points=5,
        step_size=0.2,
        nphi=31,
    )

    assert branch.status == "branch_zero_crossing"
    assert not branch.complete
    assert len(branch.solutions) == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_points": 1}, "num_points"),
        ({"num_points": True}, "num_points"),
        ({"step_size": 0.0}, "step_size"),
        ({"fold_tolerance": -1.0}, "fold_tolerance"),
        ({"etabar_start": [-1.0]}, "scalars"),
        ({"etabar_start": 0.0}, "nonzero"),
        ({"etabar_next": 0.9}, "same etabar sign"),
        ({"etabar_next": -1.0}, "distinct"),
    ],
)
def test_continuation_input_guards(overrides, message):
    parameters = {
        "axis": AXIS,
        "etabar_start": -1.0,
        "etabar_next": -0.98,
        "num_points": 3,
        "nphi": 15,
    }
    parameters.update(overrides)
    with pytest.raises(ValueError, match=message):
        qsc.continue_etabar_branch(**parameters)

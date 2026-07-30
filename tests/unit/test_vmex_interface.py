from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
from pyqsc_jax import vmex as bridge


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FakeParams:
    rbc: jax.Array
    rbs: jax.Array
    zbc: jax.Array
    zbs: jax.Array
    phiedge: jax.Array
    pres_scale: jax.Array
    curtor: jax.Array
    am: jax.Array
    ai: jax.Array
    ac: jax.Array


class FakeVmecInput:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


class FakeImplicit:
    @staticmethod
    def params_from_input(inp, *, device=None):
        del device
        return FakeParams(
            rbc=jnp.asarray(inp.rbc),
            rbs=jnp.asarray(inp.rbs),
            zbc=jnp.asarray(inp.zbc),
            zbs=jnp.asarray(inp.zbs),
            phiedge=jnp.asarray(inp.phiedge),
            pres_scale=jnp.asarray(inp.pres_scale),
            curtor=jnp.asarray(inp.curtor),
            am=jnp.asarray(inp.am),
            ai=jnp.zeros(21),
            ac=jnp.asarray(inp.ac),
        )

    @staticmethod
    def run(_inp, params, **_kwargs):
        return SimpleNamespace(
            state=params,
            runtime=SimpleNamespace(),
            wb=jnp.sum(params.rbc**2),
            wp=params.am[0],
        )

    @staticmethod
    def iota_profile(state, _runtime):
        return jnp.linspace(0.4, 0.5, 7) + 1.0e-4 * jnp.sum(state.rbc)


class FakeQuasisymmetry:
    def __init__(self, surfaces, helicity_m, helicity_n):
        self.surfaces = jnp.asarray(surfaces)
        self.helicity = helicity_m + helicity_n

    def profile_state(self, state, _runtime):
        return self.surfaces**2 + self.helicity + 1.0e-5 * jnp.sum(state.rbc)


class FakeOptimize:
    QuasisymmetryRatioResidual = FakeQuasisymmetry

    @staticmethod
    def magnetic_well(state, _runtime):
        return 0.1 * state.am[0] + 1.0e-3 * jnp.sum(state.rbc)

    @staticmethod
    def aspect_ratio(state, _runtime):
        return 5.0 + 1.0e-3 * jnp.sum(state.rbc)

    @staticmethod
    def volume(state, _runtime):
        return jnp.abs(state.phiedge) * 10.0


@pytest.fixture
def fake_vmex(monkeypatch):
    module = SimpleNamespace(
        __version__="test",
        VmecInput=FakeVmecInput,
        implicit=FakeImplicit,
        optimize=FakeOptimize,
    )
    monkeypatch.setattr(bridge, "_import_vmex", lambda: module)
    return module


def qa_solution(*, asymmetric=False):
    return qsc.Qsc(
        rc=[1.0, 0.045],
        rs=[0.0, 0.005] if asymmetric else [0.0, 0.0],
        zc=[0.0, 0.01] if asymmetric else [0.0, 0.0],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=15,
        order="r2",
    )


def make_problem(fake_vmex, solution=None, **kwargs):
    del fake_vmex
    options = {
        "r": 0.02,
        "ntheta": 8,
        "mpol": 3,
        "ntor": 2,
        "ns_array": (7,),
        "ftol": 1.0e-7,
        "max_iterations": 200,
        "multigrid": False,
        "qs_surfaces": (0.5, 1.0),
    }
    options.update(kwargs)
    return qsc.to_vmex_problem(solution if solution is not None else qa_solution(), **options)


def test_problem_builds_without_disk_and_exposes_quantities(fake_vmex):
    problem = make_problem(fake_vmex)
    result = problem.solve()
    quantities = result.quantities

    assert problem.vmex_version == "test"
    assert problem.validated_commit == qsc.VMEX_VALIDATED_COMMIT
    assert problem.input.mpol == 4
    assert problem.boundary.RBC.shape == (5, 4)
    assert not problem.finite_beta
    assert quantities.s.shape == quantities.iota.shape == (7,)
    np.testing.assert_allclose(quantities.iota, -quantities.iota_vmec)
    assert quantities.quasisymmetry.shape == (2,)
    assert np.isfinite(float(quantities.magnetic_well))


def test_finite_beta_and_traceable_parameter_remap(fake_vmex):
    solution = qsc.solve_configuration("plasma_stellarator", nphi=15)
    problem = make_problem(fake_vmex, solution=solution)
    assert problem.finite_beta

    def objective(etabar):
        candidate = qsc.solve(
            axis=solution.inputs.axis,
            etabar=etabar,
            B0=solution.inputs.B0,
            B2c=solution.inputs.B2c,
            I2=solution.inputs.I2,
            p2=solution.inputs.p2,
            nphi=15,
            order="r2",
        )
        parameters = problem.parameters_for(candidate, radius=0.018)
        return qsc.vmex_radial_quantities(problem, parameters).magnetic_well

    value, derivative = jax.value_and_grad(objective)(solution.inputs.etabar)
    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))
    assert float(problem.quantities().thermal_energy) > 0


def test_empty_qs_surface_list_supports_asymmetric_equilibrium(fake_vmex):
    problem = make_problem(fake_vmex, solution=qa_solution(asymmetric=True), qs_surfaces=())
    assert problem.input.lasym
    assert problem.quantities().quasisymmetry.shape == (0,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"r": 0.0}, "positive"),
        ({"qs_surfaces": (0.0,)}, "0 < s"),
        ({"qs_surfaces": (0.75, 0.5)}, "increasing"),
        ({"ns_array": ()}, "nonempty"),
        ({"ns_array": (7, 7)}, "increasing"),
        ({"ftol": 0.0}, "positive"),
        ({"ftol_array": (1.0e-7, 1.0e-8)}, "one positive"),
        ({"max_iterations": 0}, "positive integer"),
        ({"helicity_m": 1.5}, "integer"),
        ({"helicity_n": 1.5}, "integer"),
    ],
)
def test_problem_validation(fake_vmex, kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_problem(fake_vmex, **kwargs)


def test_asymmetric_quasisymmetry_guard(fake_vmex):
    with pytest.raises(NotImplementedError, match="stellarator-symmetric"):
        make_problem(fake_vmex, solution=qa_solution(asymmetric=True))


def test_parameter_remap_rejects_changed_field_period(fake_vmex):
    problem = make_problem(fake_vmex)
    changed = qsc.Qsc(
        rc=[1.0, 0.02],
        zs=[0.0, -0.02],
        nfp=2,
        etabar=-0.9,
        nphi=15,
        order="r2",
    )
    with pytest.raises(ValueError, match="field periods"):
        problem.parameters_for(changed)


def test_runtime_and_import_guards(fake_vmex, monkeypatch):
    problem = make_problem(fake_vmex)
    vmex = bridge._import_vmex()
    monkeypatch.setattr(
        vmex.implicit,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            state=problem.parameters,
            runtime=None,
            wb=0.0,
            wp=0.0,
        ),
    )
    with pytest.raises(RuntimeError, match="runtime"):
        problem.solve()

    monkeypatch.undo()
    monkeypatch.setattr(
        bridge.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(ImportError, match="requires VMEX"):
        bridge._import_vmex()


def test_incomplete_vmex_api_is_rejected(monkeypatch):
    monkeypatch.setattr(
        bridge.importlib,
        "import_module",
        lambda _name: SimpleNamespace(VmecInput=FakeVmecInput),
    )
    with pytest.raises(ImportError, match="implicit, optimize"):
        bridge._import_vmex()

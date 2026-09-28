"""VMEC boundary export and the VMEX problem bridge (with an in-memory VMEX stand-in)."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyqsc_jax as qsc
import pyqsc_jax.vmec as bridge
from fixtures import solve_configuration
from pyqsc_jax.near_axis import near_axis


def qa_solution(*, nphi: int = 61, asymmetric: bool = False):
    return qsc.Qsc(
        rc=[1.0, 0.045],
        rs=[0.0, 0.005] if asymmetric else [0.0, 0.0],
        zc=[0.0, 0.01] if asymmetric else [0.0, 0.0],
        zs=[0.0, -0.045],
        nfp=3,
        etabar=-0.9,
        nphi=nphi,
        order="r2",
    )


def test_surface_matches_independent_per_point_root_solve():
    """The vectorized cylindrical surface equals the adapter's per-point toroidal-angle solve."""

    field = near_axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3, etabar=-0.9, nphi=31, order="r2")
    old_R, old_Z, old_phi0 = field.Frenet_to_cylindrical(0.01, ntheta=8)
    new_R, new_Z, new_phi0, residual = qsc.uniform_cylindrical_surface(
        field.solution, 0.01, ntheta=8
    )

    np.testing.assert_allclose(new_R, old_R, atol=4.0e-13)
    np.testing.assert_allclose(new_Z, old_Z, atol=4.0e-13)
    np.testing.assert_allclose(new_phi0, old_phi0, atol=4.0e-13)
    assert float(residual) < 2.0e-15


def test_vmec_boundary_reconstructs_the_surface():
    """The fitted (mpol, ntor) boundary reproduces R and Z of the near-axis surface to 5e-6."""

    boundary = qsc.vmec_boundary(qa_solution(), 0.03, ntheta=40, mpol=12, ntor=14)

    assert float(boundary.maximum_toroidal_angle_residual) < 2.0e-15
    assert bool(boundary.toroidal_angle_converged)
    assert float(boundary.maximum_R_reconstruction_error) < 5.0e-6
    assert float(boundary.maximum_Z_reconstruction_error) < 5.0e-6


def test_to_vmec_is_deterministic_and_writes_mpol_plus_one(tmp_path):
    """Repeated exports are byte-identical; MPOL = mpol + 1 so VMEC keeps the m = mpol mode."""

    solution = qa_solution(nphi=31)
    options = dict(
        r=0.03,
        ntheta=16,
        mpol=4,
        ntor=4,
        parameters={"ns_array": [31], "ftol_array": [1e-10], "niter_array": [3000]},
    )
    export = qsc.to_vmec(solution, tmp_path / "input.a", **options)
    qsc.to_vmec(solution, tmp_path / "input.b", **options)
    text = export.path.read_text()

    assert (
        sha256(export.path.read_bytes()).digest()
        == sha256((tmp_path / "input.b").read_bytes()).digest()
    )
    assert export.phiedge == np.pi * 0.03**2
    assert not export.lasym
    assert "  MPOL = 5\n" in text
    written = max(
        int(line.split(",")[1].split(")")[0]) for line in text.splitlines() if "RBC(" in line
    )
    assert written == 4
    assert abs(float(export.boundary.RBC[4, 4])) > 1e-14


def test_asymmetric_axis_emits_all_four_coefficient_families(tmp_path):
    """A non-stellarator-symmetric axis sets LASYM and writes RBS and ZBC."""

    export = qsc.to_vmec(
        qa_solution(nphi=31, asymmetric=True),
        tmp_path / "input.asym",
        r=0.01,
        ntheta=16,
        mpol=4,
        ntor=4,
    )
    contents = export.path.read_text(encoding="utf-8")

    assert export.lasym
    assert float(np.max(np.abs(export.boundary.RBS))) > 1.0e-4
    assert float(np.max(np.abs(export.boundary.ZBC))) > 1.0e-4
    assert "LASYM = T" in contents and "RBS(" in contents and "ZBC(" in contents


def test_first_and_third_order_exports_and_finite_beta_profiles(tmp_path):
    """r1 and r3 surfaces export; ntor_max caps ntor; I2 and p2 become CURTOR and AM."""

    first_order = solve_configuration("r1_qa", nphi=15, order="r1")
    assert np.all(np.isfinite(qsc.vmec_boundary(first_order, 0.005, ntheta=8, mpol=3, ntor=2).R))

    export = qsc.to_vmec(
        solve_configuration("qa", nphi=15, order="r3"),
        tmp_path / "input.r3",
        r=0.005,
        ntheta=10,
        mpol=4,
        ntor=4,
        ntor_max=2,
        parameters=qsc.VmecInputParameters(
            ns_array=(15,), ftol_array=(1.0e-10,), niter_array=(2000,)
        ),
    )
    assert export.boundary.ntor == 2

    solution = solve_configuration("plasma_dominant_channel", nphi=15)
    export = qsc.to_vmec(solution, tmp_path / "input.plasma", r=0.1, ntheta=10, mpol=4, ntor=2)
    contents = export.path.read_text(encoding="utf-8")
    # CURTOR = 2 pi I2 r**2 / mu0 and p(axis) = -p2 r**2 for r = 0.1.
    np.testing.assert_allclose(export.curtor, 210000.0)
    np.testing.assert_allclose(export.pressure_axis, 1000.0)
    assert "AM =" in contents and "CURTOR =" in contents


def test_legacy_to_vmec_exposes_boundary_coefficients(tmp_path):
    """The adapter's to_vmec writes a file and exposes RBC/RBS/ZBC/ZBS of shape (m, 2n+1)."""

    field = near_axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3, etabar=-0.9, nphi=31, order="r2")
    export = field.to_vmec(
        tmp_path / "input.legacy", r=0.01, params={"mpol": 4, "ntor": 4}, ntheta=16, ntorMax=4
    )
    assert export.path.exists()
    assert field.RBC.shape == field.RBS.shape == field.ZBC.shape == field.ZBS.shape == (5, 9)


def test_to_vmec_rejects_bad_controls_without_writing(tmp_path):
    """Invalid resolution, tolerances, namelist controls, or an unconverged inversion raise."""

    for kwargs, message in (
        ({"ns_array": (), "ftol_array": (), "niter_array": ()}, "nonempty"),
        ({"ns_array": (15,), "ftol_array": (1.0e-10, 1.0e-11), "niter_array": (2000,)}, "aligned"),
        ({"delt": 0.0}, "positive"),
        ({"nstep": 0}, "positive"),
        ({"tcon0": 0.0}, "positive"),
        ({"ns_array": (2,), "ftol_array": (1.0e-10,), "niter_array": (2000,)}, "at least 3"),
        ({"ns_array": (15,), "ftol_array": (0.0,), "niter_array": (2000,)}, "tolerance"),
        ({"ns_array": (15,), "ftol_array": (1.0e-10,), "niter_array": (0,)}, "iteration"),
    ):
        with pytest.raises(ValueError, match=message):
            qsc.VmecInputParameters(**kwargs)

    solution = qa_solution(nphi=15)
    for kwargs, message in (
        ({"r": 0.0}, "r must be positive"),
        ({"r": float("nan")}, "r must be positive"),
        ({"ntor_max": -1}, "ntor_max"),
        ({"coefficient_tolerance": -1.0}, "coefficient_tolerance"),
        ({"coefficient_tolerance": float("nan")}, "coefficient_tolerance"),
        ({"toroidal_angle_tolerance": -1.0}, "toroidal_angle_tolerance"),
        ({"toroidal_angle_tolerance": float("nan")}, "toroidal_angle_tolerance"),
        ({"ntheta": 7, "mpol": 3}, "ntheta"),
        ({"mpol": 0}, "positive"),
        ({"ntor": -1}, "nonnegative"),
        ({"newton_iterations": 0}, "positive"),
        ({"ntheta": 8.0}, "integers"),
        ({"ntor": 8}, "nphi"),
        ({"parameters": {"missing": 1}}, "Unknown VMEC input"),
    ):
        options = {"r": 0.01, "ntheta": 8, "mpol": 3, "ntor": 2, **kwargs}
        with pytest.raises(ValueError, match=message):
            qsc.to_vmec(solution, tmp_path / "input.invalid", **options)

    output = tmp_path / "input.unconverged"
    with pytest.raises(RuntimeError, match="no VMEC input was written"):
        qsc.to_vmec(qa_solution(), output, r=0.03, ntheta=40, mpol=12, ntor=14, newton_iterations=1)
    assert not output.exists()


# --- VMEX bridge against an in-memory stand-in -------------------------------------------


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
            state=params, runtime=SimpleNamespace(), wb=jnp.sum(params.rbc**2), wp=params.am[0]
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
        __version__="test", VmecInput=FakeVmecInput, implicit=FakeImplicit, optimize=FakeOptimize
    )
    monkeypatch.setattr(bridge, "_import_vmex", lambda: module)
    return module


def make_problem(solution=None, **kwargs):
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
        **kwargs,
    }
    return qsc.to_vmex_problem(
        solution if solution is not None else qa_solution(nphi=15), **options
    )


def test_vmex_problem_builds_in_memory_and_exposes_quantities(fake_vmex):
    """The problem needs no files; iota sign follows VMEC; QS residual has one entry per surface."""

    problem = make_problem()
    quantities = problem.solve().quantities

    assert problem.vmex_version == "test"
    assert problem.validated_commit == bridge.VMEX_VALIDATED_COMMIT
    assert problem.input.mpol == 4
    assert problem.boundary.RBC.shape == (5, 4)
    assert not problem.finite_beta
    assert quantities.s.shape == quantities.iota.shape == (7,)
    np.testing.assert_allclose(quantities.iota, -quantities.iota_vmec)
    assert quantities.quasisymmetry.shape == (2,)

    asymmetric = make_problem(qa_solution(nphi=15, asymmetric=True), qs_surfaces=())
    assert asymmetric.input.lasym
    assert asymmetric.quantities().quasisymmetry.shape == (0,)


def test_vmex_parameter_remap_is_traceable(fake_vmex):
    """parameters_for() is differentiable in etabar for a finite-beta configuration."""

    solution = solve_configuration("plasma_stellarator", nphi=15)
    problem = make_problem(solution)
    assert problem.finite_beta
    assert float(problem.quantities().thermal_energy) > 0

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
    assert np.isfinite(float(value)) and np.isfinite(float(derivative))


def test_vmex_problem_guards(fake_vmex, monkeypatch):
    """Invalid options, QS on asymmetric axes, nfp changes, and missing runtime/API raise."""

    for kwargs, message in (
        ({"r": 0.0}, "positive"),
        ({"qs_surfaces": (0.0,)}, "0 < s"),
        ({"qs_surfaces": (0.75, 0.5)}, "increasing"),
        ({"ns_array": ()}, "nonempty"),
        ({"ns_array": (7, 7)}, "increasing"),
        ({"ftol": 0.0}, "positive"),
        ({"ftol_array": (1.0e-7, 1.0e-8)}, "one positive"),
        ({"max_iterations": 0}, "positive integer"),
        ({"adjoint_tol": 0.0}, "positive and finite"),
        ({"toroidal_angle_tolerance": -1.0}, "nonnegative and finite"),
        ({"newton_iterations": 0}, "positive"),
        ({"ntheta": 7}, "ntheta"),
        ({"helicity_m": 1.5}, "integer"),
        ({"helicity_n": 1.5}, "integer"),
    ):
        with pytest.raises(ValueError, match=message):
            make_problem(**kwargs)
    with pytest.raises(NotImplementedError, match="stellarator-symmetric"):
        make_problem(qa_solution(nphi=15, asymmetric=True))
    with pytest.raises(RuntimeError, match="angle inversion did not converge"):
        make_problem(newton_iterations=1)

    problem = make_problem()
    changed = qsc.Qsc(rc=[1.0, 0.02], zs=[0.0, -0.02], nfp=2, etabar=-0.9, nphi=15, order="r2")
    with pytest.raises(ValueError, match="field periods"):
        problem.parameters_for(changed)

    monkeypatch.setattr(
        fake_vmex.implicit,
        "run",
        lambda *_a, **_k: SimpleNamespace(state=problem.parameters, runtime=None, wb=0.0, wp=0.0),
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
    monkeypatch.setattr(
        bridge.importlib, "import_module", lambda _name: SimpleNamespace(VmecInput=FakeVmecInput)
    )
    with pytest.raises(ImportError, match="implicit, optimize"):
        bridge._import_vmex()

# pyQSC_JAX

Differentiable near-axis stellarator construction and surface-free
plasma–coil field jets in JAX.

[![Tests](https://github.com/uwplasma/pyQSC_JAX/actions/workflows/tests.yml/badge.svg)](https://github.com/uwplasma/pyQSC_JAX/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/uwplasma/pyQSC_JAX/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/pyQSC_JAX)
[![PyPI](https://img.shields.io/pypi/v/pyqsc-jax.svg)](https://pypi.org/project/pyqsc-jax/)
[![Docs](https://readthedocs.org/projects/pyqsc-jax/badge/?version=latest)](https://pyqsc-jax.readthedocs.io/)
[![DOI pending](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](docs/release_checklist.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Status:** development branch for the 0.2 refactor. The physics,
> compatibility, and external 3+5+7 field-jet paths are validated; release
> hardening is still in progress.

![QA, QH, B20-optimized, and finite-current stellarators](docs/_static/stellarator_gallery.png)

Four runnable configurations: vacuum QA and QH references, the
\(B_{20}\)-optimized QA surface, and an angle-dependent finite-current
stellarator. Every displayed surface reports its transform and/or singular
radius; the plotting radius is recorded in the figure metadata.

| \(B_{20}\) optimization, geometry, and singular margin | Angle-dependent plasma/external field |
| --- | --- |
| ![B20 optimization](docs/_static/B20_optimization.png) | ![Plasma and external jet](docs/_static/plasma_external_jet.png) |

| Differentiable radial VMEX quantities | VMEC2000 export validation |
| --- | --- |
| ![VMEX radial quantities](docs/_static/vmex_radial_profiles.png) | ![VMEC validation](docs/_static/vmec_validation.png) |

## Install

```bash
python -m pip install pyqsc-jax
```

Install the JAX build appropriate for your CPU or accelerator. pyQSC_JAX does
not configure a backend or enable 64-bit mode at import time.

For 3D plots and differentiable finite-radius equilibria:

```bash
python -m pip install 'pyqsc-jax[plot,vmex]'
```

## Quickstart

```python
import pyqsc_jax as qsc

solution = qsc.Qsc(
    rc=[1.0, 0.155, 0.0102],
    zs=[0.0, 0.154, 0.0111],
    nfp=2,
    etabar=0.64,
    B2c=-0.00322,
    nphi=61,
    order="r2",
)

assert solution.root_report.converged
assert solution.linear_report.converged
print("iota:", solution.iota)
print("B20 residual:", solution.B20_residual)
print("Mercier D r^2:", solution.DMerc_times_r2)
print("singular radius:", solution.r_singularity)
print("minimum L_grad_grad_B:", solution.L_grad_grad_B.min())
```

The result is an immutable JAX pytree. Canonical field arrays use
sample-first ordering: `(nphi, 3)`, `(nphi, 3, 3)`, and
`(nphi, 3, 3, 3)`.

## Pick a workflow

| Goal | Start here | Main result |
| --- | --- | --- |
| Construct QA/QH near the axis | `qsc.Qsc(...)` | immutable near-axis solution |
| Flatten \(B_{20}\) | `optimize_B2c`, `search_axis` | verified dense-grid diagnostics |
| Inspect a full 3D surface | `plot_surface_3d` | figure plus explicit plotting radius |
| Compute radial MHD quantities | `to_vmex_problem(...).solve()` | differentiable \(\iota(s)\), QS profile, magnetic well |
| Match finite-beta coils | `plasma_hessian_on_axis` | external vacuum 3+5+7 target |
| Run conventional VMEC2000 | `to_vmec` | diagnosed `input.*` file |

## Capabilities

| Capability | Status |
| --- | --- |
| General Fourier axes, QA/QH topology, spectral Frenet geometry | Validated |
| Converged periodic sigma solve with implicit JVP/VJP | Validated |
| Complete finite-pressure/current r2 and total field Hessian | Validated |
| r3 boundary correction and magnetic shear | Validated documented specialization |
| Mercier, singular radius, scale lengths, \(B_{20}\) spectrum | Validated |
| Target-\(\iota\) inverse solve and pseudo-arclength folds | Validated |
| Exact \(B_{2c}\), criteria, bounded multistart axis search | Validated |
| Surface-free plasma field, gradient, Hessian | Validated asymptotic model |
| External vacuum 3+5+7 field-jet target | Validated |
| Fast fixed-boundary VMEC export, including asymmetric boundaries | Validated against VMEC 9.0 |
| Differentiable VMEX radial iota, QS profile, magnetic well | Validated in vacuum and finite beta |
| ESSOS vacuum/finite-beta stage-two and single-stage objectives | Draft integration PR |
| Legacy `pyqsc_jax.near_axis.near_axis` contract | Preserved |

## Headline regression cases

These are checked results, not illustrative targets. The detailed optimizer,
resolution, timing, and radius-convergence tables are in the
[B20](docs/theory/b20-optimization.md),
[plasma-field](docs/validation/plasma_field.md), and
[VMEC](docs/validation/vmec.md) validation pages.

| Check | Result | Regression gate |
| --- | ---: | ---: |
| Optimized QA \(B_{20}\), weighted \(L^2\), `nphi=121` | \(1.5901\times10^{-6}\) | \(<1.6\times10^{-6}\) |
| Improvement over stock-axis exact-\(B_{2c}\) solve | \(26{,}053\times\) | \(>25{,}000\times\) |
| \(B_{20}\)-optimized singular radius | 0.248 m | displayed \(r=0.075\) m gives 3.3× clearance |
| Angle-dependent minimum \(\lvert B_\mathrm{plasma}\rvert/\lvert B_\mathrm{total}\rvert\) | 32.70% | \(>30\%\) |
| Plasma-field norm peak-to-peak / mean | 3.53% | \(>3\%\) |
| VMEC/near-axis on-axis \(\iota\), \(r=0.0025\) | 0.418543 / 0.418307 | relative error \(<0.1\%\) |
| VMEC force residuals | \(2.4\)–\(7.6\times10^{-11}\) | maximum \(<10^{-9}\) |
| `to_vmec`, Apple M4 CPU, compile+execute / warm | 0.476 s / 5.20 ms | warm unit gate \(<0.5\) s |

```python
optimized = qsc.solve_configuration("b20_optimized_qa", nphi=121)
diagnostics = qsc.b20_diagnostics(optimized)
print(diagnostics.weighted_l2, diagnostics.grid_maximum)

plasma_case = qsc.solve_configuration("plasma_stellarator", nphi=61)
plasma = qsc.plasma_field_on_axis(plasma_case, formal_radius=0.2)
```

## Target rotational transform

```python
axis = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
inverse = qsc.solve(
    axis=axis,
    iota=0.42,
    etabar=-1.0,
    solve_for="etabar",
)
print(inverse.inputs.etabar, inverse.response_derivative, inverse.branch_fold)
```

The `etabar` sign and seed select a local branch. Use
`continue_etabar_branch` when the response approaches a fold.

## Axis optimization

```python
indices = qsc.stellarator_symmetric_variable_indices(axis, modes=(1,))
problem = qsc.AxisSearchProblem(
    axis=axis,
    variable_indices=indices,
    lower_bounds=[0.02, -0.08],
    upper_bounds=[0.08, -0.02],
    etabar=-0.9,
)
result = qsc.search_axis(problem)
print(result.status, result.search_budget, result.distinct_basins)
```

Only `verified_zero` certifies the known zero lower bound of the nonnegative
primary \(B_{20}\) residual. A nonzero result is `best_found`, never a
mathematical global-minimum claim.

## The \(B_{20}\)-optimized stellarator in 3D

```python
from pyqsc_jax.plotting import plot_surface_3d

optimized = qsc.solve_configuration("b20_optimized_qa", nphi=121)
figure, axis = plot_surface_3d(optimized, radius=0.075)

print(optimized.B20_residual)   # 1.5901e-6 T/m^2
print(optimized.r_singularity) # 0.248 m
```

The displayed surface is therefore 3.3 times inside the computed singular
radius. The optimizer comparison, Fourier-resolution audit, full coefficient
set, and the distinction between `best_found` and a certified zero are in the
[B20 optimization notes](docs/theory/b20-optimization.md).

## Finite-beta coil targets

```python
solution = qsc.solve_configuration("plasma_stellarator", nphi=61)
target = qsc.plasma_hessian_on_axis(solution, formal_radius=0.2)

print(target.field.external_field.shape)                 # (nphi, 3)
print(target.field.external_gradient_independent.shape) # (nphi, 5)
print(target.external_hessian_independent.shape)         # (nphi, 7)
```

ESSOS consumes these external vacuum targets for normalized field, gradient,
and Hessian coil objectives. The dependency remains one-way:
`ESSOS -> pyQSC_JAX`.

The public plot uses Frenet components rather than only \(|B|\). This makes
the angle dependence and exact cancellation of plasma and external transverse
components visible; \(|B_\mathrm{total}|=B_0\) is constant on axis by
construction.

## Differentiable radial equilibria with VMEX

```python
import jax

near_axis = qsc.solve_configuration("qa", nphi=61)
problem = qsc.to_vmex_problem(
    near_axis,
    r=0.02,
    qs_surfaces=(0.25, 0.5, 0.75, 1.0),
)
equilibrium = problem.solve()

print(equilibrium.quantities.iota)          # full radial profile
print(equilibrium.quantities.quasisymmetry) # requested flux surfaces
print(equilibrium.quantities.magnetic_well)

well, gradient = jax.value_and_grad(
    lambda parameters: qsc.vmex_radial_quantities(
        problem, parameters
    ).magnetic_well
)(problem.parameters)
print(gradient.rbc, gradient.pres_scale)
```

`problem.parameters_for(new_near_axis_solution)` traceably rebuilds the
boundary, flux, pressure, and current leaves, allowing derivatives to
propagate from pyQSC_JAX variables through VMEX's converged fixed point.
Vacuum and scalar finite-beta fixed-boundary paths are tested against
[uwplasma/VMEX](https://github.com/uwplasma/vmex); the exact validated commit
and current limitations are recorded in the
[VMEX integration validation](docs/validation/vmex_interface.md).

## VMEC export

```python
solution = qsc.Qsc(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
    nphi=61,
    order="r2",
)
export = qsc.to_vmec(solution, "input.qa", r=0.005)
print(export.conversion_seconds)
print(export.boundary.maximum_R_reconstruction_error)
```

The exporter uses a vectorized Newton inversion and a two-dimensional FFT,
supports all four VMEC boundary coefficient families, and returns conversion
diagnostics. The committed `wout` regression checks the resulting equilibrium,
including the on-axis transform; an opt-in test can rerun a local VMEC
executable.

## Learn and validate

- [Installation and model selection](docs/getting_started/)
- [Theory and conventions](docs/theory/)
- [Executable tutorials](docs/tutorials/)
- [API reference](docs/api/)
- [pyQSC, literature, plasma, and ESSOS validation](docs/validation/)
- [VMEC conversion, equilibrium, and performance validation](docs/validation/vmec.md)
- [Differentiable VMEX radial-equilibrium validation](docs/validation/vmex_interface.md)
- [Migration from pyQSC and the legacy adapter](docs/migration.md)
- [Known limitations](docs/advanced/limitations.md)

The [draft pyQSC_JAX PR](https://github.com/uwplasma/pyQSC_JAX/pull/2) and
[draft ESSOS integration PR](https://github.com/uwplasma/ESSOS/pull/46)
record the active review state.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). A Zenodo DOI will be
minted as part of the first reviewed release; the badge remains explicitly
pending until then. pyQSC_JAX is MIT licensed. Adapted pyQSC source and
reference data retain BSD-2-Clause attribution in [NOTICE](NOTICE).

# pyQSC_JAX

Differentiable near-axis stellarator construction and plasma–coil field jets in
JAX.

> **Development status:** the immutable first-order core, complete r2
> coefficient solve, r3 flux constraint, magnetic shear, total-field Hessian,
> Mercier diagnostics, singular-radius diagnostics, and ESSOS adapter are
> validated, along with branch-local inverse solves and pseudo-arclength
> continuation, exact affine \(B_{2c}\) optimization, and the scalable Curvo
> et al. screening profile. Deterministic bounded multistart axis optimization
> is also validated. Plasma/external field jets remain under development on the
> `refactor/pyqsc-jax-complete` branch.

## Install

```bash
python -m pip install pyqsc-jax
```

For a source checkout:

```bash
python -m pip install -e .
```

No JAX backend policy is imposed by the package. Install the JAX build suitable
for your accelerator and platform.

## Quickstart

```python
import pyqsc_jax as qsc

solution = qsc.Qsc(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
    order="r1",
)

print("iota:", solution.iota)
print("sigma residual:", solution.root_report.residual_norm)
print("axis length:", solution.axis_length)
print("minimum L_grad_B:", solution.L_grad_B.min())
```

`solution` is an immutable JAX pytree. Vector samples use shape `(nphi, 3)`,
and tensor samples use `(nphi, 3, 3)`. JIT, VMAP, JVP, and VJP act on explicit
array arguments, while the nonlinear sigma solution is differentiated
implicitly at its converged root.

The supported ESSOS compatibility API remains:

```python
from pyqsc_jax.near_axis import near_axis

field = near_axis(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
)
```

The adapter preserves historical component-axis ordering and mutable
`field.x`/`field.dofs` behavior for ESSOS, but delegates its physics to the
immutable core.

A finite-pressure/current second-order solution uses the same entry point:

```python
solution = qsc.Qsc(
    rc=[1.0, 0.09],
    zs=[0.0, -0.09],
    nfp=2,
    etabar=0.95,
    I2=0.9,
    p2=-600000.0,
    B2c=-0.7,
    order="r2",
)

print("B20 residual:", solution.B20_residual)
print("linear condition:", solution.linear_report.matrix_condition_number)
print("Mercier:", solution.DMerc_times_r2)
print("Hessian inverse scale:", solution.grad_grad_B_inverse_scale_length)
print("singular radius:", solution.r_singularity)
```

Use `order="r3"` to add the differentiable flux-constraint correction and its
untwisted boundary coefficients. The result reports independent
`flux_constraint_residual` and `consistency_error` checks.
`qsc.solve_magnetic_shear(solution, B31c=0.0)` then attaches the
order-\(r^2\) rotational-transform correction as `solution.iota2`.

Prescribe transform with
`qsc.solve(axis=axis, iota=target, etabar=seed, solve_for="etabar")`, or hold
`etabar` fixed and use `solve_for="I2"`. The inverse result reports its local
response derivative and fold flag. `qsc.continue_etabar_branch(...)` traces
the full sigma collocation state across such a fold.

`qsc.optimize_B2c(solution)` eliminates the affine \(B_{2c}\) subproblem
exactly and returns dense weighted, smooth-maximum, grid-maximum, Fourier, and
tail diagnostics. `qsc.verify_B20_resolution(...)` independently rebuilds the
candidate on doubled and quadrupled grids.

`qsc.Criteria.from_curvo_2025(major_radius=1.0, B0=1.0)` evaluates the
published Table 3 screening profile and returns every measured value, signed
margin, and pass flag. The thresholds are normalization-aware and fully
overridable; the profile is not treated as a universal feasibility theorem.

`qsc.search_axis(...)` combines deterministic bounded exploration, full-JAX
residual Jacobians, damped local refinement, basin clustering, exact
\(B_{2c}\) elimination, criteria checks, and independent resolution
verification. It returns `verified_zero` only for a verified zero of the
nonnegative primary residual; every nonzero candidate is `best_found` without
a global claim.

`qsc.plasma_current_source(solution, formal_radius=a)` constructs the regular
positive-volume current measure used by the surface-free Biot–Savart
calculation and reports both \(I_2\) and enclosed amperes. A positive formal
radius is mandatory even though no finite-radius surface is created.

The lower-level immutable axis API supports general, not necessarily
stellarator-symmetric Fourier axes:

```python
import pyqsc_jax as qsc
from pyqsc_jax.geometry import compute_axis_geometry

axis = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
geometry = compute_axis_geometry(axis, nphi=31)

print("axis length:", geometry.axis_length)
print("minimum curvature:", geometry.diagnostics.minimum_curvature)
```

## Scope

The completed package will provide:

- general Fourier magnetic axes and QA/QH topology;
- converged differentiable first-order sigma solves;
- second-order field Hessians/diagnostics and complete pyQSC third-order parity;
- total, plasma, and external field/gradient/Hessian jets on the axis;
- target-transform inverse solves with continuation and fold detection;
- dense-grid `B20` diagnostics and multistart axis optimization;
- cited stellarator screening criteria with explicit margins;
- an unchanged ESSOS compatibility adapter.

See the [draft refactor PR](https://github.com/uwplasma/pyQSC_JAX/pull/2),
[architecture decisions](docs/adr/), and
[physics traceability](docs/development/physics-traceability.md).

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). pyQSC_JAX is MIT
licensed. Adapted pyQSC source and reference data retain BSD-2-Clause
attribution in [NOTICE](NOTICE).

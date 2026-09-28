---
html_theme.sidebar_secondary.remove: true
---

# pyQSC_JAX

<p class="lead" style="text-align:center; max-width: 46rem; margin: 1rem auto;">
Differentiable near-axis construction of quasisymmetric stellarators in JAX, with the
on-axis field jet and its separation into a plasma part and an external (coil) target.
</p>

```{figure} _static/figures/overview.png
:width: 55%
:alt: A four-period finite-pressure stellarator surface constructed near the axis

Stellarator-database configuration 52521 {cite}`curvo2025deep`: four field periods, finite
pressure, zero on-axis current, surface drawn at $r =$ {{ overview_radius }} m
(singular radius {{ db_r_singularity }} m).
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc
Install with 64-bit JAX and run a first solve.
:::

:::{grid-item-card} Theory
:link: theory/index
:link-type: doc
Conventions, the first-, second- and third-order equations, the field jet, the plasma
field and the VMEC export, with every algorithm.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc
Solutions and inputs, lazy diagnostics, plotting, derivatives and optimization, VMEX and
ESSOS.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc
One page per script in `examples/`, each with its figure and output.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
The public names exported by `pyqsc_jax`.
:::

:::{grid-item-card} Development
:link: development/index
:link-type: doc
Tests, validation numbers and the changelog.
:::
::::

## What the code does

Given a closed magnetic axis and a few scalars, pyQSC_JAX solves the Garren-Boozer
near-axis equations {cite}`garren1991magnetic,landreman2019numerical,landreman2019highorder`
for a quasisymmetric magnetic field expanded in the distance $r$ from the axis.

| order | content |
|---|---|
| first ($r$) | periodic $\sigma$ equation for the rotational transform $\iota$ and the elliptical cross-section |
| second ($r^2$) | complete finite-pressure, finite-current system for $X_{20}, Y_{20}$, the second harmonics and $B_{20}$ |
| third ($r^3$) | the pyQSC-compatible flux-constraint correction to the boundary |
| diagnostics | on-axis field, gradient and Hessian; singular radius; Mercier terms; $B_{20}$ spectrum |
| plasma field | matched free-space field of the plasma current, its gradient and Hessian, and the external vacuum target $3+5+7$ |
| export | VMEC `&INDATA` file and an in-memory, differentiable VMEX problem |

* Every solution is an immutable JAX pytree. `solve` is compiled with `jax.jit`, and a new
  value of a scalar input does not recompile it.
* The nonlinear and linear solves are differentiated through the implicit-function theorem
  at the converged solution. A failed solve returns NaN derivatives, never a wrong one.
* All computations use 64-bit floats.

## Quick look

```python
import jax
jax.config.update("jax_enable_x64", True)
import pyqsc_jax as qsc

solution = qsc.Qsc(rc=[1.0, 0.155, 0.0102], zs=[0.0, 0.154, 0.0111], nfp=2,
                   etabar=0.64, B2c=-0.00322, order="r2", nphi=61)
print(solution.iota, solution.B20_variation, solution.r_singularity)

# derivative of the B20 variation with respect to etabar, through both implicit solves
axis = solution.axis
d = jax.grad(lambda e: qsc.solve(axis=axis, etabar=e, B2c=-0.00322,
                                 order=2, nphi=61).B20_variation)(0.64)
```

This is section 5.1 of {cite:t}`landreman2019highorder`: $\iota =$ {{ ls51_iota }},
$\max B_{20} - \min B_{20} =$ {{ ls51_B20_variation }} T/m², $r_\mathrm{sing} =$ {{ ls51_r_singularity }} m.

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
theory/index
user_guide/index
examples/index
api
development/index
```

# Differentiation and optimization

## What is differentiated

A solve contains two implicit steps: the Newton solve of the sigma equation and the dense
second-order linear system. Both are wrapped by [SOLVAX](https://github.com/uwplasma/SOLVAX):
`root_solve` for the first and `linear_solve` for the second. Their derivatives are those
of the converged equations (implicit-function theorem), not of the iterations, so the
cost of a derivative does not grow with the number of Newton steps. One LU factorization
serves the primal and the transposed (reverse-mode) second-order solve. Everything else
(geometry, spectral derivatives, diagnostics, plasma field, VMEC boundary) is ordinary JAX
code. Forward and reverse mode both work:

```python
import jax
import pyqsc_jax as qsc

axis = qsc.Axis(rc=[1.0, 0.155, 0.0102], zs=[0.0, 0.154, 0.0111], nfp=2)

def objective(etabar, B2c):
    solution = qsc.solve(axis=axis, etabar=etabar, B2c=B2c, order=2, nphi=61)
    return solution.B20_variation + 0.1 * (solution.iota + 0.42) ** 2

value, gradient = jax.value_and_grad(objective, argnums=(0, 1))(0.64, -0.00322)
```

Axis coefficients are differentiable leaves of `Axis`, so
`jax.grad(lambda a: qsc.solve(axis=a, ...).iota)(axis)` returns an `Axis` of derivatives.
The derivative of the $B_{20}$ variation with respect to $\bar\eta$ matches a centred
difference to {{ ad_B20_variation_relative_error }}, and that of $r_\mathrm{sing}$ to
{{ ad_r_singularity_relative_error }} ({doc}`../examples/12_autodiff_check`).

## When a derivative is valid

An implicit derivative exists only at a converged, regular solution. pyQSC_JAX enforces
this instead of documenting it:

* **sigma solve**: tangents and cotangents of the root are multiplied by NaN unless
  `root_report.converged` is true and the Jacobian condition estimate is finite;
* **second-order solve**: the same, unless `linear_report.converged` (finite solution,
  relative residual $\le 10^{-11}$).

A failed solve therefore gives NaN gradients, never a plausible wrong one. With
`RootSolveOptions(max_steps=1)` the section 5.1 axis gives `converged` {{ truncated_converged }}
and $d\iota/d\bar\eta$ = {{ truncated_tangent }}. `well_conditioned` is reported but does
not gate derivatives; a large condition estimate means a large, but correct, derivative.

Some outputs are not smooth everywhere:

* `r_singularity` is the minimum over the grid; its derivative is that of the active
  point and jumps when the minimizing point changes.
* `B20_variation` is a max minus a min, with the same kink; `B20_residual` and
  `b20_diagnostics(...).weighted_l2` are smooth alternatives.
* The plasma gradient switches order at `I2 == 0` ({doc}`../theory/plasma_field`); do not
  differentiate with respect to $I_2$ there.
* At a fold of $\iota(\bar\eta)$, $d\iota/d\bar\eta = 0$ and an inverse solve for
  $\bar\eta$ is singular.

## Compilation and retracing

`solve` is a `jax.jit` function with `nphi`, `order`, `sG`, `spsi`, `root_options` and
`diagnostics` static, and with `nfp` and the number of axis modes part of the input
structure. Scalars are converted to float arrays before the call, so

* a new value of `etabar`, `B0`, `sigma0`, `I2`, `p2`, `B2c`, `B2s` or of an axis
  coefficient reuses the compiled solve;
* a new `nphi`, `order`, sign, option set, `nfp` or number of Fourier modes compiles again.

On an Apple arm64 CPU a second-order solve at $n_\phi = 61$ compiles in about
{{ time_r2_compile_s }} s and then takes {{ time_r2_warm_ms }} ms
({{ time_r2_newvalue_ms }} ms with a new input value); a jitted gradient of the $B_{20}$
variation compiles in {{ time_grad_compile_s }} s and runs in {{ time_grad_warm_ms }} ms.
These numbers are machine-specific. Wrap the whole objective in `jax.jit` so that the
diagnostics, plasma field and objective arithmetic are compiled together with the solve.

`jax.vmap` over traced inputs batches solves:

```python
etabars = jnp.linspace(0.5, 0.8, 16)
iotas = jax.vmap(lambda e: qsc.solve(axis=axis, etabar=e, nphi=61).iota)(etabars)
```

## Optimization

There is no optimizer framework in the package; a few lines of SOLVAX or SciPy are enough.

**Target $\iota$ (Newton on one scalar).** {doc}`../examples/04_target_iota` solves
$\iota(\bar\eta) = \iota_\star$ with `jax.value_and_grad` and a Newton update. Pass the
residual to `solvax.root_solve` instead to get an implicit derivative of the solved
$\bar\eta$ with respect to the target.

**Exact $B_{2c}$.** Inside an objective, eliminate $B_{2c}$ with
{func}`~pyqsc_jax.optimal_B2c_value` (two linear solves, differentiable) before
evaluating $B_{20}$ ({doc}`../examples/05_optimal_B2c`).

**Least squares on the axis.** Use `scipy.optimize.least_squares` with a jitted residual
and `jax.jacfwd` Jacobian:

```python
import numpy as np, scipy.optimize

def residual(x):
    axis = qsc.Axis(rc=jnp.concatenate([jnp.ones(1), x[:3]]),
                    zs=jnp.concatenate([jnp.zeros(1), x[3:]]), nfp=4)
    solution = qsc.solve(axis=axis, etabar=-1.33, p2=-23501.281, order=2, nphi=61)
    solution = qsc.solve(axis=axis, etabar=-1.33, p2=-23501.281, order=2, nphi=61,
                         B2c=qsc.optimal_B2c_value(solution))
    return qsc.b20_diagnostics(solution).anomaly / solution.inputs.B0

x0 = np.array([-0.517, -0.0095, -0.0059, -0.542, -0.0122, -0.0059])  # database 57409
fun, jac = jax.jit(residual), jax.jit(jax.jacfwd(residual))
result = scipy.optimize.least_squares(lambda x: np.asarray(fun(x)),
                                      x0, jac=lambda x: np.asarray(jac(x)))
```

Check `root_report.converged` at the optimum and verify the result on a finer grid
(`nphi` doubled); a small residual on one grid is not enough. {doc}`../examples/06_optimize_axis_B20`
compares a database axis with one refined this way.

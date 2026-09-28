# Solutions and parameters

## Inputs

{func}`pyqsc_jax.solve` takes an {class}`~pyqsc_jax.Axis` and keyword scalars;
{func}`pyqsc_jax.Qsc` takes `rc, zs` (and optionally `rs, zc`) and `nfp` directly.

| argument | default | meaning | kind |
|---|---|---|---|
| `axis` / `rc, rs, zc, zs, nfp` | | axis Fourier series (m) | coefficients traced, `nfp` static |
| `etabar` | required | first-order field strength $\bar\eta$ (1/m) | traced |
| `B0` | 1 | on-axis field (T) | traced |
| `sigma0` | 0 | $\sigma(0)$ | traced |
| `I2` | 0 | current coefficient (T/m) | traced |
| `p2` | 0 | pressure coefficient (Pa/m²) | traced |
| `B2c`, `B2s` | 0 | second-order field harmonics (T/m²) | traced |
| `nphi` | 61 | grid points per field period | static |
| `order` | 1 | `1`/`"r1"`, `2`/`"r2"`, `3`/`"r3"` | static |
| `sG`, `spsi` | 1 | signs of $G_0$ and of the flux | static |
| `root_options` | defaults | {class}`~pyqsc_jax.RootSolveOptions` | static |
| `diagnostics` | `False` | store diagnostics in the result | static |

Traced inputs are converted to float arrays, so a new value reuses the compiled function.
A new value of a static argument, a different `nfp` or a different number of axis modes
compiles a new one ({doc}`differentiation`). `solve` validates its inputs eagerly and
raises `ValueError`/`TypeError` for bad shapes, signs or orders.

## The solution object

The result is an immutable {class}`~pyqsc_jax.NearAxisSolution`, a JAX pytree that can be
returned from `jax.jit` and batched with `jax.vmap`. To change an input, solve again.

| group | attributes |
|---|---|
| inputs and geometry | `inputs` ({class}`~pyqsc_jax.NearAxisInputs`), `geometry` (Frenet frame, `curvature`, `torsion`, `varphi`, `d_d_varphi`, `abs_G0_over_B0`, `diagnostics`), shortcuts `axis`, `phi`, `varphi`, `R0`, `Z0`, `curvature`, `torsion`, `axis_length` |
| first order | `root_report`, `sigma`, `iota`, `iotaN`, `helicity`, `G0`, `X1s`, `X1c`, `Y1s`, `Y1c` and `*_untwisted`, `elongation`, `mean_elongation`, `B_axis`, `grad_B_axis` (and `*_cylindrical`), `L_grad_B` |
| second order | `second_order` ({class}`~pyqsc_jax.SecondOrderData`): `linear_report`, `X20` ... `Z2c`, `beta_1s`, `G2`, `B20`, `B20_mean`, `B20_anomaly`, `B20_residual`, `B20_variation`, derivatives and untwisted forms |
| third order | `third_order` ({class}`~pyqsc_jax.ThirdOrderData`): `X3c1`, `Y3s1`, `Y3c1`, ..., `flux_constraint_residual`, `consistency_error` |
| diagnostics | `r_singularity` (and `*_vs_varphi`), `grad_grad_B_axis`, `L_grad_grad_B`, `DMerc_times_r2`, `DWell_times_r2`, `DGeod_times_r2`, `d2_volume_d_psi2` |

Second- and third-order coefficients can be read directly from the solution
(`solution.X20` is `solution.second_order.X20`). Asking a first-order solution for a
second-order name raises `AttributeError`; nothing is returned stale or half-initialized.

## Checking a result

A solve never raises on numerical failure. Check

```python
assert solution.root_report.converged                 # sigma equation
assert solution.geometry.diagnostics.frenet_valid     # nonzero curvature everywhere
assert solution.linear_report.converged               # order >= 2
assert solution.linear_report.well_conditioned        # condition estimate <= 1e12
```

and, for anything that depends on the grid, solve again with twice the `nphi` and
compare. $B_{20}$, the Hessian and the plasma field need more points than $\iota$; the
{doc}`../theory/second_order` figure shows typical rates. `root_report` and
`linear_report` hold residuals, iteration counts and condition estimates.

## Solver options

```python
options = qsc.RootSolveOptions(atol=1e-12, rtol=1e-12, max_steps=40)
solution = qsc.solve(axis=axis, etabar=-0.9, root_options=options)
```

`exact_condition_number=True` adds the SVD condition number of the final Jacobian to the
report.

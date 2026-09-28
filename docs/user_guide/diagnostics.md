# Diagnostics

## Computed on demand

At second order, the Mercier terms, the total field jet and the singular radius are not
computed by `solve`. They are computed when first read:

```python
solution = qsc.Qsc(..., order="r2")
solution.r_singularity        # runs singularity_diagnostics(solution)
solution.grad_grad_B_axis     # runs total_field_jet(solution)
solution.DMerc_times_r2       # runs mercier_diagnostics(solution)
```

The result of an attribute access is not cached (the solution is immutable), so reading
the same diagnostic twice computes it twice. Inside an objective only the diagnostics that
are read are traced, and a plain forward solve does not pay for them.

## Stored diagnostics

`solve(..., diagnostics=True)` computes all three inside the compiled solve and stores
them in `solution.mercier`, `solution.field_jet` and `solution.singularity`. It also
switches on the exact (SVD) condition numbers of both reports. Use it when the same
solution is queried repeatedly, or when a solution with diagnostics must leave `jax.jit`
or `jax.vmap` as a pytree. `solution.with_diagnostics()` does the same for an existing
solution.

## The diagnostic functions

| function | returns | page |
|---|---|---|
| {func}`~pyqsc_jax.total_field_jet` | {class}`~pyqsc_jax.FieldJet`: field, gradient, Hessian, Maxwell residuals, $L_{\nabla\nabla B}$ | {doc}`../theory/field_jet` |
| {func}`~pyqsc_jax.singularity_diagnostics` | {class}`~pyqsc_jax.SingularityDiagnostics`: $r_\mathrm{sing}(\varphi)$, angle, residual, determinant coefficients | {doc}`../theory/field_jet` |
| {func}`~pyqsc_jax.mercier_diagnostics` | {class}`~pyqsc_jax.MercierDiagnostics` | {doc}`../theory/field_jet` |
| {func}`~pyqsc_jax.b20_diagnostics` | {class}`~pyqsc_jax.B20Diagnostics`: norms, spectrum, tail ratio | {doc}`../theory/field_jet` |
| {func}`~pyqsc_jax.optimize_B2c`, {func}`~pyqsc_jax.optimal_B2c_value` | exact $B_{2c}$ minimizing the $B_{20}$ variance | {doc}`../theory/field_jet` |
| {func}`~pyqsc_jax.second_order_residuals` | the four second-order equations evaluated independently | {doc}`../theory/second_order` |
| {func}`~pyqsc_jax.plasma_field_on_axis`, {func}`~pyqsc_jax.plasma_gradient_on_axis`, {func}`~pyqsc_jax.plasma_hessian_on_axis` | plasma field jet and external target | {doc}`../theory/plasma_field` |

`singularity_diagnostics` takes `angular_resolution` (256) and `newton_iterations` (8);
`b20_diagnostics` takes the power `smooth_maximum_power` (16) of its smooth maximum. The
plasma functions take the required `formal_radius` and `angular_resolution` (128).

Remainder estimates (`estimated_field_remainder`, `estimated_hessian_remainder`) are
asymptotic scales, not error bounds. `r_singularity` bounds the validity of the truncated
coordinate map, not the existence of nested surfaces.

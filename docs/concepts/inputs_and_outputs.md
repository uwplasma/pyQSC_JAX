# Inputs and outputs

## Inputs

`Axis` accepts `rc`, `rs`, `zc`, `zs`, and integer `nfp`.
`Qsc`/`solve` normalize the physical inputs into `NearAxisInputs`:

| Name | Meaning |
| --- | --- |
| `axis` or `rc, rs, zc, zs, nfp` | magnetic-axis Fourier representation |
| `etabar` | signed first-order field-strength coefficient [m\(^{-1}\)] |
| `B0` | on-axis field strength [T] |
| `sigma0` | initial cross-section tilt |
| `I2` | quadratic covariant-current coefficient [T/m] |
| `p2` | quadratic pressure coefficient [Pa/m\(^2\)] |
| `B2c`, `B2s` | second-order field-strength harmonics [T/m\(^2\)] |
| `nphi` | samples per field period |
| `order` | `r1`, `r2`, or `r3` |
| `sG`, `spsi` | signs of \(G_0\) and toroidal flux |
| `iota`, `solve_for` | optional inverse target and solved parameter |

All physical scalar inputs must be scalar arrays. `nphi`, order, sign flags,
and the solved-parameter choice are static pytree metadata.

## Common outputs

Every `NearAxisSolution` contains `inputs`, `geometry`, `root_report`,
`sigma`, `iota`, `iotaN`, `helicity`, `G0`, the first-order coefficients
`X1s`, `X1c`, `Y1s`, `Y1c` and untwisted forms, `elongation`,
`mean_elongation`, `B_axis`, `grad_B_axis`, and `L_grad_B`.

An r2 result adds the complete `SecondOrderData`: `V1`, `V2`, `V3`,
`X20`, `X2s`, `X2c`, `Y20`, `Y2s`, `Y2c`, `Z20`, `Z2s`, `Z2c`, their
needed derivatives and untwisted forms, `beta_1s`, `G2`, `B20`,
`B20_mean`, `B20_anomaly`, `B20_residual`, and `B20_variation`. The r2
solution also attaches singular-radius diagnostics, Mercier terms, and the
total `FieldJet`.

`FieldJet` contains `field`, `gradient`, `hessian`, `hessian_frenet`,
coordinate-map derivatives, reconstruction errors, Maxwell residuals, and
\(L_{\nabla\nabla B}\). Canonical shapes are `(nphi, 3)`,
`(nphi, 3, 3)`, and `(nphi, 3, 3, 3)`, with field component before
derivative indices.

An r3 result adds the first- and third-poloidal-harmonic surface coefficients,
their untwisted forms, `B0_order_a_squared_to_cancel`,
`flux_constraint_residual`, and `consistency_error`. Explicit shear adds
`B31c`, `iota2`, and the intermediate `ShearData`.

Inverse, continuation, optimization, criteria, singularity, and plasma APIs
return their own immutable records with reports rather than overloading a
single mutable object. The API pages enumerate their public fields and the
[physics traceability table](../development/physics-traceability.md) maps
them to equations and tests.

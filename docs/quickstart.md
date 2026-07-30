# Quickstart

The canonical immutable entry point constructs a first-order quasisymmetric
solution:

```python
import pyqsc_jax as qsc

solution = qsc.Qsc(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
    nphi=31,
    order="r1",
)

print(solution.iota)
print(solution.root_report.residual_norm)
print(solution.axis_length)
```

`rc` and `zs` are the cosine coefficients of cylindrical axis radius and sine
coefficients of height for a stellarator-symmetric axis. `nfp` is the number
of field periods. `etabar` controls the first-order field-strength variation.

For a general asymmetric axis, construct `Axis` explicitly:

```python
axis = qsc.Axis(
    rc=[1.0, 0.04],
    rs=[0.0, 0.01],
    zc=[0.0, -0.015],
    zs=[0.0, -0.05],
    nfp=3,
)
solution = qsc.solve(axis=axis, etabar=-0.9, nphi=61)
```

The result is an immutable JAX pytree. Its vector samples have shape
`(nphi, 3)` and its tensor samples have shape `(nphi, 3, 3)`. Always inspect
`solution.root_report.converged` and the residual norm when accepting a
configuration.

ESSOS code can continue to use:

```python
from pyqsc_jax.near_axis import near_axis

field = near_axis(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
)
```

The adapter retains the legacy `(3, nphi)` field and `(3, 3, nphi)` gradient
orientations plus mutable `x`/`dofs`; it delegates calculations to the same
immutable core.

## Finite-pressure/current second order

Set `order="r2"` to solve the coupled periodic second-order coefficient system:

```python
solution = qsc.Qsc(
    rc=[1.0, 0.09],
    zs=[0.0, -0.09],
    nfp=2,
    etabar=0.95,
    I2=0.9,
    p2=-600000.0,
    B2c=-0.7,
    nphi=31,
    order="r2",
)

print(solution.B20_mean)
print(solution.B20_residual)
print(solution.linear_report.matrix_condition_number)
print(solution.DMerc_times_r2)
print(solution.field_jet.maximum_divergence_gradient)
print(solution.grad_grad_B_inverse_scale_length)
print(solution.r_singularity)
```

The direct r2 outputs are available as attributes such as `X20`, `X2s`,
`Y20`, `Z20`, `beta_1s`, `G2`, and `B20`. The immutable nested record is also
available as `solution.second_order`. Check both the nonlinear
`root_report` and r2 `linear_report` before accepting a result. The canonical
Hessian `solution.grad_grad_B_axis` has shape `(nphi, 3, 3, 3)` and
field-component-first ordering. `solution.r_singularity` is the first
quadratic-map coordinate singularity, not an equilibrium-existence bound.

## Third-order flux constraint

Set `order="r3"` to add the surface correction required by the
order-\(r^2\) toroidal-flux constraint:

```python
solution = qsc.Qsc(
    rc=[1.0, 0.155, 0.0102],
    zs=[0.0, 0.154, 0.0111],
    nfp=2,
    etabar=0.64,
    B2c=-0.00322,
    nphi=61,
    order="r3",
)

print(solution.flux_constraint_residual)
print(solution.consistency_error)
print(solution.B0_order_a_squared_to_cancel)
```

The r3 result supplies first- and third-poloidal-harmonic coefficient arrays
and their untwisted forms to boundary conversion. Magnetic shear is a
separate, later milestone.

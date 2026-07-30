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

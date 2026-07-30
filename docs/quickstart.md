# Quickstart

The current compatibility entry point constructs the standard first-order
quasisymmetric solution:

```python
from pyqsc_jax.near_axis import near_axis

field = near_axis(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
    nphi=31,
)

print(field.iota)
print(field.axis_length)
```

`rc` and `zs` are the cosine coefficients of cylindrical axis radius and sine
coefficients of height for a stellarator-symmetric axis. `nfp` is the number
of field periods. `etabar` controls the first-order field-strength variation.

The new immutable `Axis`, `Qsc`, and `solve` API will replace this page after
its validation gates are complete. The legacy import will remain as an
adapter for ESSOS.

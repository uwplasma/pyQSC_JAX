# Quickstart

The canonical entry point returns an immutable JAX pytree:

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

assert solution.root_report.converged
print("iota:", solution.iota)
print("residual:", solution.root_report.residual_norm)
print("axis length:", solution.axis_length)
print("minimum L_grad_B:", solution.L_grad_B.min())
```

`rc` and `zs` are Fourier coefficients of cylindrical axis radius and height,
`nfp` is the field-period count, and `etabar` sets the leading
field-strength variation. Start with [choosing a model](choosing_a_model.md)
before requesting second- or third-order outputs.

The complete executable version is included directly from the repository:

```{literalinclude} ../../examples/01_first_order_qa.py
:language: python
:linenos:
```

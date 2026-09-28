# Automatic differentiation

The immutable solution records are registered JAX pytrees. Root and linear
solves differentiate converged equations implicitly instead of
backpropagating through iteration histories. Functions intended for
transformation accept physical arrays explicitly; static grid sizes and model
choices remain Python metadata.

```{literalinclude} ../../examples/12_autodiff_check.py
:language: python
:linenos:
```

The test suite checks eager/JIT parity, VMAP batches, JVP/VJP consistency, and
finite differences for axis geometry, first and second order, inverse solves,
optimization, and plasma jets. Nonsmooth decisions such as basin selection
are outside a local derivative; differentiate a selected candidate's smooth
residual, not the discrete search procedure.

# Second-order finite beta and current

This example activates pressure, current, and the complete r2 system. It
prints nonlinear and linear residuals, the matrix condition number, Mercier
quantity, \(B_{20}\) residual, and singular radius.

```{literalinclude} ../../examples/03_second_order_finite_beta.py
:language: python
:linenos:
```

The demonstrated configuration is a regression case, not an optimized stable
stellarator. A negative Mercier value is therefore reported rather than
hidden.

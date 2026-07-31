# First-order QA

This example constructs the standard three-field-period quasi-axisymmetric
case, prints convergence evidence and geometry metrics, and saves an axis
plot. Change parameters only at the top of the script.

```{literalinclude} ../../examples/01_first_order_qa.py
:language: python
:linenos:
```

The reported transform is positive for the chosen negative `etabar` branch.
Always check `root_report.converged` and the residual before using derived
quantities in an objective.

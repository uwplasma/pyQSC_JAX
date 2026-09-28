# Total field jet

Computes the on-axis field, gradient and Hessian of the section 5.1 configuration with
{func}`~pyqsc_jax.total_field_jet` and prints the Maxwell checks: agreement with the
first-order field and gradient, $\nabla\cdot\mathbf B$, derivative-index symmetry and
$\nabla(\nabla\cdot\mathbf B)$ ({doc}`../theory/field_jet`).

```{figure} ../_static/figures/example_09.png
:width: 75%
:alt: Cartesian field components (top) and three Hessian components (bottom).

Cartesian field components (top) and three Hessian components (bottom).
```

Output:

```{literalinclude} output/09.txt
:language: text
```

```{literalinclude} ../../examples/09_total_field_jet.py
:language: python
:caption: examples/09_total_field_jet.py
```

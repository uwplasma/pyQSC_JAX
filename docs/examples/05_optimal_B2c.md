# Exact B2c elimination

For the section 5.1 axis of {cite:t}`landreman2019highorder`, $B_{20}$ is affine in
$B_{2c}$, so the $B_{2c}$ that minimizes the weighted variance of $B_{20}$ follows from two
linear solves. {func}`~pyqsc_jax.optimize_B2c` returns it, the recomputed solution and the
affine reconstruction error ({doc}`../theory/field_jet`).

```{figure} ../_static/figures/example_05.png
:width: 75%
:alt: B_{20} anomaly before and after the exact B_{2c} elimination.

$B_{20}$ anomaly before and after the exact $B_{2c}$ elimination.
```

Output:

```{literalinclude} output/05.txt
:language: text
```

```{literalinclude} ../../examples/05_optimal_B2c.py
:language: python
:caption: examples/05_optimal_B2c.py
```

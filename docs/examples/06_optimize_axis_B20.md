# Nearly constant B20

Compares database configuration 57409 {cite}`curvo2025deep` after exact $B_{2c}$
elimination with an eight-mode axis refined from it by bounded least squares on the
$B_{20}$ anomaly (the refined coefficients are written in the script; the optimizer
pattern is in {doc}`../user_guide/differentiation`). The weighted $B_{20}$ residual drops by
about $2.4\times10^8$.

```{figure} ../_static/figures/example_06.png
:width: 75%
:alt: B_{20} anomaly of the database axis (left) and the refined axis (right).

$B_{20}$ anomaly of the database axis (left) and the refined axis (right).
```

Output:

```{literalinclude} output/06.txt
:language: text
```

```{literalinclude} ../../examples/06_optimize_axis_B20.py
:language: python
:caption: examples/06_optimize_axis_B20.py
```

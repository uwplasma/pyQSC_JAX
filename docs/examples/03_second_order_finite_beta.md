# Second order at finite pressure

Database configuration 52521 {cite}`curvo2025deep`: four field periods, strongly
nonplanar axis, finite $p_2$, $I_2 = 0$. The script solves the complete second-order
system and prints the linear-solve report, the $B_{20}$ residual, the Mercier criterion and
the singular radius, then plots $B_{20} - \langle B_{20}\rangle$
({doc}`../theory/second_order`).

```{figure} ../_static/figures/example_03.png
:width: 75%
:alt: B_{20} anomaly over one field period.

$B_{20}$ anomaly over one field period.
```

Output:

```{literalinclude} output/03.txt
:language: text
```

```{literalinclude} ../../examples/03_second_order_finite_beta.py
:language: python
:caption: examples/03_second_order_finite_beta.py
```
